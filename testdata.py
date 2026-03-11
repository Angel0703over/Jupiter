import argparse
import gc
import json
import os
import torch
import torch.distributed as dist
import threading
from tasks.medusa_llama.llama_config import LlamaConfig
from tools.utils import initialize_distributed, get_model_type
from tools.sot import get_skeleton_prompt, get_point_expanding_prompt
from jupiter.prefilling_pipeline import PrefillingPipeline
from jupiter.utils import (
    jupiter_prefilling,
    normal_decoding,
    jupiter_prefilling_no_finish,
    point_prefilling,
    outline_based_decoding,
)

from tasks.medusa_llama.outline_decoding_controller import (
    get_controller,
    OutlineDecodingController,
    set_controller,
    reset_controller,
)
from jupiter.prefilling_pipeline import PrefillingPipeline
from jupiter.decoding_pipeline import DecodingPipeline


# ------------------------------------------------
# Model Init
# ------------------------------------------------
def init(args, init_dist=True):

    global config
    global model
    global tokenizer

    if get_model_type(args.config_file) in ["vicuna_7b", "vicuna_13b"]:
        config = LlamaConfig.from_pretrained(args.config_file)
        temp_path = "temp_{}_world_{}_rank_{}/stage.bin".format(
            get_model_type(args.config_file),
            args.world,
            args.rank
        )
        from tasks.medusa_llama.medusa_llama_pp import (
            PPMedusaLlamaForCausalLM as PPMedusaModel
        )
    else:
        raise NotImplementedError
    if args.rank == 0:
        print("temp_path:", temp_path, flush=True)
    if init_dist:
        initialize_distributed(config, args)
    config.update_pp_stage_config(args)
    # load model
    if config.device == "cuda":
        with torch.device("cuda"):
            model = PPMedusaModel.from_pretrained(
                pretrained_model_name_or_path=temp_path,
                config=config,
                use_safetensors=False,
                torch_dtype=config.torch_dtype,
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit,
            )
    else:
        model = PPMedusaModel.from_pretrained(
            pretrained_model_name_or_path=temp_path,
            config=config,
            use_safetensors=False,
            torch_dtype=config.torch_dtype,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
        )
    model.eval()
    if not args.load_in_8bit and not args.load_in_4bit:
        model = model.to(config.device)
    tokenizer = model.tokenizer
    return model, config, args


# ------------------------------------------------
# Reset Context
# ------------------------------------------------
def reset_model_context(model, config):
    # reset model internal state
    if hasattr(model, "current_length_data"):
        model.current_length_data.zero_()
    if hasattr(model, "model"):
        model.model.medusa_mask = None
        model.model.medusa_mode = None
        # 清理每层 attention cache
    #     if hasattr(model.model, "layers"):
    #         for layer in model.model.layers:
    #             if hasattr(layer, "self_attn"):
    #                 attn = layer.self_attn
    #                 if hasattr(attn, "past_key_value"):
    #                     attn.past_key_value = None
    #                 if hasattr(attn, "point_past_key_value"):
    #                     attn.point_past_key_value = None

    #                 if hasattr(attn, "shared_past_key_value"):
    #                     attn.shared_past_key_value = None


    # if hasattr(model, "past_key_values"):
    #     model.past_key_values = None

    # if hasattr(model, "point_past_key_values"):
    #     model.point_past_key_values = None

    # release point controller caches between queries

    reset_controller()
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
    model.eval()
    if dist.is_initialized():
        print("rank", dist.get_rank(), "before barrier", flush=True)
        print(
            f"rank {dist.get_rank()} active threads: {threading.enumerate()}",
            flush=True
        )
        dist.barrier()
        print("rank", dist.get_rank(), "after barrier", flush=True)

# ------------------------------------------------
# Load Dataset
# ------------------------------------------------
def load_questions(dataset_path):
    questions = []
    with open(dataset_path, "r") as f:
        for line in f:
            data = json.loads(line)
            question = data["turns"][0]
            questions.append(question)
    return questions

# ------------------------------------------------
# Main Inference
# ------------------------------------------------
def run_single_query(question, model, config, args,pefilling_runtime):
    show_text_output = os.getenv("JUPITER_SHOW_TEXT_OUTPUT", "0") == "1"
    max_points = int(os.getenv("JUPITER_MAX_POINTS", "10"))

    print(f"\n==============================")
    print(f"[{args.rank}] Question: {question}", flush=True)
    prompt = get_skeleton_prompt(question)
    print("Start prefilling...", flush=True)
    medusa_logits, logits = jupiter_prefilling(
        pefilling_runtime,
        tokenizer.encode(prompt, return_tensors="pt"),
        model,
        config,
        args,
    )

    answer = normal_decoding(prompt, model, config, medusa_logits, logits)

    skeleton = "\n".join([line.lstrip() for line in answer.splitlines()])


    if show_text_output:
        print(f"[{args.rank}] skeleton:\n{skeleton}", flush=True)
    else:
        print(f"[{args.rank}] skeleton lines: {len(skeleton.splitlines())}", flush=True)

    points, shared_prefix, prompts_for_points = get_point_expanding_prompt(
        skeleton, question
    )
    points = points[:max_points]
    prompts_for_points = prompts_for_points[:max_points]

    if dist.is_initialized():
        src_rank = config.total_stage - 1
        payload = [None]
        if config.is_last_stage:
            payload[0] = (points, shared_prefix, prompts_for_points)
        dist.broadcast_object_list(payload, src=src_rank)
        points, shared_prefix, prompts_for_points = payload[0]

    print(f"[{args.rank}] point count: {len(points)}", flush=True)

    # shared prefix
    input_ids = tokenizer.encode(shared_prefix, return_tensors="pt")
    jupiter_prefilling_no_finish(pefilling_runtime,input_ids, model, config, args)

    dist.barrier()

    set_controller(OutlineDecodingController(points, config, model))

    medusa_logits_list, logits_list = point_prefilling(
        pefilling_runtime, prompts_for_points, model, config, args
    )

    dist.barrier()

    if config.is_last_stage:
        get_controller().add_requests(medusa_logits_list, logits_list)

    input_ids_for_point = []

    for p in prompts_for_points:

        ids1 = tokenizer.encode(shared_prefix, return_tensors="pt")
        ids2 = tokenizer.encode(p, return_tensors="pt")

        input_ids = torch.cat([ids1, ids2[:, 2:]], dim=1)
        if config.device == "cuda":
            input_ids = input_ids.cuda()

        input_ids_for_point.append(input_ids)

    get_controller().set_up_input_ids_for_point(input_ids_for_point)

    dist.barrier()

    print("outline_based_decoding begin", flush=True)

    outline_based_decoding(model, config, args)

    print("outline_based_decoding done", flush=True)

    if show_text_output:
        get_controller().get_output()
    else:
        print(f"[{args.rank}] skip expanded text output (set JUPITER_SHOW_TEXT_OUTPUT=1 to enable)", flush=True)
    dist.barrier()

# ------------------------------------------------
# Main
# ------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--world", default=2, type=int)

    parser.add_argument(
        "--config_file",
        type=str,
        default="config/vicuna_7b_config.json",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="question.jsonl",
    )

    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")
    args = parser.parse_args()
    model, config, args = init(args)
    prefilling_runtime = PrefillingPipeline(model, config, args)
    questions = load_questions(args.dataset)
    print(f"[rank {args.rank}] dataset size: {len(questions)}", flush=True)
    for i, q in enumerate(questions):
        run_single_query(q, model, config, args, prefilling_runtime)
        print(f"[{args.rank}] Reset model context...", flush=True)
        reset_model_context(model, config)
    prefilling_runtime.comm_handler.stop_helper_threads()
    print(f"[{args.rank}] All queries finished!")
