import argparse
import time
import torch
import torch.distributed as dist
from tasks.medusa_llama.llama_config import LlamaConfig
from tools.utils import initialize_distributed, get_max_memory, get_model_type
from tools.sot import get_skeleton_prompt, get_point_expanding_prompt
from jupiter.utils import jupiter_prefilling, normal_decoding, jupiter_prefilling_no_finish, point_prefilling, outline_based_decoding
from tasks.medusa_llama.outline_decoding_controller import get_controller, OutlineDecodingController, set_controller

def main(args):
    # --- 1. 模型初始化 ---
    if get_model_type(args.config_file) == 'vicuna_7b' or get_model_type(args.config_file) == 'vicuna_13b':
        config = LlamaConfig.from_pretrained(args.config_file)
        temp_path = "temp_{}_world_{}_rank_{}/stage.bin".format(get_model_type(args.config_file), args.world, args.rank)
        from tasks.medusa_llama.medusa_llama_pp import PPMedusaLlamaForCausalLM as PPMedusaModel
    else:
        raise NotImplementedError("NotImplementedError")
    
    if args.rank == 0:
        print("temp_path:", temp_path, flush=True)
    
    initialize_distributed(config, args)
    config.update_pp_stage_config(args)

    # 加载模型
    if config.device == "cuda":
        with torch.device("cuda"):
            model = PPMedusaModel.from_pretrained(
                pretrained_model_name_or_path=temp_path,
                config=config,
                use_safetensors=False,
                torch_dtype=config.torch_dtype,
                load_in_4bit=args.load_in_4bit,
                load_in_8bit=args.load_in_8bit
            )
    else:
        model = PPMedusaModel.from_pretrained(
            pretrained_model_name_or_path=temp_path,
            config=config,
            use_safetensors=False,
            torch_dtype=config.torch_dtype,
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit
        )
    model.eval()
    if args.rank == 0:
        print("Data_type:", model.dtype, flush=True)

    if not args.load_in_8bit and not args.load_in_4bit:
        model = model.to(config.device)
    tokenizer = model.tokenizer

    # --- 2. 准备输入 ---
    question = "大模型的优缺点是什么？"
    
    # [关键步骤] 在开始计时前，强制同步所有显卡，确保大家在同一起跑线
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    dist.barrier()

    # [开始计时]
    start_time = time.time()

    prompt = get_skeleton_prompt(question)

    # Step 1: Prefilling (生成骨架)
    medusa_logits, logits = jupiter_prefilling(tokenizer.encode(prompt, return_tensors="pt"), model, config, args)
    dist.barrier()

    # Step 2: Normal decoding (解码骨架)
    answer = normal_decoding(prompt, model, config, medusa_logits, logits)
    
    skeleton = "\n".join([line.lstrip() for line in answer.splitlines()])
    
    if args.rank == 0:
        print("===========================================\n", flush=True)
        print("Skeleton:\n", skeleton, flush=True)

    points, shared_perfix, prompts_for_points = get_point_expanding_prompt(skeleton, question)

    if args.rank == 0:
        print("===========================================\n", flush=True)
        print("Shared perfix:\n", shared_perfix, flush=True)
        print("prompts_for_points: ", flush=True)
        for i in prompts_for_points:
            print(i, flush=True)

    # Step 3: Shared prefix prefilling
    input_ids_1 = tokenizer.encode(shared_perfix, return_tensors="pt")
    jupiter_prefilling_no_finish(input_ids_1, model, config, args)
    dist.barrier()

    # Step 4: Point request prefilling
    set_controller(OutlineDecodingController(points, config, model))
    if args.rank == 0:
        print("==============================\n point_prefilling")
    
    medusa_logits_list, logits_list = point_prefilling(prompts_for_points, model, config, args)
    dist.barrier()

    # 准备 Request
    if config.is_last_stage:
        get_controller().add_requests(medusa_logits_list, logits_list)
    
    # 准备 Input IDs
    input_ids_for_point = []
    for i in range(len(prompts_for_points)):
        input_ids_1 = tokenizer.encode(shared_perfix, return_tensors="pt")
        input_ids_2 = tokenizer.encode(prompts_for_points[i], return_tensors="pt")
        input_ids = torch.cat([input_ids_1, input_ids_2[:, 2:]], dim=1)
        if config.device == "cuda":
            input_ids = input_ids.cuda()
        input_ids_for_point.append(input_ids)
    
    get_controller().set_up_input_ids_for_point(input_ids_for_point)
    dist.barrier()

    # Step 5: Jupiter decoding (并行解码)
    if args.rank == 0:
        print("==============================\n outline_based_decoding")
    
    outline_based_decoding(model, config, args)
    
    # 等待并处理最终输出
    get_controller().get_output()
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--world', default=2, type=int)
    parser.add_argument("--config_file", type=str, default="config/vicuna_7b_config.json", help="Model name or path.")
    parser.add_argument("--load_in_8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--load_in_4bit", action="store_true", help="Use 4-bit quantization")
    args = parser.parse_args()

    main(args)