import argparse
import time
import torch
import torch.distributed as dist
import threading
from tasks.medusa_llama.llama_config import LlamaConfig
from tools.utils import initialize_distributed,get_max_memory,get_model_type
from tools.sot import get_skeleton_prompt,get_point_expanding_prompt
from jupiter.utils import jupiter_prefilling,normal_decoding,jupiter_prefilling_no_finish,point_prefilling,outline_based_decoding
from  tasks.medusa_llama.outline_decoding_controller  import get_controller,OutlineDecodingController,set_controller  #[modified]
from jupiter.decoding_pipeline import DecodingPipeline

def init(args, init_dist=True):
    global config
    global model
    global tokenizer
    # --- 1. 模型初始化 ---
    if get_model_type(args.config_file) == 'vicuna_7b' or get_model_type(args.config_file) == 'vicuna_13b':
        config = LlamaConfig.from_pretrained(args.config_file)
        temp_path = "temp_{}_world_{}_rank_{}/stage.bin".format(get_model_type(args.config_file), args.world, args.rank)
        from tasks.medusa_llama.medusa_llama_pp import PPMedusaLlamaForCausalLM as PPMedusaModel
    else:
        raise NotImplementedError("NotImplementedError")
    
    if args.rank == 0:
        print("temp_path:", temp_path, flush=True)
    
    if init_dist:
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
    return model,config,args

def reset_model_context(model, config):
    print(f"{args.rank}: reset model start")
    """
    重置模型的内部上下文和控制器状态
    """
    # 1. 重置模型主 KV cache
    if hasattr(model, 'current_length_data'):
        model.current_length_data.zero_()
    
    # 2. 重置 Medusa 模式
    if hasattr(model, 'model'):
        model.model.medusa_mask = None
        model.model.medusa_mode = None
    
    # 3. 重置控制器的单例实例
    OutlineDecodingController._instance = None  
    
    # 4. 确保模型处于 eval 模式
    model.eval()
    if dist.is_initialized():
        print("rank", dist.get_rank(), "before barrier", flush=True)
        print(
            f"rank {dist.get_rank()} active threads: {threading.enumerate()}",
            flush=True
        )
        dist.barrier()
        print("rank", dist.get_rank(), "after barrier", flush=True)
    print(f"{args.rank}: reset model done")
def main(question, model, config, args):
    #question = "What are the most effective ways to deal with stress?"
    print(f"--------\n{args.rank}: Start to process question: {question}", flush=True)
    # print(f"{args.rank}: reset model start")
    # reset_model_context(model, config)
    # print(f"{args.rank}: reset model done")


    prompt = get_skeleton_prompt(question)
    print("prompt: ", prompt, flush=True)
    print("===[FINISH]===", flush=True)
    # Step 1: prefilling with sequence slicing
    #if dist.is_initialized():
        #print("waiting for barrier", flush=True)
        #dist.barrier()
        #print("waiting for barrier done", flush=True)
    print("Start to prefilling...", flush=True)
    medusa_logits, logits = jupiter_prefilling(tokenizer.encode(prompt, return_tensors="pt"),model,config,args)
    print(f"[{args.rank}]: jupiter_prefilling done!", flush=True)
    # # Step 2: normal decoding
    answer = normal_decoding(prompt,model,config,medusa_logits,logits)
    skeleton = "\n".join([line.lstrip() for line in answer.splitlines()])
    print(f"[{args.rank}]: normal_decoding done!", flush=True)
    print(f"[{args.rank}] answer =", answer, flush=True)
    print("===========================================\n", flush=True)
    print("Skeleton:\n", skeleton, flush=True)
    points,shared_perfix,prompts_for_points = get_point_expanding_prompt(skeleton, question)
    print("===========================================\n", flush=True)
    print("Shared perfix:\n",shared_perfix, flush=True)
    print("===========================================\n", flush=True)
    print("prompts_for_points: ", flush=True)
    for i in prompts_for_points:
        print(i, flush=True)
    # Step 3: shared perfix prefiling
    input_ids_1 = tokenizer.encode(shared_perfix, return_tensors="pt")
    jupiter_prefilling_no_finish(input_ids_1 ,model,config,args)
    print(f"{args.rank}: jupiter_prefilling_no_finish done!", flush=True)
    dist.barrier()
    print(f"{args.rank}: dist barrier done!", flush=True)
    # Step 4: point request prefiling, and get medusa_logits, logits for every point 
    set_controller(OutlineDecodingController(points,config,model))
    print(f"{args.rank}: set_controller", flush=True)
    medusa_logits_list,logits_list = point_prefilling(prompts_for_points ,model,config,args )
    print(f"{args.rank}: point_prefilling", flush=True)
    dist.barrier()
    print(f"{args.rank}: dist barrier done!", flush=True)
    # prepare reuquets for every point
    if config.is_last_stage:
        get_controller().add_requests(medusa_logits_list,logits_list)
    print(f"{args.rank}: get_controller done!", flush=True)
    # prepare input_ids for every point
    input_ids_for_point=[]
    for i in range(len(prompts_for_points)): 
        input_ids_1 = tokenizer.encode(shared_perfix, return_tensors="pt")
        input_ids_2 = tokenizer.encode(prompts_for_points[i], return_tensors="pt")   
        input_ids = torch.cat([input_ids_1, input_ids_2[:,2:] ], dim=1)
        if config.device == "cuda":
            input_ids = input_ids.cuda()
        input_ids_for_point.append(input_ids)
    get_controller().set_up_input_ids_for_point(input_ids_for_point)
    print(f"{args.rank}: set_up_input_ids_for_point done!", flush=True)
    dist.barrier()
    print(f"{args.rank}: dist barrier done!", flush=True)
    # Step 5: jupiter decoding
    print("==============================\n outline_based_decoding begin")
    outline_based_decoding(model,config,args)
    print("==============================\n outline_based_decoding done!")
    get_controller().get_output( ) 
    print(f"{args.rank}: Done!")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--world', default=2, type=int)
    parser.add_argument("--config_file", type=str, default="config/vicuna_7b_config.json", help="Model name or path.")
    parser.add_argument(
        "--load_in_8bit", action="store_true", help="Use 8-bit quantization"
    )
    parser.add_argument(
        "--load_in_4bit", action="store_true", help="Use 4-bit quantization"
    )
    args = parser.parse_args()

    model,config,args = init(args)

    q1 = "What are the most effective ways to deal with stress?"
    main(q1, model, config, args)
    print(f"[{args.rank}] Resetting model context for next query...", flush=True)
    reset_model_context(model, config)
    # model,config,args = init(args, init_dist=False)
    q2 = "What are the most effective ways to deal with stress?"
    main(q2, model, config, args)
    print(f"[{args.rank}] Resetting model context for next query...", flush=True)
    q3 = q2
    reset_model_context(model, config)
    main(q3, model, config, args)
    reset_model_context(model, config)
    main(q3, model, config, args)
    print(f"{args.rank}: main exit!")
    exit(0)