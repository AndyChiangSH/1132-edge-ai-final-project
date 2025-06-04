import gc
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm.auto import tqdm
from datasets import load_dataset
import random
import numpy as np
from hqq.core.quantize import BaseQuantizeConfig
from hqq.utils.patching import prepare_for_inference
from hqq_utils import AutoHQQHFModel

#####################################################################
# === SPEC NOTICE ===
# Only "load model" and "generate" function selection can be modified.
# DO NOT change PPL calculation, timing, or throughput logic.
#####################################################################

# === 极致优化的 generate 函数 ===
def generate(model, input_ids, past_key_values, max_new_tokens):
    input_ids = input_ids.clone()
    batch_size, seq_len = input_ids.shape
    
    # 预分配输出tensor，避免反复cat操作
    output_ids = torch.empty((batch_size, seq_len + max_new_tokens), 
                            dtype=input_ids.dtype, device=input_ids.device)
    output_ids[:, :seq_len] = input_ids
    
    with torch.no_grad():
        # === Prefill Phase - 直接传入完整参数 ===
        attention_mask = torch.ones_like(input_ids)
        position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        output_ids[:, seq_len] = next_token.squeeze(-1)

        # === Autoregressive Decoding Phase - 优化循环 ===
        for step in range(1, max_new_tokens):
            current_pos = seq_len + step - 1
            
            # 使用更简化的调用
            outputs = model(
                input_ids=next_token,
                past_key_values=past_key_values,
                use_cache=True,
            )
            
            logits = outputs.logits.squeeze(1)  # 直接squeeze
            next_token = torch.argmax(logits, dim=-1, keepdim=True)
            output_ids[:, seq_len + step] = next_token.squeeze(-1)
            past_key_values = outputs.past_key_values

    return output_ids

def evaluate_ppl(model, tokenizer, device="cuda:0"):
    test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
    model.seqlen = 2048
    test_enc = test_enc.input_ids.to(device)
    
    nsamples = test_enc.numel() // model.seqlen
    nlls = []  
    for i in tqdm(range(nsamples), desc="Evaluating..."):
        batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
        
        with torch.no_grad():
            lm_logits = model(batch).logits

        shift_logits = lm_logits[:, :-1, :].contiguous().float()
        shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    
    return ppl.item()

# === T4 GPU 优化量化配置 ===
def get_quant_config_slm(model):
    quant_config = {}
    n_layers = model.config.num_hidden_layers
    
    # T4 优化：使用64作为group_size，因为T4的Tensor Core对这个尺寸优化更好
    group_size = 64
    
    for i in range(n_layers):
        ratio = i / n_layers
        
        # T4 特化量化策略：
        # T4 在4bit上有很好的支持，2bit性能反而可能不如4bit
        # 因为T4没有专门的2bit Tensor Core支持
        if ratio < 0.2:  # 前20%层保持较高精度
            bits_qkv, bits_proj, bits_mlp = 4, 4, 4
        elif ratio < 0.6:  # 中间40%层使用4bit（T4最优）
            bits_qkv, bits_proj, bits_mlp = 4, 4, 4
        else:  # 后40%层可以用3bit，但避免2bit和1bit
            bits_qkv, bits_proj, bits_mlp = 4, 4, 4
            
        # T4对4bit INT8有很好的支持，所以MLP层保持4bit
        # 避免使用1bit或2bit，这些在T4上性能不佳
        # if ratio > 0.8:  # 只有最后20%层的MLP用3bit
        #     bits_mlp = 3
        # else:
        #     bits_mlp = 4
            
        q2_config_attn1 = BaseQuantizeConfig(nbits=bits_qkv, group_size=group_size)
        q2_config_attn2 = BaseQuantizeConfig(nbits=bits_proj, group_size=group_size)
        q2_config_mlp = BaseQuantizeConfig(nbits=bits_mlp, group_size=group_size)
        
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q2_config_attn1
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q2_config_attn1
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q2_config_attn1
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q2_config_attn2
        
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q2_config_mlp
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q2_config_mlp
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q2_config_mlp

    return quant_config

def main():
    # T4优化的内存配置
    import os
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:256'  # T4适中的分块大小
    os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # 异步执行
    
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()
    
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)
    
    max_new_tokens = 256
    device = 'cuda:0'
    backend = 'gemlite'
    
    # T4 GPU 特化优化
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # T4 特化：Flash Attention可能在T4上效果不如传统attention
    # T4的compute capability是7.5，Flash Attention在更新的GPU上效果更好
    torch.backends.cuda.enable_flash_sdp(False)  # T4上可能不是最优
    torch.backends.cuda.enable_math_sdp(True)    # T4上传统数学实现可能更好
    torch.backends.cuda.enable_mem_efficient_sdp(True)  # 内存效率优化保留
    
    ### === 加载模型 - 使用最小内存配置 ===
    model_name = "../llama3.2-3B-instruct_lora_bf16"   
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        trust_remote_code=True,
    )
    
    model.eval()
    
    # 关闭不必要的功能
    for param in model.parameters():
        param.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 设置模型为推理优化模式
    model.config.use_cache = True
    model.config.output_attentions = False
    model.config.output_hidden_states = False
    
    print("Quantizing model with extreme settings...")
    quant_config = get_quant_config_slm(model)
    AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float16, device=device)

    prepare_for_inference(model, backend=backend)
    
    # T4优化的编译设置
    print("Compiling with T4-optimized settings...")
    model = torch.compile(
        model, 
        mode="reduce-overhead",  # T4上reduce-overhead通常比max-autotune更稳定
        fullgraph=True,
        dynamic=False
    )
    
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    # 使用动态cache而不是static cache，可能更快
    past_key_values = None
    
    # 最小warmup - 只做必要的编译warmup
    print("Minimal warmup...")
    warmup_prompt = "Hi"
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    
    # 只做2次warmup，但涵盖不同代码路径
    for i in range(2):
        with torch.no_grad():
            _ = model.generate(
                input_ids=input_ids,
                max_new_tokens=16,  # 很短的生成
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
    
    # 主要测试 - 使用模型自带的generate而不是自定义函数
    prompt = "How to learn a new language?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    tputs = []
    time_record = []
    
    print("Starting inference tests...")
    for i in tqdm(range(10), desc="Test Inference"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # 使用模型原生generate，通常比自定义函数更优化
        with torch.no_grad():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # 确保使用贪心解码
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,  # 确保是贪心解码
            )

        end.record()
        torch.cuda.synchronize()
        elapsed_ms = start.elapsed_time(end)
        tput = generated[0][input_ids.shape[1]:].shape[0]/(elapsed_ms / 1000)
        time_record.append(elapsed_ms / 1000)
        tputs.append(tput)
        
    response = tokenizer.decode(generated[0][input_ids.shape[1]:], skip_special_tokens=True)
    sorted_tputs = np.sort(tputs)[2:-2]
    org_tput = np.mean(sorted_tputs)
    print(f'Prompt: {prompt}\nResponse: {response}\n')
    
    print(f'Time Record: {time_record}')
    print(f'Throughput Record: {tputs} toks/s\n')

    ### Your final throughput result ###
    print(f'Throughput: {org_tput} toks/s')
    ppl = evaluate_ppl(model, tokenizer, device)
    print(f"Perplexity (PPL): {ppl}")
    
    # Save results to CSV
    import csv
    rounded_tput = round(org_tput, 1)
    ppl = round(ppl, 2)

    with open("result_0529_lora_v2.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "value"])
        writer.writerow([0, ppl])
        writer.writerow([1, rounded_tput])
        
if __name__ == '__main__':
    main()