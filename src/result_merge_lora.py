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

# === (Optional) Define your own custom generate function. ===
# This is useful if you want full control over KV cache and generation steps.
# You can modify this function to suit your needs.
# By default, we use model.generate() for simplicity and general use.
def generate(model, input_ids, past_key_values, max_new_tokens):
    input_ids = input_ids.clone()
    
    # Set static cache mode and compilation config
    compiled_model = model
    compiled_model.generation_config.cache_implementation = "static"
    #compile using kv-cache
    compiled_model.generation_config.max_cache_size = input_ids.shape[1] + max_new_tokens + 16

    # Compile the forward pass once for improved performance
    if not hasattr(model, "_compiled_forward"):
        compiled_model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
    

    with torch.no_grad():
        # === Prefill Phase ===
        outputs = model.prefill_forward(
            input_ids=input_ids,
            past_key_values=past_key_values,
            position_ids=None,
            attention_mask=None,
            cache_position=None,
            logits_to_keep=1,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_token], dim=-1)

        # === Autoregressive Decoding Phase ===
        for step in range(max_new_tokens - 1):
            pos = input_ids.shape[1] - 1
            cache_position = torch.arange(pos, pos + 1, device=input_ids.device, dtype=torch.long)


            outputs = compiled_model(
                next_token,
                past_key_values=past_key_values,
                position_ids=cache_position.unsqueeze(0),
                cache_position=cache_position
            )
            logits = outputs.logits
            next_token = torch.argmax(logits, dim=-1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            past_key_values = outputs.past_key_values

    return input_ids

# def generate(model, input_ids, past_key_values, max_new_tokens):
#     input_ids = input_ids.clone()
    
#     # 設置靜態快取模式 - 不需要重新編譯，因為模型已經在 main() 中編譯過了
#     model.generation_config.cache_implementation = "static"
#     model.generation_config.max_cache_size = input_ids.shape[1] + max_new_tokens + 16

#     with torch.no_grad():
#         # === Prefill Phase ===
#         outputs = model.prefill_forward(
#             input_ids=input_ids,
#             past_key_values=past_key_values,
#             position_ids=None,
#             attention_mask=None,
#             cache_position=None,
#             logits_to_keep=1,
#         )
#         past_key_values = outputs.past_key_values
#         logits = outputs.logits[:, -1, :]
#         next_token = torch.argmax(logits, dim=-1, keepdim=True)
#         input_ids = torch.cat([input_ids, next_token], dim=-1)

#         # === Autoregressive Decoding Phase ===
#         for step in range(max_new_tokens - 1):
#             pos = input_ids.shape[1] - 1
#             cache_position = torch.arange(pos, pos + 1, device=input_ids.device, dtype=torch.long)

#             # 直接使用已編譯的模型，不需要重新編譯
#             outputs = model(
#                 next_token,
#                 past_key_values=past_key_values,
#                 position_ids=cache_position.unsqueeze(0),
#                 cache_position=cache_position
#             )
#             logits = outputs.logits
#             next_token = torch.argmax(logits, dim=-1)
#             input_ids = torch.cat([input_ids, next_token], dim=-1)
#             past_key_values = outputs.past_key_values

#     return input_ids
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

# TODO: Make your own quant config for Language Model
def get_quant_config_slm(model):
    quant_config = {}
    
    n_layers = model.config.num_hidden_layers
    q_config = BaseQuantizeConfig(nbits=4, group_size=512)
    
    for i in range(n_layers):
        quant_config[f'model.layers.{i}.self_attn.q_proj'] = q_config
        quant_config[f'model.layers.{i}.self_attn.k_proj'] = q_config
        quant_config[f'model.layers.{i}.self_attn.v_proj'] = q_config
        quant_config[f'model.layers.{i}.self_attn.o_proj'] = q_config
        
        quant_config[f'model.layers.{i}.mlp.gate_proj'] = q_config
        quant_config[f'model.layers.{i}.mlp.up_proj'] = q_config
        quant_config[f'model.layers.{i}.mlp.down_proj'] = q_config
    # group_size = 64
    # for i in range(n_layers):
    #     ratio = i / n_layers
    #     if ratio < 0.5:
    #         bits_qkv, bits_proj, bits_mlp = 4, 4, 4
    #     else:
    #         bits_qkv, bits_proj, bits_mlp = 4, 4, 4
    #     q2_config_attn1 = BaseQuantizeConfig(nbits=bits_qkv, group_size=group_size)
    #     q2_config_attn2 = BaseQuantizeConfig(nbits=bits_proj, group_size=group_size)
    #     q2_config_mlp = BaseQuantizeConfig(nbits=bits_mlp, group_size=group_size)
    #     # quant_config[f'blocks.{i}.attn.qkv'] = q2_config_attn1
    #     # quant_config[f'blocks.{i}.attn.proj'] = q2_config_attn2
    #     # quant_config[f'blocks.{i}.mlp.fc1'] = q2_config_mlp
    #     # quant_config[f'blocks.{i}.mlp.fc2'] = q2_config_mlp
    #     quant_config[f'model.layers.{i}.self_attn.q_proj'] = q2_config_attn1
    #     quant_config[f'model.layers.{i}.self_attn.k_proj'] = q2_config_attn1
    #     quant_config[f'model.layers.{i}.self_attn.v_proj'] = q2_config_attn1
    #     quant_config[f'model.layers.{i}.self_attn.o_proj'] = q2_config_attn2
        
    #     quant_config[f'model.layers.{i}.mlp.gate_proj'] = q2_config_mlp
    #     quant_config[f'model.layers.{i}.mlp.up_proj'] = q2_config_mlp
    #     quant_config[f'model.layers.{i}.mlp.down_proj'] = q2_config_mlp

    return quant_config

def main():
    gc.collect()              # Python 層級垃圾回收
    torch.cuda.empty_cache()  # PyTorch 層級釋放未使用快取
    torch.cuda.ipc_collect()  # 清理 inter-process memory
    
    ############## Set Up ##############
    torch.manual_seed(0)
    random.seed(0)
    
    max_new_tokens = 256    # Number of new tokens to generate
    device = 'cuda:0'
    backend = 'gemlite'
    
    ### === TODO: Load your model (you may change this part) ===
    # model_name = "meta-llama/Llama-3.2-3B-Instruct"   
    model_name = "x21530317x/llama3.2-3B-instruct_lora_bf16"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map=device,
    )
    # model.generation_config.cache_implementation = "static"
    # model.forward = torch.compile(model.forward, mode="reduce-overhead", fullgraph=True)
    #####################################
    
    model.eval() 
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # === (Optional) Uncomment the following lines if using the custom generate() function. ===
    model.prefill_forward = model.forward

    # TODO: Quantize    
    quant_config = get_quant_config_slm(model)
    AutoHQQHFModel.quantize_model(model, quant_config=quant_config, compute_dtype=torch.float16, device=device)

    prepare_for_inference(model, backend=backend)
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    warmup_prompt = "Explain what AI is."
    inputs = tokenizer(warmup_prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    # === (Optional) Set up StaticCache for manual KV cache management ===
    from transformers import StaticCache
    past_key_values = StaticCache(
        config=model.config, 
        max_batch_size=1, 
        max_cache_len=max_new_tokens + 16, 
        device=model.device, 
        dtype=torch.float16
    )
    ####################################################################
    
    for i in tqdm(range(5), desc="Warm Up..."):
        #  === Default: use model.generate() for end-to-end warm-up === 
        # _ = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     pad_token_id=tokenizer.eos_token_id,
        # )
        
        # === (Optional) Use custom generate() if uncommented ===
        generated = generate(model, input_ids, past_key_values, max_new_tokens)
        past_key_values.reset()
        
    prompt = "How to learn a new language?"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    tputs = []
    time_record = []
    for _ in tqdm(range(10), desc="Test Inference"):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        # === Default: Use model.generate() for end-to-end timing === 
        # generated = model.generate(
        #     input_ids=input_ids,
        #     attention_mask=attention_mask,
        #     max_new_tokens=max_new_tokens,
        #     pad_token_id=tokenizer.eos_token_id,
        # )
        
        # === Optional: Use custom generate() if uncommented ===
        generated = generate(model, input_ids, past_key_values, max_new_tokens)
        past_key_values.reset()

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

    with open("result/result.csv", mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Id", "value"])
        writer.writerow([0, ppl])
        writer.writerow([1, rounded_tput])
        
if __name__ == '__main__':
    main()