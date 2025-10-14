import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import argparse

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Benchmark a model on specified or all GPUs")
parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda:0', 'cuda:1'). If not specified, uses all available GPUs.")
args = parser.parse_args()

# Config
model_name = "microsoft/DialoGPT-medium"
input_length = 512
output_length = 128
num_runs = 20
cooldown_sec = 5

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

if args.device:
    device = args.device
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    low_cpu_mem_usage=True
).to(device)
model.eval()

prompt_text = "Explain quantum computing in simple terms. " * 50
inputs = tokenizer(
    prompt_text, return_tensors="pt", truncation=True, max_length=input_length
).to(device)

# Warmup
with torch.no_grad():
    for _ in range(5):
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
torch.cuda.synchronize()

ttft_list = []
tps_list = []

for i in range(num_runs):
    torch.cuda.empty_cache()
    
    # --- TTFT: Prompt eval + first token (full forward) ---
    start_event = torch.cuda.Event(enable_timing=True)
    mid_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True, use_cache=True)  # Prompt eval, get KV cache
        mid_event.record()
        
        # First token from logits
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
    end_event.record()
    torch.cuda.synchronize()
    
    prompt_time = start_event.elapsed_time(mid_event) / 1000.0
    first_token_time = mid_event.elapsed_time(end_event) / 1000.0
    ttft = prompt_time + first_token_time
    ttft_list.append(ttft)
    
    # --- TPS: Decode remaining tokens reusing KV cache ---
    past_key_values = outputs.past_key_values
    generated_tokens = 1  # Already have first
    decode_start = torch.cuda.Event(enable_timing=True)
    decode_start.record()
    
    current_input = next_token
    with torch.no_grad():
        while generated_tokens < output_length:
            outputs = model(
                input_ids=current_input,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
            current_input = next_token
            past_key_values = outputs.past_key_values
            generated_tokens += 1
    decode_end = torch.cuda.Event(enable_timing=True)
    decode_end.record()
    torch.cuda.synchronize()
    
    decode_time = decode_start.elapsed_time(decode_end) / 1000.0
    decode_tokens = output_length - 1  # Exclude first token (measured in TTFT)
    tps = (output_length) / (first_token_time + decode_time) if (first_token_time + decode_time) > 0 else 0  # Total decode TPS
    tps_list.append(tps)
    
    print(f"Run {i+1}: TTFT={ttft:.4f}s (prompt={prompt_time:.4f}+first={first_token_time:.4f}), TPS={tps:.2f}")
    
    if i < num_runs - 1:
        time.sleep(cooldown_sec)

avg_ttft = sum(ttft_list) / num_runs
std_ttft = (sum((x - avg_ttft) ** 2 for x in ttft_list) / num_runs) ** 0.5
avg_tps = sum(tps_list) / num_runs
std_tps = (sum((x - avg_tps) ** 2 for x in tps_list) / num_runs) ** 0.5

print("\n+--------------------------+"  )
print(  "|     Benchmark Summary    |"  )
print(  "+--------------------------+\n")
print(f"Average TTFT: {avg_ttft:.4f}s ± {std_ttft:.4f}")
print(f"Average TPS (decode): {avg_tps:.2f} ± {std_tps:.2f}")
print(f"System: {torch.cuda.get_device_name(device)}")
