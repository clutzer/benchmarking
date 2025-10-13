import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
import argparse

warnings.filterwarnings("ignore")  # Suppress minor warnings

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Benchmark a model on specified or all GPUs")
parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda:0', 'cuda:1'). If not specified, uses all available GPUs.")
args = parser.parse_args()

# Config
model_name = "microsoft/DialoGPT-medium"  # Or "meta-llama/Llama-2-7b-chat-hf"
input_length = 512  # Approx input tokens
output_length = 128  # Output tokens to generate
num_runs = 10  # Average over runs
batch_size = 1  # Single prompt for latency focus

# Determine device(s)
if args.device:
    device = args.device
    if not torch.cuda.is_available() or (device.startswith("cuda") and int(device.split(":")[-1]) >= torch.cuda.device_count()):
        raise ValueError(f"Device {device} is not available. Available devices: {[f'cuda:{i}' for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ['cpu']}")
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Use all GPUs if available, else CPU

# Prompt (adjust to hit ~input_length tokens)
prompt = "Explain quantum computing in simple terms. " * 50  # ~512 tokens

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Set pad_token if unset

# Initialize model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,  # FP16 for speed (H100/Blackwell optimized)
    low_cpu_mem_usage=True
)

# Move model to device(s)
if device == "cuda" and torch.cuda.device_count() > 1:
    print(f"Using all {torch.cuda.device_count()} GPUs")
    model = torch.nn.DataParallel(model)  # Use all GPUs with DataParallel
model = model.to(device)

# Tokenize input with attention mask
inputs = tokenizer(
    prompt,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=input_length,
    return_attention_mask=True
)
input_ids = inputs["input_ids"].to(device)
attention_mask = inputs["attention_mask"].to(device)

# Warmup run
with torch.no_grad():
    _ = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )

# Benchmark runs
ttft_list = []
tps_list = []

for _ in range(num_runs):
    # Measure TTFT (prompt eval + first token)
    start_time = time.time()
    with torch.no_grad():
        # Prompt evaluation
        outputs = model(input_ids, attention_mask=attention_mask)
        prompt_eval_time = time.time() - start_time
    
    # First token decode
    logits = outputs.logits[:, -1, :]  # Last token's logits
    next_token = torch.argmax(logits, dim=-1, keepdim=True)  # Greedy decode
    first_token_time = time.time() - start_time - prompt_eval_time
    ttft = prompt_eval_time + first_token_time  # TTFT = eval + first decode

    # Full generation for TPS
    gen_start = time.time()
    with torch.no_grad():
        gen_outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=output_length,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    gen_time = time.time() - gen_start
    decode_tokens = gen_outputs.shape[1] - input_ids.shape[1]  # Generated tokens
    tps = decode_tokens / gen_time if gen_time > 0 else 0

    ttft_list.append(ttft)
    tps_list.append(tps)

# Results
avg_ttft = sum(ttft_list) / num_runs
avg_tps = sum(tps_list) / num_runs
print(f"Average TTFT: {avg_ttft:.4f} seconds")
print(f"Average Tokens/Second (decode): {avg_tps:.2f}")

# System info
if device == "cuda" and torch.cuda.device_count() > 1:
    print(f"System Info: {torch.cuda.device_count()} GPUs, {[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}")
else:
    print(f"System Info: 1 GPU, {torch.cuda.get_device_name(device) if device.startswith('cuda') else 'CPU'}")
