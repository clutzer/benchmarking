import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, FPQuantConfig
import warnings
import argparse

warnings.filterwarnings("ignore")

# Argument parser
parser = argparse.ArgumentParser(description="Benchmark a model on specified GPUs with precision and quantization")
parser.add_argument("--device", type=str, default=None, help="Device to use (e.g., 'cuda:0'). If not specified, uses all available GPUs.")
parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp32"], help="Base precision: 'fp16' or 'fp32' (default: fp16)")
parser.add_argument("--quantization", type=str, default="none", choices=["none", "fp4"], help="Quantization: 'none' or 'fp4' (Blackwell-only for full speed)")
parser.add_argument("--num_runs", type=int, default=20, help="Number of benchmark runs (default: 20)")
parser.add_argument("--cooldown_sec", type=float, default=5.0, help="Cooldown time between runs in seconds (default: 5.0)")
parser.add_argument("--force_precision", type=str, default=None, help="Force precision (e.g., 'fp16') overriding auto-detection")
args = parser.parse_args()

# Config
model_name = "microsoft/DialoGPT-medium"
input_length = 512
output_length = 128
num_runs = args.num_runs
cooldown_sec = args.cooldown_sec

# Map precision to torch dtype
precision_map = {
    "fp16": torch.float16,
    "fp32": torch.float32
}
dtype = precision_map[args.precision]

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

if args.device:
    device = args.device
else:
    device = "cuda" if torch.cuda.is_available() else "cpu"

# GPU detection for auto-precision
gpu_name = "CPU"
is_blackwell = False
if device.startswith("cuda"):
    prop = torch.cuda.get_device_properties(int(device.split(":")[1]) if ":" in device else 0)
    gpu_name = prop.name
    # Detect Blackwell (RTX PRO 6000 Server Edition)
    if "blackwell" in gpu_name.lower() or "rtx pro 6000" in gpu_name.lower():
        is_blackwell = True
        if args.force_precision is None:
            args.precision = "fp16"  # Base for FP4 quant
            dtype = torch.float16

if args.force_precision:
    args.precision = args.force_precision
    dtype = precision_map[args.precision]

# Check for qutlass availability for FP4 on Blackwell
qutlass_available = False
if args.quantization == "fp4" and is_blackwell:
    try:
        import qutlass  # Check if qutlass is installed
        qutlass_available = True
    except ImportError:
        print("Warning: qutlass not installed. Using pseudo-quantization for FP4 on Blackwell GPU. Install qutlass for full FP4 performance: "
              "'git clone https://github.com/IST-DASLab/qutlass.git && cd qutlass && pip install --no-build-isolation .'")

# Load tokenizer and model with quantization
tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

load_kwargs = {
    "low_cpu_mem_usage": True,
    "dtype": dtype if args.quantization == "none" else torch.bfloat16,  # BF16 base for FP4 stability
    "device_map": "auto" if device == "cpu" else None
}

if args.quantization == "fp4":
    quantization_config = FPQuantConfig(
        forward_dtype="nvfp4" if is_blackwell and qutlass_available else "mxfp4",
        pseudo_quantization=not (is_blackwell and qutlass_available)  # Pseudo for non-Blackwell or missing qutlass
    )
    load_kwargs["quantization_config"] = quantization_config
    if not is_blackwell or not qutlass_available:
        print(f"Warning: FP4 quantization on {gpu_name} uses pseudo-quantization (no speedup, emulates FP4). "
              "Full FP4 speed requires Blackwell GPU with qutlass.")

# Load model with fallback to non-quantized if FP4 fails
try:
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
except ImportError as e:
    if args.quantization == "fp4":
        print(f"Error: Failed to load model with FP4 quantization: {e}. Falling back to non-quantized {args.precision.upper()}.")
        load_kwargs.pop("quantization_config", None)  # Remove quantization config
        load_kwargs["dtype"] = dtype  # Use base precision
        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        args.quantization = "none"  # Update quantization status
    else:
        raise e  # Rethrow if not FP4-related

if device.startswith("cuda"):
    model = model.to(device)
model.eval()

# Calculate number of parameters (non-quantized equivalent)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print model information
print("\n+--------------------------+")
print("|       Model Details       |")
print("+--------------------------+\n")
print(f"Model: {model_name}")
print(f"Number of Parameters: {num_params:,}")
print(f"Base Precision: {args.precision.upper()} (torch.{dtype})")
print(f"Quantization: {args.quantization.upper() if args.quantization != 'none' else 'None'}")
if args.quantization == "fp4":
    mem_savings = " ~4x vs FP32" if args.quantization == "fp4" else ""
    print(f"Effective Precision: FP4{mem_savings} (Quality: Near-lossless on LLMs)")
print(f"GPU Architecture: {gpu_name} ({'Blackwell (FP4-optimized)' if is_blackwell else 'H100-compatible (FP16/FP32)'})")
print(f"Device: {gpu_name if device.startswith('cuda') else device}")
print(f"Number of Runs: {num_runs}")
print(f"Cooldown Between Runs: {cooldown_sec:.1f}s")

prompt_text = "Explain quantum computing in simple terms. " * 50
inputs = tokenizer(
    prompt_text, return_tensors="pt", truncation=True, max_length=input_length
).to(device)

# Warmup (increased for quantized models)
with torch.no_grad():
    for _ in range(10 if args.quantization == "fp4" else 5):
        _ = model.generate(**inputs, max_new_tokens=10, do_sample=False)
torch.cuda.synchronize()

ttft_list = []
tps_list = []
mem_list = []  # Track peak memory

for i in range(num_runs):
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # --- TTFT: Prompt eval + first token ---
    start_event = torch.cuda.Event(enable_timing=True)
    mid_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    with torch.no_grad():
        outputs = model(**inputs, return_dict=True, use_cache=True)
        mid_event.record()
        
        logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
    end_event.record()
    torch.cuda.synchronize()
    
    prompt_time = start_event.elapsed_time(mid_event) / 1000.0
    first_token_time = mid_event.elapsed_time(end_event) / 1000.0
    ttft = prompt_time + first_token_time
    ttft_list.append(ttft)
    
    # --- TPS: Decode remaining tokens ---
    past_key_values = outputs.past_key_values
    generated_tokens = 1
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
    tps = output_length / (first_token_time + decode_time) if (first_token_time + decode_time) > 0 else 0
    tps_list.append(tps)
    
    peak_mem = torch.cuda.max_memory_allocated(device) / 1024**3 if device.startswith("cuda") else 0  # GB
    mem_list.append(peak_mem)
    
    print(f"Run {i+1}: TTFT={ttft:.4f}s (prompt={prompt_time:.4f}+first={first_token_time:.4f}), TPS={tps:.2f}, Mem={peak_mem:.2f}GB")
    
    if i < num_runs - 1:
        time.sleep(cooldown_sec)

avg_ttft = sum(ttft_list) / num_runs
std_ttft = (sum((x - avg_ttft) ** 2 for x in ttft_list) / num_runs) ** 0.5
avg_tps = sum(tps_list) / num_runs
std_tps = (sum((x - avg_tps) ** 2 for x in tps_list) / num_runs) ** 0.5
avg_mem = sum(mem_list) / num_runs

# Summary
print("\n+--------------------------+")
print("|      Benchmark Summary    |")
print("+--------------------------+\n")
print(f"Average TTFT: {avg_ttft:.4f}s ± {std_ttft:.4f}")
print(f"Average TPS (decode): {avg_tps:.2f} ± {std_tps:.2f}")
print(f"Average Peak Memory: {avg_mem:.2f}GB")
print(f"System: {gpu_name if device.startswith('cuda') else device}")
