python-venv:
	python3 -m venv python-venv
	. python-venv/bin/activate && \
	pip install --upgrade pip && \
	pip install 'vllm>=0.5.5'

check-docker-nivida:
	docker compose exec trtllm nvidia-smi


# Define token variable using shell command
HF_TOKEN := $(shell cat ~/.secrets/hf-token.txt 2>/dev/null || echo "")

.PHONY: Llama-3.3-70B-Instruct-FP4
Llama-3.3-70B-Instruct-FP4:
	@if [ -z "$(HF_TOKEN)" ]; then \
		echo "Error: Hugging Face token not found in ~/.secrets/hf-token.txt"; \
		exit 1; \
	fi; \
	hf download --token $(HF_TOKEN) nvidia/$@

IMAGE := nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc0.post1
MODEL := meta-llama/Llama-3.3-70B-Instruct


test-quantize:
	MODEL=$(MODEL) IMAGE=$(IMAGE) HF_TOKEN=$(HF_TOKEN) ./test-quantize.sh

test-load-hf:
	MODEL_PATH=$(HOME)/.cache/huggingface/hub/models--nvidia--Llama-3.3-70B-Instruct-FP4/shapshots/6f6073b423013f6a7d4d9f39144961bfbfbc386b \
	IMAGE=$(IMAGE) \
	HF_TOKEN=$(HF_TOKEN) \
	./test-load.sh
