#!/bin/sh

docker run \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  -v "$MODEL_PATH:/workspace/model" \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  $IMAGE \
  bash -c '
    python examples/llm-api/quickstart_advanced.py \
      --model_dir /workspace/model/ \
      --prompt "Paris is great because" \
      --max_tokens 64
    '

