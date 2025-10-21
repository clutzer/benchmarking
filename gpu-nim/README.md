# NIM-Based Benchmarking with GenAI-Perf

## Getting Started

### Secrets

You must create a ~/.secrets directory on your host with an NGC and Huggingface tokens.  E.g.

```
$ ls ~/.secrets
hf-token.txt  ngc-api-key.txt
```

These secrets are used by both the Makefile to create a docker compose .env as well as by benchmarking scripts to perform `hf auth login` calls.

### Docker

This benchmarking suite requires Docker.  The most reliable way to do this is to follow the official [Docker instructions](https://docs.docker.com/engine/install/ubuntu/).  At the time of writing (2025) I strongly recommend against using an Ubuntu packaged version.

### NIM Model Profiles

It's critical to select an appropriate NIM model profile and set the `NIM_MODEL_PROFILE` environment variable accordingly.  We don't go into detail here on what these are, but suffice to say they have to be correctly set for your hardware and environment.  Use the following `make` command to get a list of supported profiles for you system:

```
make list-nim-profiles
```

Then, look for output like the following:

```
WARNING 2025-10-21 15:06:40.531 profile_utils.py:879] Profile fd7da1db9a16d730859cda85c0f0487f37a49f2a58d7e66a006bde58338a61e4 does not support LoRA, but LoRA is enabled. Skipping profile.
INFO 2025-10-21 15:06:40.532 manifest.rs:160] first-level manifest validation skipped for version 1 schema
MODEL PROFILES
- Compatible with system and runnable:
  - 8333205760e6953934deaa1f47f4b5ca37a26c5ad03b076bcb43d017b282becc (vllm-bf16-tp2-pp1-b84343a5f0204e11c83c4e0dce9626d4795db4b7f251375b3d321dc0d8c19d46)
  - bbbb4a404c66d299bf36015162aae3c73615d39011213ffd5a8cad75c42ee80b (tensorrt_llm-h100_nvl-fp8-tp2-pp1-throughput-2321:10de-5e20634213cb4c32e86f048dbc274eb8bff74720af81fe400d8924e766f3e723-4)
  - With LoRA support:
    - 5b6bc247253a3ceffd79844090873b9222ef14de9220f9af7dfdffb9b2289e92 (vllm-bf16-tp2-pp1-lora-32-b84343a5f0204e11c83c4e0dce9626d4795db4b7f251375b3d321dc0d8c19d46)
    - a1f3d53570f80123a72c8ca2e1875cb7a7f58671a3e098596f6acde4241adc0b (tensorrt_llm-h100_nvl-fp8-tp2-pp1-throughput-lora-lora-2321:10de-35ff79694b241ad9598afd3b69727121b5678b417fb17ce7ef84b241388e9c2e-4)
- Compilable to TRT-LLM using just-in-time compilation of HF models to TRTLLM engines: <None>
- Incompatible with system:
  - 5b66d3f59554952168cc9a6a51b34c2c380d8c7a3ac436e121df235deaf2cdf8 (vllm-bf16-tp8-pp1-b84343a5f0204e11c83c4e0dce9626d4795db4b7f251375b3d321dc0d8c19d46)
  - 2ee05e2cc6d34802c557e15e3baa7af5e8387f1354e124036a9ea25f87d779bd (vllm-bf16-tp4-pp1-b84343a5f0204e11c83c4e0dce9626d4795db4b7f251375b3d321dc0d8c19d46)
  - 84e8a5d3d6c920ecd52151e13d2d92fb28fc7e53bd10c6b95165ba922630aa28 (tensorrt_llm-b200-nvfp4-tp2-pp1-latency-2901:10de-de8e5c7fcc38713239f8ca4db1136a4aa45a3e610f68c78fdd957a566051b033-2)
```

You will want to select a profile ID that is compatible and runnable with your
system.  As an example, on an NVIDIA H100 based system we used:

```
# gpu-nim
export NIM_MODEL_PROFILE=bbbb4a404c66d299bf36015162aae3c73615d39011213ffd5a8cad75c42ee80b
``` 
### TODO: Prometheus & Grafan

### Running a Benchmark

The easiest way to run a bench mark is to exec into the `benchmarking-server` container and execute one of the benchmarking scripts.  E.g.

`docker compose exec benchmark-server /benchmark-scripts/baseline.sh`

## Benchmarking Ecosystem

This benchmarking ecosystem consists of two docker containers: one to run the model (NIM inference server) and one to run the benchmark (genai-perf benchmark server).  There is a `Makefile` that handles some of the initial setup of the ecosystem as well as provides some helper targets showing you how to run benchmarks, list models, etc.

### NVIDIA NIM for Large Language Models (LLMs)

NVIDIA maintains a [Supported Models for NVIDIA NIM for LLMs](https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html) page that you can use to find a model to use with benchmarking.  Once you selected a model, you can retrieve a NIM optimized for that model using NVIDIA's [Optimized Models](https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html#optimized-models) list.  As an example, when developing this suite, we used [Llama-3.30-70b-Instruct](https://catalog.ngc.nvidia.com/orgs/nim/teams/meta/containers/llama-3.3-70b-instruct).  Once landing on that page (which requires a login which you've probably already created by virtue of creating the NGC token required in the *Getting Started* section, you should find a button entitled *Get Container* which will give you the image name to use in your compose file.  As a best-practice, we recommend creating a `docker-compose.override.yml` file with additional `services` in there, once for each model you want to benchmark against.  E.g. `inference-server-llama-3.3-3b-instruct`.  This way the original, baseline `docker-compose.yml` doesn't suffer from constant churn and we know we always have a good working benchmarking suite.

