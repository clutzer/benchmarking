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

### TODO: Prometheus & Grafan

### Running a Benchmark

The easiest way to run a bench mark is to exec into the `benchmarking-server` container and execute one of the benchmarking scripts.  E.g.

`docker compose exec benchmark-server /benchmark-scripts/baseline.sh`

## Benchmarking Ecosystem

This benchmarking ecosystem consists of two docker containers: one to run the model (NIM inference server) and one to run the benchmark (genai-perf benchmark server).  There is a `Makefile` that handles some of the initial setup of the ecosystem as well as provides some helper targets showing you how to run benchmarks, list models, etc.

### NVIDIA NIM for Large Language Models (LLMs)

NVIDIA maintains a [Supported Models for NVIDIA NIM for LLMs](https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html) page that you can use to find a model to use with benchmarking.  Once you selected a model, you can retrieve a NIM optimized for that model using NVIDIA's [Optimized Models](https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html#optimized-models) list.  As an example, when developing this suite, we used [Llama-3.30-70b-Instruct](https://catalog.ngc.nvidia.com/orgs/nim/teams/meta/containers/llama-3.3-70b-instruct).  Once landing on that page (which requires a login which you've probably already created by virtue of creating the NGC token required in the *Getting Started* section, you should find a button entitled *Get Container* which will give you the image name to use in your compose file.  As a best-practice, we recommend creating a `docker-compose.override.yml` file with additional `services` in there, once for each model you want to benchmark against.  E.g. `inference-server-llama-3.3-3b-instruct`.  This way the original, baseline `docker-compose.yml` doesn't suffer from constant churn and we know we always have a good working benchmarking suite.

