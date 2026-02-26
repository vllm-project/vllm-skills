---
name: vllm-bench-serve
description: Benchmark vLLM or OpenAI-compatible serving endpoints using vllm bench serve. Supports multiple datasets (random, sharegpt, sonnet, HF), backends (openai, openai-chat, vllm-pooling, embeddings), throughput/latency testing with request-rate control, and result saving. Use when benchmarking LLM serving performance, measuring TTFT/TPOT, or load testing inference APIs.
---

# vLLM Bench Serve

Benchmark vLLM or any OpenAI-compatible serving endpoint using the `vllm bench serve` CLI. Measures throughput, latency (TTFT, TPOT), and goodput against configurable request load.

Reference: [vLLM Bench Serve Documentation](https://docs.vllm.ai/en/latest/cli/bench/serve/)

## Prerequisites

- vLLM installed (or any OpenAI-compatible server running)
- A vLLM server or API endpoint already serving a model
- Python environment with vLLM for the benchmark client

## Quick Start

**Basic benchmark against local vLLM server (default random dataset, 1000 prompts):**

```bash
vllm bench serve \
  --backend openai-chat \
  --host 127.0.0.1 \
  --port 8000 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --endpoint /v1/chat/completions
```

**Save results to JSON:**

```bash
vllm bench serve \
  --backend openai-chat \
  --host 127.0.0.1 \
  --port 8000 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --endpoint /v1/chat/completions \
  --save-result \
  --result-dir ./bench-results \
  --metadata "version=0.6.0" "tp=1"
```

> **Note:** When using `--backend openai-chat`, you must specify `--endpoint /v1/chat/completions` (default is `/v1/completions`).

## Core Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--backend` | `openai` | Backend type: `openai`, `openai-chat`, `openai-embeddings`, `vllm`, `vllm-pooling`, `vllm-rerank`, etc. |
| `--host` | `127.0.0.1` | Server host |
| `--port` | `8000` | Server port |
| `--base-url` | - | Alternative: full base URL instead of host:port |
| `--endpoint` | `/v1/completions` | API endpoint; use `/v1/chat/completions` for openai-chat |
| `--model` | (from /v1/models) | Model name |
| `--num-prompts` | `1000` | Number of prompts to process |
| `--request-rate` | `inf` | Requests per second; `inf` = burst all at once |
| `--max-concurrency` | - | Max concurrent requests (caps parallelism) |
| `--num-warmups` | `0` | Warmup requests before measuring |

## Datasets

| `--dataset-name` | Use Case |
|------------------|----------|
| `random` | Synthetic random prompts (default) |
| `sharegpt` | ShareGPT conversation format; requires `--dataset-path` |
| `sonnet` | Sonnet-style prompts |
| `hf` | HuggingFace dataset; requires `--dataset-path` (dataset ID) |
| `custom` / `custom_mm` | Custom dataset; requires `--dataset-path` |
| `prefix_repetition` | Prefix repetition benchmark |
| `random-mm` | Random multimodal (images/videos) |
| `spec_bench` | Spec bench dataset |

**Dataset-specific options (examples):**

```bash
# Random: control input/output length
--dataset-name random --random-input-len 1024 --random-output-len 128

# Sonnet defaults: input 550, output 150, prefix 200
--dataset-name sonnet --sonnet-input-len 550 --sonnet-output-len 150

# HuggingFace dataset
--dataset-name hf --dataset-path "lmarena-ai/VisionArena-Chat" --hf-split test

# General overrides (map to dataset-specific args)
--input-len 512 --output-len 256
```

## Load Control

```bash
# Fixed request rate (Poisson process)
--request-rate 10

# More bursty arrivals (gamma distribution, burstiness < 1)
--request-rate 10 --burstiness 0.5

# Ramp-up from low to high RPS
--ramp-up-strategy linear --ramp-up-start-rps 1 --ramp-up-end-rps 50

# Limit concurrency (useful for rate-limited APIs)
--max-concurrency 32
```

## Results and Metrics

| Argument | Description |
|----------|-------------|
| `--save-result` | Save benchmark results to JSON |
| `--save-detailed` | Include per-request TTFT, TPOT, errors in JSON |
| `--append-result` | Append to existing result file |
| `--result-dir` | Directory for result files |
| `--result-filename` | Custom filename (default: `{label}-{request_rate}qps-{model}-{timestamp}.json`) |
| `--percentile-metrics` | Metrics for percentiles: `ttft`, `tpot`, `itl`, `e2el` (default: `ttft,tpot,itl`) |
| `--metric-percentiles` | Percentile values, e.g. `25,50,99` (default: `99`) |
| `--goodput` | SLO for goodput: `ttft:500 tpot:50` (ms) |

## Sampling Parameters (OpenAI-compatible backends)

```bash
--temperature 0.7 --top-p 0.95 --top-k 50
--frequency-penalty 0 --presence-penalty 0 --repetition-penalty 1.0
```

## Common Workflows

**1. Throughput test with random dataset (burst):**

```bash
vllm bench serve --backend openai-chat --host 127.0.0.1 --port 8000 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name random \
  --num-prompts 500 --random-input-len 512 --random-output-len 128
```

**2. Latency test with fixed QPS:**

```bash
vllm bench serve --backend openai-chat --host 127.0.0.1 --port 8000 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --endpoint /v1/chat/completions \
  --request-rate 5 --num-prompts 200 \
  --save-result --percentile-metrics ttft,tpot --metric-percentiles 50,99
```

**3. Benchmark against remote API (base-url):**

```bash
vllm bench serve --backend openai-chat \
  --base-url "https://api.example.com/v1" \
  --model my-model \
  --header "Authorization=Bearer $API_KEY"
```

**4. Run inside Docker (when vLLM client not on host):**

```bash
docker exec <container-name> vllm bench serve \
  --backend openai-chat --host 127.0.0.1 --port 8000 \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --endpoint /v1/chat/completions \
  --dataset-name random --num-prompts 100
```

## Troubleshooting

- **Connection refused**: Ensure the server is running and `--host`/`--port` or `--base-url` are correct.
- **Model not found**: Pass `--model` explicitly or ensure `/v1/models` returns the model.
- **URL must end with chat/completions**: Use `--endpoint /v1/chat/completions` when `--backend openai-chat`.
- **Rate limit / 429**: Reduce `--request-rate` or `--max-concurrency`.
- **Ready check**: Use `--ready-check-timeout-sec 60` to wait for the endpoint before benchmarking.
- **SSL**: Use `--insecure` for self-signed certificates.

## Notes

- For embeddings/rerank benchmarks, use `--backend openai-embeddings`, `vllm-pooling`, or `vllm-rerank`.
- `--profile` requires `--profiler-config` on the server for vLLM profiling.
- Goodput SLOs are useful for SLA-style analysis; see [DistServe paper](https://arxiv.org/pdf/2401.09670) for details.