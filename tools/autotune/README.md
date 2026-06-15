# Launch Autotune

This directory contains a script for tuning Xinference launch parameters for a
fixed model and hardware setup. It currently supports the vLLM engine only.

The script is intentionally separate from the normal launch path. It launches a
temporary model for each trial, runs a synthetic serving benchmark, records the
metrics, terminates the temporary model, and writes the best launch parameters
to disk.

## Requirements

Install Xinference with the engine dependencies required by the target model, and
install Optuna:

```bash
pip install optuna
```

Start a Xinference endpoint before running the tuner:

```bash
xinference-local --host 0.0.0.0 --port 9997
```

## Quick Start

Run the script from the repository root:

```bash
python tools/autotune/launch.py \
  --endpoint http://localhost:9997 \
  --model-name qwen2.5-instruct \
  --model-engine vllm \
  --model-format pytorch \
  --size-in-billions 7 \
  --quantization none \
  --n-gpu 1 \
  --max-model-len 4096 \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --num-trials 8 \
  --num-prompts 64 \
  --input-len 1024 \
  --output-len 128 \
  --random-range-ratio 0.2 \
  --concurrency 32 \
  --stream \
  --ignore-eos \
  --objective balanced
```

Use the same fixed model fields that you would use for `launch_model`. For
example, pass `--model-path` for a local model path, `--gpu-idx` to pin devices,
or `--extra-launch-param KEY=VALUE` for a fixed launch option not exposed as a
dedicated flag. `VALUE` may be JSON:

```bash
python tools/autotune/launch.py \
  --model-name custom-llm \
  --model-engine vllm \
  --model-path /models/custom-llm \
  --tokenizer /models/custom-llm \
  --extra-launch-param trust_remote_code=true \
  --extra-launch-param tensor_parallel_size=2
```

## What Gets Tuned

The vLLM search space is fixed in the script. By default it tunes:

- `gpu_memory_utilization`: `0.85,0.90,0.95`
- `max_num_seqs`: `32,64,128`
- `max_num_batched_tokens`: `4096,8192,16384`
- `enable_chunked_prefill`: `false,true`

The candidate lists can be narrowed from the command line:

```bash
python tools/autotune/launch.py \
  --model-name qwen2.5-instruct \
  --model-engine vllm \
  --tokenizer Qwen/Qwen2.5-7B-Instruct \
  --gpu-memory-utilization-candidates 0.9,0.95 \
  --max-num-seqs-candidates 32,64 \
  --max-num-batched-tokens-candidates 8192,16384
```

`max_model_len` is not tuned. If you pass `--max-model-len`, the value is treated
as a fixed context length and forwarded to launch.

`enable_prefix_caching` and `enforce_eager` are not changed by default. Use
`--enable-prefix-caching`, `--disable-prefix-caching`, `--enforce-eager`, or
`--disable-enforce-eager` to force a fixed value. Use `--tune-prefix-caching` or
`--tune-enforce-eager` only if those booleans should be part of the search space.

## Benchmark Workload

The tuner uses the same synthetic random request generation style as
`benchmark/benchmark_serving.py --dataset-name random`. It does not require a
ShareGPT dataset.

Important workload flags:

- `--num-prompts`: number of synthetic requests per trial.
- `--input-len`: target prompt length.
- `--output-len`: target output length.
- `--random-range-ratio`: prompt/output length variation. This can be a single
  float or a JSON object with `input` and `output` keys.
- `--random-prefix-len`: shared prefix length for generated prompts.
- `--concurrency`: request concurrency.
- `--request-rate`: request arrival rate. The default is unlimited.
- `--stream`: use streaming completions and collect TTFT/TPOT/ITL metrics.
- `--ignore-eos`: ask the model to ignore EOS for stable output length.

Keep `--num-prompts` small while exploring the search space. Increase it for the
final confirmation run after the candidate space has been narrowed.

## Objective

The default objective is `balanced`, which rewards output token throughput,
success rate, and lower latency. Other objective choices are:

- `throughput`: maximize output token throughput.
- `request_throughput`: maximize completed requests per second.
- `latency`: minimize p99 TTFT when streaming, or p99 latency otherwise.

Trials below `--min-success-rate` are penalized.

## Results

Results are written under:

```text
autotune_results/<study-name>/
```

The directory contains:

- `config.json`: parsed arguments and fixed launch parameters.
- `trials.jsonl`: one record per trial, including launch kwargs, metrics, score,
  and errors.
- `study.db`: default Optuna SQLite storage.
- `best.json`: best trial score, params, metrics, and full launch kwargs.

Use `--study-name` to choose a stable output directory, and `--resume` to resume
an existing study:

```bash
python tools/autotune/launch.py \
  --study-name qwen25-7b-a10-vllm \
  --resume \
  --model-name qwen2.5-instruct \
  --model-engine vllm \
  --tokenizer Qwen/Qwen2.5-7B-Instruct
```
