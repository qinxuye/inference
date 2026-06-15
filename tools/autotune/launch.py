#!/usr/bin/env python
# Copyright 2022-2023 XProbe Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio
import json
import logging
import math
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = REPO_ROOT / "benchmark"
if str(BENCHMARK_ROOT) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_ROOT))

from benchmark_serving import (  # noqa: E402
    ServingBenchmarkRunner,
    parse_random_range_ratio,
)
from utils import get_tokenizer, sample_random_requests  # noqa: E402

from xinference.client import RESTfulClient  # noqa: E402

logger = logging.getLogger(__name__)

FAILURE_SCORE = -1e12

VLLM_GPU_MEMORY_UTILIZATION_CANDIDATES = [0.85, 0.90, 0.95]
VLLM_MAX_NUM_SEQS_CANDIDATES = [32, 64, 128]
VLLM_MAX_NUM_BATCHED_TOKENS_CANDIDATES = [4096, 8192, 16384]
VLLM_BOOLEAN_CANDIDATES = [False, True]
SUMMARY_METRIC_KEYS = [
    "request_throughput",
    "input_token_throughput",
    "output_token_throughput",
    "mean_ttft",
    "p99_ttft",
    "mean_tpot",
    "p99_tpot",
    "mean_latency",
    "p99_latency",
    "success_rate",
]


def import_optuna():
    try:
        import optuna
    except ImportError as exc:
        raise SystemExit(
            "Optuna is required for autotune. Install it with `pip install optuna`."
        ) from exc
    return optuna


def parse_csv_ints(value: str) -> List[int]:
    try:
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid integer list: {value}") from exc


def parse_csv_floats(value: str) -> List[float]:
    try:
        return [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid float list: {value}") from exc


def parse_bool(value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_csv_bools(value: str) -> List[bool]:
    return [parse_bool(item) for item in value.split(",") if item.strip()]


def parse_key_value(value: str) -> Tuple[str, Any]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Expected KEY=VALUE for extra launch parameter, got: {value}"
        )
    key, raw_value = value.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError("Extra launch parameter key cannot be empty.")
    try:
        parsed_value = json.loads(raw_value)
    except json.JSONDecodeError:
        parsed_value = raw_value
    return key, parsed_value


def parse_gpu_idx(value: Optional[str]) -> Optional[List[int]]:
    if value is None:
        return None
    if value.strip() == "":
        return None
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def as_launch_model_size(value: Optional[str]) -> Optional[Any]:
    if value is None:
        return None
    if "_" in value or "." in value:
        return value
    try:
        return int(value)
    except ValueError:
        return value


def unique_values(values: Sequence[Any]) -> List[Any]:
    result = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def build_vllm_search_space(args: argparse.Namespace) -> Dict[str, List[Any]]:
    search_space: Dict[str, List[Any]] = {
        "enable_chunked_prefill": unique_values(args.enable_chunked_prefill_candidates),
        "gpu_memory_utilization": unique_values(args.gpu_memory_utilization_candidates),
        "max_num_seqs": unique_values(args.max_num_seqs_candidates),
        "max_num_batched_tokens": unique_values(args.max_num_batched_tokens_candidates),
    }
    if args.tune_prefix_caching:
        search_space["enable_prefix_caching"] = list(VLLM_BOOLEAN_CANDIDATES)
    if args.tune_enforce_eager:
        search_space["enforce_eager"] = list(VLLM_BOOLEAN_CANDIDATES)
    return search_space


def search_space_size(search_space: Dict[str, List[Any]]) -> int:
    size = 1
    for candidates in search_space.values():
        size *= len(candidates)
    return size


def build_base_launch_kwargs(args: argparse.Namespace) -> Dict[str, Any]:
    launch_kwargs: Dict[str, Any] = {
        "model_name": args.model_name,
        "model_type": args.model_type,
        "model_engine": args.model_engine,
        "model_size_in_billions": as_launch_model_size(args.size_in_billions),
        "model_format": args.model_format,
        "quantization": args.quantization,
        "replica": args.replica,
        "n_worker": args.n_worker,
        "n_gpu": args.n_gpu,
        "request_limits": args.request_limits,
        "worker_ip": args.worker_ip,
        "gpu_idx": parse_gpu_idx(args.gpu_idx),
        "model_path": args.model_path,
    }
    if args.max_model_len is not None:
        launch_kwargs["max_model_len"] = args.max_model_len

    for key, value in args.extra_launch_param:
        launch_kwargs[key] = value
    return {key: value for key, value in launch_kwargs.items() if value is not None}


def unique_model_uid(model_name: str, trial_number: int) -> str:
    safe_name = "".join(char if char.isalnum() else "-" for char in model_name.lower())
    safe_name = "-".join(part for part in safe_name.split("-") if part)
    prefix = safe_name[:32] or "model"
    return f"autotune-{prefix}-t{trial_number}-{uuid.uuid4().hex[:8]}"


def suggest_vllm_params(args: argparse.Namespace, trial: Any) -> Dict[str, Any]:
    enable_chunked_prefill = trial.suggest_categorical(
        "enable_chunked_prefill",
        unique_values(args.enable_chunked_prefill_candidates),
    )
    max_num_seqs = trial.suggest_categorical(
        "max_num_seqs", unique_values(args.max_num_seqs_candidates)
    )

    params = {
        "gpu_memory_utilization": trial.suggest_categorical(
            "gpu_memory_utilization",
            unique_values(args.gpu_memory_utilization_candidates),
        ),
        "max_num_seqs": max_num_seqs,
        "max_num_batched_tokens": trial.suggest_categorical(
            "max_num_batched_tokens",
            unique_values(args.max_num_batched_tokens_candidates),
        ),
        "enable_chunked_prefill": enable_chunked_prefill,
    }
    if args.tune_prefix_caching:
        params["enable_prefix_caching"] = trial.suggest_categorical(
            "enable_prefix_caching", VLLM_BOOLEAN_CANDIDATES
        )
    elif args.enable_prefix_caching is not None:
        params["enable_prefix_caching"] = args.enable_prefix_caching

    if args.tune_enforce_eager:
        params["enforce_eager"] = trial.suggest_categorical(
            "enforce_eager", VLLM_BOOLEAN_CANDIDATES
        )
    elif args.enforce_eager is not None:
        params["enforce_eager"] = args.enforce_eager

    return params


def validate_vllm_trial_params(
    args: argparse.Namespace, trial_params: Dict[str, Any]
) -> Optional[str]:
    min_batched_tokens = int(trial_params["max_num_seqs"])
    if args.max_model_len is not None and not trial_params["enable_chunked_prefill"]:
        min_batched_tokens = max(min_batched_tokens, args.max_model_len)

    if int(trial_params["max_num_batched_tokens"]) < min_batched_tokens:
        return (
            "max_num_batched_tokens must be at least "
            f"{min_batched_tokens} for this fixed context configuration."
        )
    return None


def calculate_metrics(benchmark: ServingBenchmarkRunner) -> Dict[str, Any]:
    total_time = float(benchmark.benchmark_time or 0.0)
    outputs = benchmark.outputs
    completed_outputs = [output for output in outputs if output.success]
    completed = len(completed_outputs)
    total = len(outputs)
    total_input_tokens = sum(output.prompt_len for output in completed_outputs)
    total_output_tokens = sum(output.completion_tokens for output in completed_outputs)

    metrics: Dict[str, Any] = {
        "completed": completed,
        "total": total,
        "success_rate": completed / total if total else 0.0,
        "duration_s": total_time,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "request_throughput": completed / total_time if total_time > 0 else 0.0,
        "input_token_throughput": (
            total_input_tokens / total_time if total_time > 0 else 0.0
        ),
        "output_token_throughput": (
            total_output_tokens / total_time if total_time > 0 else 0.0
        ),
    }

    if benchmark.stream:
        ttfts = [output.ttft for output in completed_outputs if output.ttft > 0]
        tpots = [
            (output.latency - output.ttft) / (output.completion_tokens - 1)
            for output in completed_outputs
            if output.completion_tokens > 1
        ]
        itls = [latency for output in completed_outputs for latency in output.itl]
        metrics.update(_latency_metrics("ttft", ttfts, scale=1000.0))
        metrics.update(_latency_metrics("tpot", tpots, scale=1000.0))
        metrics.update(_latency_metrics("itl", itls, scale=1000.0))
    else:
        latencies = [output.latency for output in completed_outputs]
        metrics.update(_latency_metrics("latency", latencies, scale=1.0))

    return metrics


def _latency_metrics(
    name: str, values: Sequence[float], scale: float
) -> Dict[str, Any]:
    if not values:
        return {
            f"mean_{name}": None,
            f"median_{name}": None,
            f"p99_{name}": None,
        }
    array = np.array(values, dtype=float) * scale
    return {
        f"mean_{name}": float(np.mean(array)),
        f"median_{name}": float(np.median(array)),
        f"p99_{name}": float(np.percentile(array, 99)),
    }


def objective_score(args: argparse.Namespace, metrics: Dict[str, Any]) -> float:
    success_rate = float(metrics["success_rate"])
    if success_rate < args.min_success_rate:
        return FAILURE_SCORE * (1.0 - success_rate)

    output_throughput = float(metrics["output_token_throughput"])
    if args.objective == "throughput":
        return output_throughput
    if args.objective == "request_throughput":
        return float(metrics["request_throughput"])
    if args.objective == "latency":
        if args.stream:
            p99_ttft = metrics.get("p99_ttft")
            return -float(p99_ttft if p99_ttft is not None else math.inf)
        p99_latency = metrics.get("p99_latency")
        return -float(p99_latency if p99_latency is not None else math.inf)

    latency_s = 0.0
    if args.stream:
        p99_ttft = metrics.get("p99_ttft")
        latency_s = float(p99_ttft or 0.0) / 1000.0
    else:
        p99_latency = metrics.get("p99_latency")
        latency_s = float(p99_latency or 0.0)
    return output_throughput * success_rate / (1.0 + latency_s)


def append_jsonl(path: Path, payload: Dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    tmp_path.replace(path)


def record_trial_result(
    trial: Any, result_dir: Path, trial_payload: Dict[str, Any]
) -> None:
    for key in (
        "status",
        "error",
        "metrics",
        "score",
        "launch_kwargs",
        "trial_params",
        "model_uid",
    ):
        if key in trial_payload:
            trial.set_user_attr(key, trial_payload[key])
    append_jsonl(result_dir / "trials.jsonl", trial_payload)


def run_trial(
    args: argparse.Namespace,
    client: RESTfulClient,
    tokenizer: Any,
    base_launch_kwargs: Dict[str, Any],
    result_dir: Path,
    trial: Any,
) -> float:
    started_at = time.time()
    model_uid = unique_model_uid(args.model_name, trial.number)
    cleanup_model = False
    trial_payload: Dict[str, Any] = {
        "trial_number": trial.number,
        "model_uid": model_uid,
        "launch_kwargs": None,
        "trial_params": None,
        "status": "running",
        "started_at": started_at,
    }

    try:
        launch_kwargs = dict(base_launch_kwargs)
        trial_params = suggest_vllm_params(args, trial)
        invalid_reason = validate_vllm_trial_params(args, trial_params)
        launch_kwargs.update(trial_params)
        launch_kwargs["model_uid"] = model_uid
        trial_payload["launch_kwargs"] = launch_kwargs
        trial_payload["trial_params"] = trial_params
        logger.info("Starting trial %s with params: %s", trial.number, trial_params)

        if invalid_reason:
            trial_payload.update(
                {
                    "status": "invalid",
                    "error": invalid_reason,
                    "score": FAILURE_SCORE,
                    "finished_at": time.time(),
                }
            )
            logger.info("Skipping invalid trial %s: %s", trial.number, invalid_reason)
            return FAILURE_SCORE

        actual_model_uid = client.launch_model(wait_ready=True, **launch_kwargs)
        model_uid = actual_model_uid
        cleanup_model = True
        trial_payload["model_uid"] = model_uid

        input_requests = sample_random_requests(
            args.num_prompts,
            tokenizer,
            input_len=args.input_len,
            output_len=args.output_len,
            range_ratio=args.random_range_ratio,
            prefix_len=args.random_prefix_len,
            seed=args.seed,
        )
        benchmark = ServingBenchmarkRunner(
            f"{args.endpoint.rstrip('/')}/v1/chat/completions",
            model_uid,
            input_requests,
            args.stream,
            args.concurrency,
            args.request_rate,
            api_key=args.api_key,
            print_error=args.print_error,
            ignore_eos=args.ignore_eos,
        )
        asyncio.run(benchmark.run())
        metrics = calculate_metrics(benchmark)
        score = objective_score(args, metrics)
        trial_payload.update(
            {
                "status": "completed",
                "metrics": metrics,
                "score": score,
                "finished_at": time.time(),
            }
        )
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("launch_kwargs", launch_kwargs)
        trial.set_user_attr("model_uid", model_uid)
        logger.info("Completed trial %s with score %.6f", trial.number, score)
        return score
    except Exception as exc:
        logger.exception("Trial %s failed", trial.number)
        trial_payload.update(
            {
                "status": "failed",
                "error": repr(exc),
                "score": FAILURE_SCORE,
                "finished_at": time.time(),
            }
        )
        return FAILURE_SCORE
    finally:
        if cleanup_model:
            try:
                client.terminate_model(model_uid)
            except Exception:
                logger.info(
                    "Failed to terminate trial model %s", model_uid, exc_info=True
                )
        record_trial_result(trial, result_dir, trial_payload)


def best_trial_payload(study: Any) -> Dict[str, Any]:
    trial = study.best_trial
    return {
        "trial_number": trial.number,
        "score": trial.value,
        "params": trial.params,
        "metrics": trial.user_attrs.get("metrics"),
        "launch_kwargs": trial.user_attrs.get("launch_kwargs"),
        "model_uid": trial.user_attrs.get("model_uid"),
    }


def trial_summary_payload(study: Any) -> List[Dict[str, Any]]:
    summary = []
    for trial in study.trials:
        attrs = trial.user_attrs
        metrics = attrs.get("metrics") or {}
        row = {
            "trial_number": trial.number,
            "state": trial.state.name,
            "status": attrs.get("status") or trial.state.name.lower(),
            "score": trial.value,
            "params": attrs.get("trial_params") or trial.params,
            "error": attrs.get("error"),
        }
        for key in SUMMARY_METRIC_KEYS:
            row[key] = metrics.get(key)
        summary.append(row)
    return summary


def format_summary_value(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def format_summary_error(value: Any, max_length: int = 96) -> str:
    if not value:
        return "-"
    first_line = str(value).splitlines()[0]
    if len(first_line) <= max_length:
        return first_line
    return first_line[: max_length - 3] + "..."


def print_trial_summary(best: Dict[str, Any], trials: List[Dict[str, Any]]) -> None:
    print("\nTrial summary:")
    headers = [
        "trial",
        "status",
        "score",
        "req/s",
        "out tok/s",
        "in tok/s",
        "p99 ttft",
        "p99 tpot",
        "p99 lat",
        "success",
        "error",
        "params",
    ]
    rows = []
    for trial in trials:
        rows.append(
            [
                str(trial["trial_number"]),
                str(trial["status"]),
                format_summary_value(trial["score"]),
                format_summary_value(trial["request_throughput"]),
                format_summary_value(trial["output_token_throughput"]),
                format_summary_value(trial["input_token_throughput"]),
                format_summary_value(trial["p99_ttft"]),
                format_summary_value(trial["p99_tpot"]),
                format_summary_value(trial["p99_latency"]),
                format_summary_value(trial["success_rate"]),
                format_summary_error(trial["error"]),
                json.dumps(trial["params"], sort_keys=True),
            ]
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        if rows
        else len(headers[index])
        for index in range(len(headers))
    ]
    print(
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))
    )
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))

    print("\nBest trial:")
    print(json.dumps(best, ensure_ascii=False, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Autotune Xinference launch parameters for a fixed model."
    )
    parser.add_argument("--endpoint", default="http://localhost:9997")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-type", default="LLM")
    parser.add_argument("--model-engine", choices=["vllm"], default="vllm")
    parser.add_argument("--model-format", default=None)
    parser.add_argument("--size-in-billions", default=None)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--replica", type=int, default=1)
    parser.add_argument("--n-worker", type=int, default=1)
    parser.add_argument("--n-gpu", default="auto")
    parser.add_argument("--gpu-idx", default=None)
    parser.add_argument("--worker-ip", default=None)
    parser.add_argument("--request-limits", type=int, default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Fixed context length to pass to vLLM. This is not tuned.",
    )
    parser.add_argument(
        "--extra-launch-param",
        action="append",
        type=parse_key_value,
        default=[],
        help="Additional fixed launch parameter as KEY=VALUE. VALUE may be JSON.",
    )

    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--num-prompts", type=int, default=128)
    parser.add_argument("--input-len", type=int, default=1024)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument(
        "--random-input-len",
        dest="input_len",
        type=int,
        help="Alias for --input-len.",
    )
    parser.add_argument(
        "--random-output-len",
        dest="output_len",
        type=int,
        help="Alias for --output-len.",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=parse_random_range_ratio,
        default=0.0,
        help="Float or JSON object with input/output keys.",
    )
    parser.add_argument("--random-prefix-len", type=int, default=0)
    parser.add_argument("--concurrency", "-c", type=int, default=32)
    parser.add_argument("--request-rate", type=float, default=float("inf"))
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--print-error", action="store_true")

    parser.add_argument("--num-trials", type=int, default=12)
    parser.add_argument(
        "--objective",
        choices=["balanced", "throughput", "request_throughput", "latency"],
        default="balanced",
    )
    parser.add_argument("--min-success-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--output-dir", default="autotune_results")
    parser.add_argument(
        "--storage",
        default=None,
        help="Optuna storage URL. Defaults to sqlite in the study output directory.",
    )
    parser.add_argument("--resume", action="store_true")

    parser.add_argument(
        "--gpu-memory-utilization-candidates",
        type=parse_csv_floats,
        default=VLLM_GPU_MEMORY_UTILIZATION_CANDIDATES,
    )
    parser.add_argument(
        "--max-num-seqs-candidates",
        type=parse_csv_ints,
        default=VLLM_MAX_NUM_SEQS_CANDIDATES,
    )
    parser.add_argument(
        "--max-num-batched-tokens-candidates",
        type=parse_csv_ints,
        default=VLLM_MAX_NUM_BATCHED_TOKENS_CANDIDATES,
    )
    parser.add_argument(
        "--enable-chunked-prefill-candidates",
        type=parse_csv_bools,
        default=VLLM_BOOLEAN_CANDIDATES,
        help="Comma-separated booleans for the enable_chunked_prefill search space.",
    )
    prefix_caching_group = parser.add_mutually_exclusive_group()
    prefix_caching_group.add_argument(
        "--enable-prefix-caching",
        dest="enable_prefix_caching",
        action="store_true",
        default=None,
        help="Force vLLM enable_prefix_caching=True when not tuning it.",
    )
    prefix_caching_group.add_argument(
        "--disable-prefix-caching",
        dest="enable_prefix_caching",
        action="store_false",
        help="Force vLLM enable_prefix_caching=False when not tuning it.",
    )
    parser.add_argument(
        "--tune-prefix-caching",
        action="store_true",
        help="Include enable_prefix_caching in the search space.",
    )
    enforce_eager_group = parser.add_mutually_exclusive_group()
    enforce_eager_group.add_argument(
        "--enforce-eager",
        dest="enforce_eager",
        action="store_true",
        default=None,
        help="Force vLLM enforce_eager=True when not tuning it.",
    )
    enforce_eager_group.add_argument(
        "--disable-enforce-eager",
        dest="enforce_eager",
        action="store_false",
        help="Force vLLM enforce_eager=False when not tuning it.",
    )
    parser.add_argument(
        "--tune-enforce-eager",
        action="store_true",
        help="Include enforce_eager in the search space.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the tokenizer.",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not (0.0 < args.min_success_rate <= 1.0):
        raise ValueError("--min-success-rate must be in (0, 1].")
    if args.num_trials <= 0:
        raise ValueError("--num-trials must be positive.")
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive.")
    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive.")
    if args.model_engine != "vllm":
        raise ValueError("Only vLLM autotune is supported in this script.")
    search_space = build_vllm_search_space(args)
    empty_candidates = [
        name for name, candidates in search_space.items() if len(candidates) == 0
    ]
    if empty_candidates:
        raise ValueError(f"Search candidates cannot be empty: {empty_candidates}.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    validate_args(args)

    optuna = import_optuna()
    if args.study_name is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        args.study_name = f"{args.model_name}-{args.model_engine}-{timestamp}"

    result_dir = Path(args.output_dir) / args.study_name
    result_dir.mkdir(parents=True, exist_ok=True)
    storage = args.storage or f"sqlite:///{result_dir / 'study.db'}"

    search_space = build_vllm_search_space(args)
    total_candidates = search_space_size(search_space)
    if args.max_model_len is None and False in search_space["enable_chunked_prefill"]:
        logger.warning(
            "enable_chunked_prefill=false is in the search space while "
            "--max-model-len is not set. vLLM will use the model default "
            "context length, which may require much larger "
            "max_num_batched_tokens values."
        )
    if args.num_trials > total_candidates:
        logger.info(
            "Only %s unique candidate combinations are available; "
            "stopping after the grid is exhausted even though --num-trials=%s.",
            total_candidates,
            args.num_trials,
        )
    sampler = optuna.samplers.GridSampler(search_space)
    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=args.resume,
        sampler=sampler,
    )

    client = RESTfulClient(args.endpoint, api_key=args.api_key)
    tokenizer = get_tokenizer(args.tokenizer, trust_remote_code=args.trust_remote_code)
    base_launch_kwargs = build_base_launch_kwargs(args)
    write_json(
        result_dir / "config.json",
        {
            "args": vars(args),
            "base_launch_kwargs": base_launch_kwargs,
            "search_space": search_space,
            "search_space_size": total_candidates,
            "storage": storage,
        },
    )

    study.optimize(
        lambda trial: run_trial(
            args,
            client,
            tokenizer,
            base_launch_kwargs,
            result_dir,
            trial,
        ),
        n_trials=args.num_trials,
    )

    best = best_trial_payload(study)
    trials = trial_summary_payload(study)
    write_json(result_dir / "best.json", best)
    write_json(result_dir / "summary.json", {"best": best, "trials": trials})
    print_trial_summary(best, trials)


if __name__ == "__main__":
    main()
