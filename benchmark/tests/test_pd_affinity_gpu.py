# Copyright 2022-2026 Xinference Holdings Pte. Ltd
# Licensed under the Apache License, Version 2.0.
"""Opt-in ABBA routing comparison; results are not four-GPU scaling estimates."""
import asyncio
import importlib.util
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path

import pytest

from xinference.model.llm.vllm.xavier.test.test_pd_gpu import pd_cluster  # noqa: F401

pytestmark = pytest.mark.skipif(
    os.environ.get("XINFERENCE_TEST_PD_AFFINITY_GPU") != "1",
    reason="requires two CUDA GPUs and local Qwen2.5-0.5B weights",
)


def _gpu_snapshot():
    """Record foreign GPU processes; known idle daemons must be explicit."""
    import psutil

    output = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout
    gpu_pids = {int(line.strip()) for line in output.splitlines() if line.strip()}
    own = {os.getpid(), *(p.pid for p in psutil.Process().children(recursive=True))}
    allowed = {
        int(pid)
        for pid in os.environ.get("XINFERENCE_TEST_PD_IDLE_GPU_PIDS", "").split(",")
        if pid
    }
    return {
        "time": time.time(),
        "gpu_pids": sorted(gpu_pids),
        "foreign_pids": sorted(gpu_pids - own - allowed),
        "allowed_idle_pids": sorted(allowed),
    }


async def _guarded_measure(bench, endpoint, uid, workload, concurrency):
    samples = []
    stopped = asyncio.Event()

    async def monitor():
        while True:
            samples.append(await asyncio.to_thread(_gpu_snapshot))
            try:
                await asyncio.wait_for(stopped.wait(), timeout=1)
                return
            except asyncio.TimeoutError:
                pass

    task = asyncio.create_task(monitor())
    try:
        records, elapsed, _ = await bench.measure(
            endpoint, uid, workload, concurrency, 1
        )
    finally:
        stopped.set()
        await task
    return records, elapsed, samples


@pytest.mark.parametrize("backend", ["xavier"])
@pytest.mark.parametrize(
    "policy_name,run",
    [("round_robin", 0), ("affinity", 0), ("affinity", 1), ("round_robin", 1)],
)
def test_pd_affinity_benchmark(pd_cluster, backend, policy_name, run):  # noqa: F811
    import xoscar as xo

    from xinference.client import Client
    from xinference.core.pd_model import (
        PDModelActor,
        PrefixAffinitySchedulingPolicy,
        RoundRobinSchedulingPolicy,
    )

    spec = importlib.util.spec_from_file_location(
        "benchmark_pd", Path(__file__).parents[1] / "benchmark_pd.py"
    )
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    endpoint, log_path = pd_cluster
    client = Client(endpoint)
    worker = client.get_workers_info()[0]["work-ip"]
    uid = "affinity-benchmark"
    loop = asyncio.new_event_loop()
    long_run = os.environ.get("XINFERENCE_TEST_PD_AFFINITY_LONG") == "1"
    document_count = int(os.environ.get("XINFERENCE_TEST_PD_AFFINITY_DOCUMENTS", "12"))
    request_count = int(
        os.environ.get(
            "XINFERENCE_TEST_PD_AFFINITY_REQUESTS", "2400" if long_run else "120"
        )
    )
    assert (
        document_count > 0 and request_count > 0 and request_count % document_count == 0
    )
    profiles = os.environ.get(
        "XINFERENCE_TEST_PD_AFFINITY_PROFILES",
        "uniform,mixed" if long_run else "uniform",
    ).split(",")
    assert all(profile in {"uniform", "mixed"} for profile in profiles)
    phases = []
    for profile in profiles:
        documents = []
        for i in range(document_count):
            repetitions = (50, 100, 200, 350)[i % 4] if profile == "mixed" else 200
            text = (
                f"{profile} Document {i}: "
                + "Water evaporates and condenses in clouds. " * repetitions
            )
            documents.append(
                {
                    "messages": [
                        {"role": "user", "content": text + "\nBriefly explain rain."}
                    ],
                    "max_tokens": 16,
                    "temperature": 0,
                }
            )
        rng = random.Random(173)
        warm = []
        for _ in range(3):
            batch = list(documents)
            rng.shuffle(batch)
            warm.extend(batch)
        concurrent = []
        for _ in range(request_count // document_count):
            batch = list(documents)
            rng.shuffle(batch)
            concurrent.extend(batch)
        prefix = profile + "_" if long_run else ""
        phases.extend(
            [
                (prefix + "cold", documents, 1),
                (prefix + "warm", warm, 1),
                (prefix + "concurrent", concurrent, 8),
            ]
        )
    output = Path(os.environ["XINFERENCE_TEST_PD_AFFINITY_RESULTS"])
    output.mkdir(parents=True, exist_ok=True)

    async def select_policy():
        address = client._get_supervisor_internal_address()
        actor_uid = f"{uid}-{PDModelActor.default_uid()}"
        old = await xo.actor_ref(address=address, uid=actor_uid)
        info = await old.get_pd_info()
        producers = [
            (name, await old.get_prefill_actor(name))
            for name in info["prefill_replica_uids"]
        ]
        decoders = [
            (name, await old.get_decode_actor(name))
            for name in info["decode_replica_uids"]
        ]
        warmup = [
            {
                "messages": [
                    {"role": "user", "content": "Briefly explain rain. " * 10}
                ],
                "max_tokens": 4,
                "temperature": 0,
            }
        ]
        for turn in range(2):
            records, _, _ = await bench.measure(endpoint, uid, warmup, 1, 2)
            assert all("error" not in row for row in records)
            name, ref = decoders[0]
            await old.remove_decode_actor(name)
            await old.add_decode_actor(name, ref)
        await xo.destroy_actor(old)
        policy = (
            PrefixAffinitySchedulingPolicy
            if policy_name == "affinity"
            else RoundRobinSchedulingPolicy
        )
        router = await xo.create_actor(
            PDModelActor,
            uid,
            scheduling_policy=policy,
            transport_backend=backend,
            address=address,
            uid=actor_uid,
        )
        for name, ref in producers:
            await router.add_prefill_actor(name, ref)
        for name, ref in decoders:
            await router.add_decode_actor(name, ref)

    try:
        if long_run:
            assert not _gpu_snapshot()["foreign_pids"], "GPU is busy before launch"
        client.launch_model(
            model_uid=uid,
            model_name="qwen2.5-instruct",
            model_engine="vLLM",
            model_size_in_billions="0_5",
            model_path=os.environ["XINFERENCE_TEST_PD_MODEL_PATH"],
            model_format="pytorch",
            quantization="none",
            replica=4,
            vllm_transfer_backend_type=backend,
            enable_virtual_env=False,
            max_model_len=4096,
            max_num_seqs=16,
            gpu_memory_utilization=0.35,
            enforce_eager=True,
            dtype="float16",
            enable_prefix_caching=False,
            replica_config=[
                {
                    "role": role,
                    "devices": [{"worker_ip": worker, "n_gpu": 1, "gpu_idx": [gpu]}],
                }
                for role, gpu in [("prefill", 0)] * 2 + [("decode", 1)] * 2
            ],
        )
        loop.run_until_complete(select_policy())
        results = []
        for name, workload, concurrency in phases:
            offset = Path(log_path).stat().st_size
            started = time.time()
            if long_run:
                records, elapsed, gpu_samples = loop.run_until_complete(
                    _guarded_measure(bench, endpoint, uid, workload, concurrency)
                )
            else:
                records, elapsed, gpu_samples = loop.run_until_complete(
                    bench.measure(endpoint, uid, workload, concurrency, 1)
                )
            assert all(
                "error" not in row and row["text"].strip() for row in records
            ), records
            with open(log_path, "rb") as log:
                log.seek(offset)
                evidence = log.read().decode(errors="replace")
            hits = re.findall(
                r"Restored Xavier history: blocks=(\d+) gpu=(\d+) cpu=(\d+)", evidence
            )
            prefill = [
                float(v)
                for v in re.findall(
                    r"PD prefill complete: request=\S+ backend=xavier elapsed_s=([0-9.]+)",
                    evidence,
                )
            ]
            post_prefill = [
                float(v)
                for v in re.findall(
                    r"PD first decode response: request=\S+ backend=xavier post_prefill_s=([0-9.]+)",
                    evidence,
                )
            ]
            assert len(post_prefill) == len(records)
            assert len(prefill) == len(records)
            result = {
                "phase": name,
                "started_at": started,
                "gpu_samples": gpu_samples,
                "summary": bench.summarize(records, elapsed, 1.0, 0.05),
                "history_requests": len(hits),
                "history_gpu_blocks": sum(int(h[1]) for h in hits),
                "history_cpu_blocks": sum(int(h[2]) for h in hits),
                "post_prefill_p50_s": bench.percentile(post_prefill, 0.5),
                "post_prefill_p95_s": bench.percentile(post_prefill, 0.95),
                "prefill_p50_s": bench.percentile(prefill, 0.5),
                "prefill_p95_s": bench.percentile(prefill, 0.95),
                "records": records,
            }
            results.append(result)
            (output / f"{policy_name}-{run}.json").write_text(
                json.dumps(results, indent=2)
            )
            assert not any(
                s["foreign_pids"] for s in gpu_samples
            ), "Foreign GPU workload overlapped measurement; results are invalid"
            print(
                "AFFINITY_PHASE "
                + json.dumps(
                    {
                        "policy": policy_name,
                        "run": run,
                        **{
                            k: v
                            for k, v in result.items()
                            if k not in {"records", "gpu_samples"}
                        },
                    }
                ),
                flush=True,
            )
    finally:
        if uid in client.list_models():
            client.terminate_model(uid)
        loop.run_until_complete(loop.shutdown_asyncgens())
        loop.close()
