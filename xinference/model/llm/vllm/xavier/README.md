# Xavier developer notes

This document covers implementation, diagnostics and contributor validation. For installation, replica placement, supported models and cache settings, see the [PD separation user guide](../../../../../doc/source/user_guide/pd_separation.rst).

## Backend boundaries

Xavier and vLLM's native `NixlConnector` are separate backends. Selecting `vllm_transfer_backend_type="nixl"` uses the native connector; configuring `xavier_gpu_cache_bytes` does not select it.

The Xavier V1 connector supports vLLM 0.21.0 or newer. P/D requires xoscar[nixl]>=0.11.1 in both worker and model environments. The legacy V0 adapter remains for non-P/D deployments with vLLM below 0.11.0; versions 0.11–0.20 are unsupported. GPU integration CI pins vLLM 0.21.0 and NIXL 1.1.0. Match vLLM, Torch and NIXL versions: importing vLLM alone does not validate engine startup.

The native backend passes producer handoff metadata to the selected decoder and lets NIXL manage transfer completion and cache release. Failed handoffs fail the request rather than silently recomputing. `VLLM_NIXL_ABORT_REQUEST_TIMEOUT` bounds producer retention if a decoder never consumes a completed prefill. This path does not create Xavier collective actors or force eager execution. Parallel sampling is rejected because child requests cannot safely share one producer transfer lease.

## Cache ownership and transfer

P/D uses request-scoped GPU-to-GPU handoff by default, with no opt-in flag. The producer keeps its engine blocks until decoder reads and optional history copies finish. Independent history uses 256 MiB of GPU storage per replica by default and spills to CPU, bounded by engine KV block capacity. `xavier_gpu_cache_bytes=0` disables history while keeping direct handoff. CPU-history hits restore to the local GPU in per-layer batches; P-to-D traffic remains xoscar NIXL. Missing NIXL fails launch.

Unclaimed producer tickets expire after 120 seconds. D atomically claims a live ticket before reporting a remote hit; expiry is a scheduler miss, not a worker transfer error. Claimed tickets have a 600-second idle lease refreshed by scheduling retries and slab transfers. Active reads cannot expire. If a claim expires before or between slabs, the worker reports every destination block of that request through vLLM's load-error callback together with receive completion; the default Xavier recompute policy retries locally without exposing partial KV or crashing EngineCore. Other transfer/layout failures still propagate. Allocation marks the handoff consumed, so preempted decode requests recompute locally rather than rereading the ticket. Router cleanup abandons only unclaimed tickets, while allocated loads drain before release.

History uses one writer, a 64 MiB candidate-prefix cap (also limited by total history capacity), and a 10 ms soft deadline. Long prompts retain the leading admissible prefix instead of being skipped entirely; all engine blocks remain owned until the active chunk drains. `history_capacity_limited_blocks` counts candidates beyond the cap, `history_deadline_dropped_blocks` and `history_closing_dropped_blocks` count remaining candidates abandoned at the deadline or shutdown, and `history_skipped_requests` counts a busy writer. When both tiers are full, new content must be observed again before replacing retained blocks. Admission stops at the first rejected or capacity-limited prefix block. Cache and probation LRU order retain prefix heads longer than tails. History hit counters count unique blocks actually written by successful loads, not reservations or scheduling retries. Active leases prevent eviction. History is restored on P; routing still visits P before D. Direct handoff supports `n=1`.

The following snapshot-path details apply to hybrid deployments. Xavier CPU snapshots are keyed by prompt content, not reusable engine block IDs. Readers reserve complete snapshots during transfer; unavailable snapshots or cache pressure can cause a cache miss and local computation. This differs from a transfer failure, which propagates as an error. Only complete layers are published for remote reuse. CPU capacity is bounded by the engine's KV block capacity.

In the GPU-first path, engine caches are shared with the local TransferActor through CUDA IPC. Retained GPU snapshots use xoscar NIXL; CPU snapshots use batched Gloo transfers. The worker sets `UCX_MEMTYPE_CACHE=n` and defaults `UCX_TLS` to `tcp,cuda_copy,cuda_ipc`, preserving an explicit `UCX_TLS` setting.

The snapshot budget excludes the model, engine KV cache, persistent transfer buffers, temporary tensors and allocator overhead. Separate send and receive slabs allow a replica to serve a peer while loading from another. Slabs normally use up to 16 MiB each; a single larger block can require more. Persistent small views share their allocations. Each source rank must produce consecutive small batches before selecting the small view, avoiding repeated NIXL re-registration for alternating full/small batches.

GPU overflow demotes unleased snapshots to CPU. If both tiers have no evictable space, new snapshots are skipped. Layout mismatches and transfer failures propagate instead of silently converting data or promising recomputation. If staging fails, outstanding CUDA work is fenced before unpublished snapshots are discarded; synchronization errors still propagate.

P/D always uses asynchronous loads. In the hybrid snapshot path, when `xavier_gpu_cache_bytes` is set (including zero), KV loads execute asynchronously in the TransferActor so ready requests can continue decoding. Omitting the setting for hybrid replicas retains synchronous CPU loading. Cancellation holds destination blocks and source leases until writes complete. Cleanup attempts every lease release and preserves the original load exception if cleanup also fails. Shutdown drains outstanding operations before releasing buffers and snapshot storage, and logs cache placement and transfer counters. Repeated close calls share one close task; the closed runtime remains a guard against remapping.

If the last request is aborted while an async load is pending, an idle vLLM EngineCore may not poll completion again until the next request arrives. Actor-side writes and source lease release continue independently, but destination block reclamation and connector-side completion cleanup wait for that next engine step or shutdown.

The connector accepts supported block-first and K/V-first layouts and rejects an ambiguous block axis. Shared null-block positions in allocated cache groups are omitted from destination mappings; conflicting writes to real destinations remain errors. This is not a claim of support for recurrent state: hybrid/recurrent attention models such as Qwen3.5 are rejected. Successful model launch alone does not demonstrate correct recurrent-state transfer.

In the legacy V0 path, prefill replicas release only the requested completed sequence; ordinary hybrid and decode replicas retain automatic cleanup.

## Request routing and lifecycle

The PD router issues a one-output-token prefill subrequest without changing the decoder's generation settings. Routing becomes available only after the replicas and transfer components are ready. Aborts reach both roles. Deployment termination also removes the PD router and Xavier's rank-zero coordinator, collective manager and block tracker.

Native NIXL routes are owned by the supervisor, so worker restart requires deployment relaunch; model-subprocess recovery on a live worker re-registers its PD route. Each native replica receives a dynamically allocated side-channel port, including after recovery. Host discovery and explicit interface configuration are documented in the user guide.

## Profiling

Set `XINFERENCE_XAVIER_PROFILE=1` in the model process environment and collect server logs. From the repository root:

```bash
python benchmark/analyze_pd_profile.py server.log --output profile.json
```

Profiling synchronizes CUDA and adds logging overhead. Run it separately from throughput benchmarks. Timings are nested: `load_rpc` includes actor timings, and `actor_receive` includes `gloo_receive`. Do not add nested totals or interpret Gloo receive wait as isolated network time.

## Backend comparison

From the repository root:

```bash
python benchmark/benchmark_pd.py --help
```

Supply launch JSON with model settings and P/D placement, plus a JSONL workload of chat request bodies. The runner compares ordinary hybrid replicas, Xavier and native NIXL sequentially using the same GPU allocation and workload. It records per-request results, TTFT, average time per output token after the first token (TPOT), latency percentiles and throughput under the selected latency limits.

Repeated workload entries exercise warm prefix reuse; distinct prefixes exercise cold requests. Verify actual cache-hit counters before attributing gains to Xavier history rather than the engine's own cache. Two GPUs cover 1P1D versus two hybrid replicas; 2P2D requires four independent replica GPUs. Record model/runtime versions, cache budgets and hardware with results. Diagnostic timings and historical measurements should not be presented as current-head serving benchmarks.

## Two-GPU integration test

### Multiple P/D replicas sharing GPUs

Run the opt-in multi-replica test with a small full-attention model:

```bash
XINFERENCE_ALLOW_MULTI_REPLICA_PER_GPU=1 XINFERENCE_TEST_PD_MULTI_GPU=1 \
  python -m pytest -v xinference/model/llm/vllm/xavier/test/test_pd_multi_gpu.py
```

The test covers 2P1D, 1P2D and 2P2D for Xavier and native NIXL. All producers
share GPU 0 and all decoders share GPU 1, with a 0.35 engine memory budget per
replica. The existing `XINFERENCE_TEST_PD_MODEL_PATH`, model name and size
overrides apply. It checks request-specific answers, streaming, eight concurrent
requests, actual KV loads and Xavier history restoration with engine prefix
caching disabled. For 2P2D it also removes and re-registers one decoder route:
equal-length round robins otherwise cover only two of the four P/D pairs.
This changes route registration, not the decoder process, and is not a crash
recovery test. `MULTI_PD_RESULT` records observed pairs and request counts.

This setup validates multi-replica behavior on two GPUs. The colocated engines
share compute and memory bandwidth, so its performance does not establish
four-GPU scaling.

### Prefix-affinity routing experiment

The opt-in benchmark compares round-robin routing with bounded prefix hints on
2P2D sharing two GPUs. Each policy gets a fresh deployment, all four transfer
pairs are warmed before measurement, and ABBA ordering reduces run-order bias:

```bash
XINFERENCE_TEST_PD_AFFINITY_GPU=1 \
XINFERENCE_ALLOW_MULTI_REPLICA_PER_GPU=1 \
XINFERENCE_TEST_PD_VLLM_LOG_LEVEL=INFO \
XINFERENCE_TEST_PD_MODEL_PATH=/path/to/Qwen2.5-0.5B-Instruct \
XINFERENCE_TEST_PD_AFFINITY_RESULTS=/tmp/pd-affinity-results \
  python -m pytest -vs benchmark/tests/test_pd_affinity_gpu.py
```

The runner records cold document prefixes, shuffled repeated prefixes and eight
concurrent requests. JSON results include TTFT, TPOT, prefill RPC latency and
actual GPU/CPU history block restores. Connection warm-up is excluded from these
measurements; production's first use of a new P/D pair still pays that cost.
The candidate policy is injected only by this benchmark; the production default
remains round robin until measurements justify changing it.

#### Exploratory measurements (2026-10-05)

Qwen2.5-0.5B-Instruct FP16, vLLM 0.21.0, xoscar 0.11.1 and NIXL 1.1
on two RTX 3090 Ti GPUs with NVLink. Two P replicas share GPU 0 and two D
replicas share GPU 1. Each replica uses `gpu_memory_utilization=0.35`, eager
execution, engine prefix caching disabled and the default 256 MiB Xavier GPU
history. Each deployment measures 12 cold documents (1839–1840 input tokens),
36 shuffled repeated documents sequentially, then 120 requests at concurrency
8; every request generates 16 tokens. All four cases passed (672 requests).
All reported history restores used GPU memory; CPU spill was not exercised.

The final candidate prefers a previously successful prefix while allowing at
most one extra outstanding request relative to the least-loaded producer.
Its load count includes KV handoff until the first D response for streams,
or the complete response for non-streaming calls. These are routing hints,
not engine queue lengths or guaranteed cache hits.

| Policy and repetition | Repeated TTFT p50 / p95 (ms) | Repeated GPU blocks restored | Concurrent requests/s | Concurrent TTFT p95 (ms) |
| --- | ---: | ---: | ---: | ---: |
| round_robin 0 | 86.6 / 114.1 | 2956 | 30.56 | 200.9 |
| affinity 0 | 81.1 / 86.9 | 4068 | 33.89 | 181.7 |
| affinity 1 | 81.1 / 114.4 | 4086 | 33.80 | 191.3 |
| round_robin 1 | 83.7 / 109.5 | 2674 | 34.57 | 164.7 |

Prefix hints increased sequential history reuse, but concurrent throughput
ranges overlap and tail latency did not consistently improve. Earlier variants
that stopped counting P load at the end of its RPC likewise failed to show a
stable concurrent benefit. These results do not justify enabling the policy by
default. This benchmark compares Xavier routing policies, not Xavier against
native vLLM NIXL.

Each deployment now has an isolated non-rotating evidence log. An earlier run
was discarded because shared log rotation lost phase counters despite successful
responses. A separate SGLang job was observed during the final deployment's
startup; exclusive hardware isolation was not established. With short measured
phases, colocated replicas and only two repetitions, these are exploratory
measurements, not evidence of four-GPU scaling or a general speedup. Follow-up
performance work needs an exclusive GPU window, longer runs and mixed prompt
lengths to distinguish scheduling and handoff costs from cache savings.

#### Sustained and mixed-length validation

Enable `XINFERENCE_TEST_PD_AFFINITY_LONG=1` for 2,400 concurrent requests per
profile, with fixed input length followed by mixed lengths (approximately
500, 900, 1,800 and 3,200 input tokens). Each profile first runs 12 cold and
36 shuffled repeated requests. Both profiles run in the same deployment;
cache contents persist across the workload transition. Four deployments use
the same ABBA order, seed, model and GPU budgets as above. An optional
`XINFERENCE_TEST_PD_AFFINITY_PROFILES=uniform` selects only fixed lengths;
`XINFERENCE_TEST_PD_AFFINITY_DOCUMENTS` and
`XINFERENCE_TEST_PD_AFFINITY_REQUESTS` override the working-set and concurrent
request counts (the latter must be a positive multiple of the former).

Long runs sample GPU process ownership once per second. Foreign GPU processes
invalidate the measurement; monitor failures also fail the test. Explicitly
allowlisted idle daemons can be recorded with
`XINFERENCE_TEST_PD_IDLE_GPU_PIDS=pid1,pid2`. The completed run recorded 608
samples without foreign GPU processes and passed all 19,584 requests. This is
sampled interference detection, not an exclusive hardware reservation.

| Policy and repetition | Profile | Requests/s | TTFT p95 (ms) | P RPC / post-P p50 (ms) |
| --- | --- | ---: | ---: | ---: |
| round_robin 0 | uniform | 34.68 | 154.3 | 37.7 / 64.7 |
| round_robin 0 | mixed | 34.79 | 173.7 | 33.2 / 64.5 |
| affinity 0 | uniform | 34.08 | 157.3 | 36.5 / 65.9 |
| affinity 0 | mixed | 34.01 | 177.8 | 34.4 / 65.7 |
| affinity 1 | uniform | 33.85 | 156.2 | 37.3 / 66.5 |
| affinity 1 | mixed | 33.75 | 182.0 | 34.3 / 66.0 |
| round_robin 1 | uniform | 33.61 | 159.1 | 38.7 / 66.7 |
| round_robin 1 | mixed | 33.71 | 179.1 | 33.9 / 66.2 |

Neither workload showed a consistent concurrent speedup. Fixed-length sequential
reuse did improve TTFT p95 (85.3–85.9 ms with affinity versus 107.4–109.6 ms
with round robin), but that did not translate into sustained throughput.
The post-P interval includes decode RPC/queueing, KV load and first execution;
it is not a pure transfer measurement. All methods use the same DEBUG evidence
logging, so absolute rates include instrumentation overhead. The working sets
and small model constrain the scope of these measurements.

### One producer and one decoder

Use two free NVIDIA GPUs and the pinned vLLM environment, with NIXL installed. Run from the repository root:

```bash
XINFERENCE_TEST_PD_GPU=1 python -m pytest -v \
  xinference/model/llm/vllm/xavier/test/test_pd_gpu.py
```

The test runs default Xavier direct handoff with tiered history, Xavier with history disabled, and native NIXL in real subprocesses, with one GPU per role. It checks streaming and non-streaming responses, repeated prompts and four concurrent requests, then terminates the deployment. Local prefix caching is disabled. Exact repeated-text checks are restricted to native NIXL and Xavier with history disabled; fixed-prefix and raw-bit history tests cover history reuse separately. Xavier requires producer registration and successful async completion logs from `get_finished`, and the default-history case requires a positive P-side history restore for each repeated prompt; native NIXL requires `calling _read_blocks` and completed receive logs, including the concurrent requests.

For locally cached weights, set `XINFERENCE_TEST_PD_MODEL_PATH`, `XINFERENCE_TEST_PD_MODEL_NAME` and `XINFERENCE_TEST_PD_MODEL_SIZE` to the path and registered model name/size. The manually triggered **PD GPU integration** GitHub Actions workflow runs the same test on a selected two-GPU runner.
