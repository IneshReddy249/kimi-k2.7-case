# Kimi-K2.7-Code: The First 48 Hours (Part 3)

**Given:** the replica hardware, TP/EP/PP layout, concurrency band, latency SLOs and read-only telemetry.

**Rule:** I do not change production until the baseline can separate a real improvement from normal traffic noise.

## 1. Minimum information I request

| What I request | Why it is load-bearing |
|---|---|
| **Exact serving configuration:** vLLM version or commit, complete launch command, container image, model revision, KV dtype, CUDA-graph settings and GPU/NVLink topology | These settings decide the real execution graph. If EP, FP8 KV, speculation or graph mode differs from Parts 1–2, my bottleneck ranking describes a different system. |
| **Real traffic shape:** request rate, prompt/output-length distributions, concurrency over time, burstiness, prefix reuse and the fraction of prefill, decode and mixed steps | The Part 1 workload—32 requests at 8,192 cached tokens—is only one operating point. Real traffic determines batching, KV pressure, collective sizes and whether a benchmark result will survive production. |
| **Exact SLO definition:** TTFT, TPOT and end-to-end targets; percentile; client- or server-side measurement; evaluation window and error budget | “p99 latency” is not actionable without saying which latency, where it is measured and over what window. These definitions become the test guardrails and rollback triggers. |
| **Safe experiment path:** whether one replica can be a canary, how requests are randomized between canary and control, who can restart it and who owns rollback | A before/after comparison can confuse traffic drift with improvement. Without a concurrent control and an agreed rollback path, I will not change production. |

## 2. Baseline before any change

I first record request rate, queue time, TTFT, TPOT and end-to-end latency at p50, p95 and p99. I also record tokens/s, batch size, prefill/decode/mixed-step share, KV-pool use, GPU utilization, HBM use, preemptions, retries, errors and cancellations. If existing telemetry or an operator-approved trace exposes it, I also measure collective latency `L`, CUDA launch gaps and time per decode step.

I compare only matched traffic regimes—for example, the same prompt-length, concurrency and burstiness buckets. I collect repeated peak and off-peak windows and require a pre-agreed sample size. A concurrent **A/A test** between two unchanged replicas establishes the noise floor. A later difference smaller than that A/A spread is not treated as an improvement.

To detect contamination, I check that collecting telemetry does not change the operator's existing SLO dashboard. I also track GPU clocks, temperature, power state and co-tenant activity so a faster or slower node is not mistaken for a server change. Client latency is separated into queue, network and server time.

## 3. First intervention

I test one change only: on the canary, compare vLLM's default custom all-reduce path with `--disable-custom-all-reduce`. I confirm the actual default and fallback paths from startup logs or a trace; I do not assume the fallback is NCCL or symmetric memory.

I run this test only if the baseline shows that communication is material—approximately `L > 31 µs`, above the Part 1 KV-read time divided by the corrected **123 TP all-reduces per decode step**:

```text
1 embedding all-reduce
+ 61 attention all-reduces
+ 61 FFN/MoE all-reduces
= 123 all-reduces
+ 1 logits gather
```

This intervention attacks the communication-latency term from the Part 1 ranking. If the alternative reduces each collective by `ΔL`, the approximate decode-step saving is `123 × ΔL`; saving 10 µs per collective would save about 1.23 ms per step. TPOT should improve most. TTFT may also change because prefill uses TP collectives, so it remains a guardrail.

The canary and unchanged control receive randomized, comparable live traffic. I report results separately for decode-only, prefill and mixed regimes. This prevents a benchmark-only win: a small decode collective may use custom all-reduce, while a much larger mixed-step collective may take another path.

### Guardrails and rollback

- TPOT p99, TTFT p99 and end-to-end p99 must remain inside their defined SLOs.
- Error, retry and preemption rates must not worsen beyond the A/A noise floor.
- Memory headroom must remain safe.
- Deterministic greedy decoding on a fixed prompt set must produce the same tokens; optional logit comparisons use a defined numerical tolerance.
- Roll back immediately for a new collective failure, correctness failure or SLO breach.
- Roll back after the agreed sample size if the gain is absent, smaller than the A/A noise floor or limited to a traffic regime that is unimportant in production.

Rollback means restoring the original launch setting and restarting only the canary.

## 4. When I decline to intervene

I make no production change in the first 48 hours when:

- The service already meets every SLO with adequate margin.
- Telemetry cannot separate queue, network and server time.
- The noise floor is as large as the expected improvement.
- There is no canary, concurrent control or named rollback owner.
- The actual configuration differs materially from Parts 1–2; I re-derive the graph first.
- Communication is not a measured bottleneck.
- Traffic during the observation window is not representative.
- The dominant cost is expert-weight or KV traffic and the safe recoverable portion is too small to justify production risk.

In that case I tell the operator plainly: **the current evidence does not justify a production change.** I provide the measured bottleneck, the maximum recoverable time and the missing evidence required before testing.

## Source evidence

- vLLM pinned revision: [`v0.19.1 / b1388b1`](https://github.com/vllm-project/vllm/tree/b1388b1fbf5aaef47937fabe98931211684666a6)
- Collective selection and fallback: [`CudaCommunicator.all_reduce`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/device_communicators/cuda_communicator.py#L180-L230)
- Custom-all-reduce eligibility: [`CustomAllreduce.should_custom_ar`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/device_communicators/custom_all_reduce.py#L233-L246)
- CUDA-graph dispatch: [`CudagraphDispatcher.dispatch`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/cudagraph_dispatcher.py#L234-L323)
