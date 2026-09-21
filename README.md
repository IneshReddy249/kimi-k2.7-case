# Kimi-K2.7-Code: Execution Graph and Deployment Case

Predicted decode-step execution of `moonshotai/Kimi-K2.7-Code` on one 8×H200 node (TP8),
reconciled against vLLM, with a plan for the first 48 hours of a live engagement.

- **Checkpoint:** `moonshotai/Kimi-K2.7-Code @ 74797c9c62378b951a1f6fcf5c4631024e9b8bef`
- **Engine:** vLLM `v0.19.1` (`b1388b1fbf5aaef47937fabe98931211684666a6`)
- **Sources:** `config.json`, `model.safetensors.index.json`, the checkpoint's modeling files, and vLLM source at the pinned tag. No traces, no GPU runs, no third-party writeups.

## Files

| File | What it is |
|---|---|
| `kimi-k2.7-code.yaml` | Model spec in the GitM planner schema |
| `Kimi-K2.7-Code — Design Note (Part 1).md` | Per-GPU memory ledger, fit verdict, ranked top-3 time consumers, expert-bank arithmetic |
| `Checkpoint Versus vLLM Execution (Part 2).md` | What the engine decides that the checkpoint cannot, with cited code paths |
| `KIMI-K2.7-PART-3.md` | The first 48 hours: information requests, baseline, first intervention, rollback, when to hold still |
| `AI-DISCLOSURE.md` | What the agent wrote, what I verified by hand, what I am least sure of |

## Key findings

1. **The checkpoint supports FP8 nowhere.** Routed experts are INT4 (group 32, symmetric), everything else is BF16, and `kv_cache_scheme` is null. That is why the model fits on one node: about 88–104 GB in use per GPU out of 141 GB.
2. **Decode is memory-bound.** Routed expert weight reads (~7.3 ms floor) rank first, then KV cache reads (~3.8 ms at BF16). No compute-bound nodes.
3. **Communication is latency-bound, not bandwidth-bound.** 123 all-reduces of 0.46 MB per step. Where it ranks depends on the per-call latency, which the checkpoint cannot give and must be measured.
4. **The MLA KV cache is replicated per GPU, not sharded.** The latent has no head axis to split.

## Reproduce the planner cross-check

```bash
git clone https://github.com/GitM-Labs/runtime.git && cd runtime
pip install -e .
cp /path/to/kimi-k2.7-code.yaml gitm/planner/models/
gitm plan kimi-k2.7-code --gpu H200 --batch 32 --kv-len 8192 --tp 8
```

All times are roofline floors at datasheet bandwidth, not measurements.
