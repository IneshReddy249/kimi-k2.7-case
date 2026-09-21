# Kimi-K2.7-Code — Design Note (Part 1)

## Scope

Official checkpoint: [`moonshotai/Kimi-K2.7-Code`](https://huggingface.co/moonshotai/Kimi-K2.7-Code/tree/74797c9c62378b951a1f6fcf5c4631024e9b8bef)  
Revision: `74797c9c62378b951a1f6fcf5c4631024e9b8bef`

Sources: `config.json`, `model.safetensors.index.json`, `modeling_kimi_k25.py`, `modeling_deepseek.py`, and the checkpoint shard header. No traces were used.

Baseline: 8×H200 SXM, TP8, EP1, PP1, 32 decode sequences, 8,192 cached tokens each, one new token per sequence, text only, speculation disabled.

Tags: **[checkpoint]** comes from checkpoint files, **[assumed]** is stated explicitly, and **[engine]** is decided by vLLM and handled in Part 2.

Planner command:

```bash
gitm plan kimi-k2.7-code --gpu H200 --batch 32 --kv-len 8192 --tp 8 --ep 1
```

## 1. Checkpoint facts

- **[checkpoint]** 61 layers, hidden size 7,168, 64 attention heads, vocabulary 163,840.
- **[checkpoint]** MLA dimensions: `q_lora_rank=1536`, `kv_lora_rank=512`, `qk_nope_head_dim=128`, `qk_rope_head_dim=64`, `v_head_dim=128`.
- **[checkpoint]** Layer 0 has a dense FFN of width 18,432. Layers 1–60 are MoE with 384 routed experts, top-8 routing, one shared expert, and expert width 2,048.
- **[checkpoint]** Routed experts are symmetric INT4, group size 32, with BF16 scales. Attention, the dense FFN, shared experts, embeddings and `lm_head` are BF16.
- **[checkpoint]** `kv_cache_scheme` is null. Therefore BF16 KV is the checkpoint-supported baseline; FP8 KV would be an engine choice.
- **[checkpoint]** MLA stores one 512-element latent plus one 64-element RoPE key: **576 elements per token per layer**. `num_key_value_heads=64` must not be used as a GQA cache shape.

The submitted YAML contains the complete planner schema and provenance.

## 2. Does TP8 fit?

All sizes below are decimal GB.

### Expert weights

```text
weights per expert = 7168 × 2048 × 3
                   = 44,040,192

bytes per weight   = 0.5 INT4 + 2/32 BF16 scale
                   = 0.5625 B

one expert         = 44,040,192 × 0.5625
                   = 24.77 MB

all routed experts = 24.77 MB × 384 × 60
                   = 570.76 GB

per GPU under TP8  = 570.76 / 8
                   = 71.35 GB
```

The shard header confirms `weight_scale` is BF16 with shape `[2048, 224]`, where `224 = 7168/32`.

The checkpoint index reports **595.15 GB total**. The remaining checkpoint weights are:

```text
595.15 − 570.76 = 24.39 GB
```

### KV cache

```text
tokens             = 32 × 8192 = 262,144
elements           = 262,144 × 61 × 576
BF16 KV bytes      = elements × 2
                   = 18.42 GB per GPU
```

**[engine]** Ordinary head-based TP cannot naturally divide the shared MLA latent, so this note assumes the cache is replicated on each TP rank. Part 2 verifies the actual vLLM placement.

### Per-GPU memory ledger

| Item | Capacity lower bound | Conservative deployable case |
|---|---:|---:|
| Routed experts, including scales | 71.35 GB | 71.35 GB |
| Remaining checkpoint weights | 3.05 GB (`24.39/8`) | 24.39 GB, all replicated |
| BF16 KV cache | 18.42 GB | 18.42 GB |
| Workspace | 0 | 6.00 GB **[assumed]** |
| Communication buffers + CUDA context | 0 | 2.00 GB **[assumed]** |
| Explicit reserve | 0 | 14.10 GB |
| **Total** | **92.82 GB** | **136.26 GB** |

Each H200 has 141 GB. The conservative case remains below capacity, so **one 8×H200 node fits**.

TP4 cannot fit because routed experts alone require:

```text
570.76 / 4 = 142.69 GB per GPU
```

That already exceeds 141 GB before other weights or KV. Therefore the smallest documented single-node TP layout is **TP8, EP1, PP1**.

## 3. Predicted decode flow

```text
token embedding
→ repeat for layers 0–60:
    RMSNorm
    → MLA projections
    → append 576-element KV entry
    → attention over 8,192 cached tokens
    → output projection
    → TP all-reduce
    → RMSNorm
    → layer 0: dense FFN
      layers 1–60: router → top-8 routed experts + shared expert
    → TP all-reduce
→ final RMSNorm
→ lm_head
→ logits gather and sampling
```

Precision by operation:

- Routed-expert weights: INT4 with BF16 scales; BF16 activations.
- All other model weights and activations: BF16.
- KV cache: BF16 for Part 1.

## 4. Communication

**[assumed]** Standard TP uses two all-reduces per layer:

```text
count            = 61 × 2 = 122 all-reduces per step
payload          = 32 × 7168 × 2 B = 0.459 MB each
ring bytes/rank  = 2 × (7/8) × 0.459 MB = 0.803 MB
bandwidth floor  = 0.803 MB / 450 GB/s = 1.8 µs each
```

Total NVLink bandwidth floor is about **0.22 ms**. Real cost is dominated by fixed collective latency `L`, approximately `122 × L`.

There is no inter-node communication in this layout:

```text
NVLink traffic: present, 122 TP all-reduces
Inter-node traffic: 0 bytes, 0 ms
```

## 5. Expert-bank reads

Selected expert-weight bytes for one token in one MoE layer:

```text
8 × 24.77 MB = 198.2 MB across the full model
198.2 / 8    = 24.77 MB per TP rank
```

For 32 tokens, there are 256 expert selections. **[assumed]** If multiple tokens select the same expert, its weights are read once for that batch.

| Routing case | Distinct experts/layer | Per-GPU read across 60 layers | HBM floor at 4.8 TB/s |
|---|---:|---:|---:|
| All tokens select the same eight | 8 | 1.49 GB | 0.31 ms |
| Uniform independent routing | 188 | 34.97 GB | 7.29 ms |
| No overlap between selections | 256 | 47.56 GB | 9.91 ms |

Expected distinct experts:

```text
384 × (1 − (1 − 8/384)^32) = 188.23
```

Real routing is data-dependent; 188 is an assumed expectation, not a bound.

## 6. Ranked top three

| Rank | Consumer | Bound | Time |
|---:|---|---|---:|
| 1 | Routed-expert weight reads | Memory-bound | 7.29 ms expected; 0.31–9.91 ms bounds |
| 2 | BF16 KV-cache reads | Memory-bound | `18.42 GB / 4.8 TB/s = 3.84 ms` |
| 3 | TP all-reduces | Communication latency-bound | 2.44 ms with **[assumed]** `L=20 µs` |

Communication is third under the stated `20 µs` assumption. It overtakes KV reads when `L > 3.84 ms / 122 = 31 µs`, and expert reads when `L > 7.29 ms / 122 = 60 µs`. Part 2 identifies the vLLM collective path; a trace is required to measure `L`.

GitM's roofline output for the submitted YAML gives approximately 7.29 ms for routed-expert reads, 3.84 ms for KV reads, and a 13.3 ms total floor. These are lower bounds, not measured latency.

## 7. Not determined by the checkpoint

- KV-cache physical placement and optional FP8 KV.
- TP versus EP expert execution and the exact collective implementation.
- MLA absorption and attention backend.
- CUDA graphs versus eager execution.
- Actual distinct experts selected by the workload.
- Actual collective latency.

## 8. AI disclosure

AI assistance was used to draft the YAML and design note and to check arithmetic. I manually rechecked the checkpoint fields, expert-size calculation, KV-size calculation, and distinct-expert bounds. I did not verify runtime engine behavior; those items remain marked **[engine]** or **[assumed]**.
