# AI Disclosure and Uncertainty

## AI assistance

I used an AI agent to help inspect the checkpoint and pinned vLLM source, draft
the YAML and written sections, and check the arithmetic. The agent parsed
`config.json`, `model.safetensors.index.json`, selected safetensors headers and
the modeling files. It also located the vLLM code paths cited in Part 2 and ran
GitM's `load_spec` and planner commands. No GPU execution or runtime trace was
performed.

## What I verified manually

I manually checked:

- The 61-layer schedule: layer 0 dense and layers 1–60 MoE.
- MLA KV width: `512 + 64 = 576` elements per token per layer.
- BF16 KV size: `576 × 61 × 32 × 8,192 × 2 B = 18.42 GB` per GPU.
- One routed expert: `7168 × 2048 × 3 × 0.5625 B = 24.77 MB`.
- Total routed-expert storage: `24.77 MB × 384 × 60 = 570.76 GB`.
- Expected distinct experts:
  `384 × (1 − (1 − 8/384)^32) = 188.23`.
- Expert-read bounds and the 7.29 ms expected HBM floor.
- TP communication: 123 all-reduces per decode step plus one logits gather.
- Communication ranking thresholds: approximately 31 µs against KV reads and
  59 µs against expected expert reads.

I also opened the pinned Part 2 source links and checked that the cited functions
make the decisions claimed in the table. Engine behavior was read statically;
it was not confirmed with a runtime trace.

## Remaining uncertainty

- Actual collective latency `L` must be measured on the target H200 node.
- Real expert selections depend on production tokens; 188 assumes uniform,
  independent routing.
- Workspace, CUDA-graph and communication-buffer memory are estimates.
- The installed attention backend, captured batch shapes, final Marlin memory
  layout and collective fallback path depend on the runtime environment.
- If the operator's launch arguments differ from the assumed TP8, EP1, PP1,
  BF16-KV configuration, the execution graph and ranking must be recalculated.
