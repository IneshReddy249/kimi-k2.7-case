# Kimi-K2.7-Code: Checkpoint Versus vLLM Execution (Part 2)

## 1. Pinned engine and layout

- Checkpoint: [`moonshotai/Kimi-K2.7-Code @ 74797c9c62378b951a1f6fcf5c4631024e9b8bef`](https://huggingface.co/moonshotai/Kimi-K2.7-Code/tree/74797c9c62378b951a1f6fcf5c4631024e9b8bef)
- Engine: [`vLLM v0.19.1`, commit `b1388b1fbf5aaef47937fabe98931211684666a6`](https://github.com/vllm-project/vllm/tree/b1388b1fbf5aaef47937fabe98931211684666a6)
- Layout: one 8×H200 node, TP8, EP1, PP1, text only, speculation disabled.

The checkpoint's deployment guide and the [official vLLM recipe](https://recipes.vllm.ai/moonshotai/Kimi-K2.7-Code.json) support vLLM 0.19.1 and single-node TP8 on H200 without `--enable-expert-parallel`. For this text-only case I add `--language-model-only` and omit the recipe's optional speculative-decoding argument. In the pinned source, `--language-model-only` disables multimodal inputs; I did not find proof that it skips loading the vision weights, so vision-weight residency remains **assumed**.

Labels used below:

- **source**: read from the pinned vLLM source.
- **docs**: read from the recipe, deploy guide or source docstring.
- **assumed**: not established by checkpoint or static source reading.

## 2. Engine decisions the checkpoint cannot make

| Decision | Evidence | Pinned vLLM code path | Change to the predicted graph |
|---|---|---|---|
| Text implementation | source | [`kimi_k25.py::KimiK25ForConditionalGeneration.__init__`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/kimi_k25.py#L317-L371) | vLLM builds Kimi's text backbone using its native `DeepseekV2ForCausalLM`, rather than executing the checkpoint's Python modeling file directly. |
| TP versus EP experts | docs + source | [`parallel.py::enable_expert_parallel`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/parallel.py#L130-L170), [`DeepseekV2MoE.forward`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/models/deepseek_v2.py#L348-L398) | EP is off. Every routed expert is TP-sharded eight ways. There is no EP dispatch/combine all-to-all; routed and shared outputs are added and reduced once per MoE layer. |
| INT4 expert kernel and runtime precision | source | [`CompressedTensorsMoEMethod.get_moe_method`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py#L119-L197), [`is_flashinfer_mxint4_moe_available`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/quantization/utils/flashinfer_mxint4_moe.py#L20-L32) | FlashInfer's INT4 MoE branch requires Blackwell capability 10.0, so H200 uses the supported Marlin W4A16 path. Storage is INT4 and activations are BF16; the kernel's internal accumulation precision is not established here. Repacked resident bytes remain **assumed**. |
| KV-cache precision | source | [`cache.py::CacheConfig`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/cache.py#L35-L60), [`MLAAttention.__init__`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/attention/mla_attention.py#L310-L369) | With no `--kv-cache-dtype`, `auto` resolves to model dtype, so this layout uses BF16 KV: 18.42 GB per GPU for the stated workload. FP8 KV requires an explicit engine choice. |
| KV-cache placement | source-derived | [`MLAAttention.get_kv_cache_spec`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/attention/mla_attention.py#L830-L845) | The cache spec has one KV head with width `512 + 64 = 576`. With TP8 and no context parallelism, each TP rank allocates this cache; it is not divided by eight. |
| MLA backend | source | [`cuda.py::_get_backend_priorities`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/platforms/cuda.py#L50-L113), [`get_attn_backend_cls`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/platforms/cuda.py#L248-L311) | On H200, vLLM tries `FLASH_ATTN_MLA`, `FLASHMLA`, `FLASHINFER_MLA`, then `TRITON_MLA`, choosing the first installed valid backend. BF16 makes `FLASH_ATTN_MLA` the expected first choice. FP8 excludes it, making `FLASHMLA` the expected next choice. Startup logs must confirm what is installed and selected. |
| MLA decode and weight absorption | source | [`MLAAttention.forward_impl`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/attention/mla_attention.py#L535-L708), [`process_weights_after_loading`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/attention/mla_attention.py#L710-L760) | vLLM separates `kv_b_proj` into `W_UK_T` and `W_UV`. During decode it transforms the current query, attends directly over the compressed cache, and applies value expansion after attention. It does not run `kv_b_proj` over every cached token. |
| CUDA graph versus eager | source + docs | [`CompilationConfig.cudagraph_mode`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/config/compilation.py#L550-L582), [`CudagraphDispatcher.dispatch`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/cudagraph_dispatcher.py#L234-L323) | V1 defaults to full graphs for captured uniform-decode shapes and piecewise graphs for other supported batches. In piecewise mode, attention is outside the captured graph pieces. Batch 32 being captured is **assumed** until confirmed from logs. |
| TP collective implementation | source-derived under default flags | [`CudaCommunicator.all_reduce`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/device_communicators/cuda_communicator.py#L180-L230), [`CustomAllreduce.should_custom_ar`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/distributed/device_communicators/custom_all_reduce.py#L233-L246) | On a fully connected eight-H200 node, the 0.46 MB payload meets the default custom-all-reduce conditions, so that path is expected. It can be disabled by configuration; logs or a trace must confirm it. The fixed latency `L` must be measured. |
| Embedding and logits communication | source | [`VocabParallelEmbedding.forward_native`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/vocab_parallel_embedding.py#L464-L487), [`LogitsProcessor._gather_logits`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/model_executor/layers/logits_processor.py#L75-L104) | The vocabulary-parallel embedding adds one TP all-reduce per step. The TP-sharded `lm_head` is followed by a logits gather before sampling. |
| Memory budget | source | [`GPUWorker.determine_available_memory`](https://github.com/vllm-project/vllm/blob/b1388b1fbf5aaef47937fabe98931211684666a6/vllm/v1/worker/gpu_worker.py#L332-L410) | Default `gpu_memory_utilization=0.9` excludes roughly 10% of HBM from the engine's budget. After profiling, the remaining permitted memory becomes the paged KV pool; 18.42 GB is the occupied cache for this workload, not necessarily the whole allocated pool. |

## 3. Ordered vLLM execution flow

1. **Create GPU workers.** vLLM starts one worker on each H200 and forms one TP8 communication group. Each worker selects its CUDA device and initializes GPU communication.
2. **Build and load the model.** The Kimi wrapper creates the `DeepseekV2ForCausalLM` text backbone. Each worker loads its TP shard: approximately one eighth of each sharded matrix, including every routed expert.
3. **Prepare INT4 experts.** Expert weights remain INT4 with group-32 BF16 scales and BF16 activations. vLLM repacks them for the Marlin W4A16 MoE kernel selected on H200.
4. **Prepare absorbed MLA weights.** At load time, `kv_b_proj` is separated into `W_UK_T` and `W_UV`, allowing decode to use the compressed cache directly.
5. **Select the MLA backend.** vLLM checks hardware capability, KV datatype, head shape, block size and installed libraries, then selects the first valid backend.
6. **Profile and allocate memory.** A dummy forward measures activation, workspace, communication and CUDA-graph memory. Remaining permitted HBM is allocated as a paged KV-cache pool.
7. **Warm and capture.** vLLM warms or compiles kernels, prepares communication buffers and captures supported fixed decode shapes as CUDA graphs.
8. **Schedule the decode batch.** After prefill has created the caches, the scheduler selects 32 active requests and prepares one current token, position and KV-block mapping for each. Every request already has 8,192 cached tokens.
9. **Embed the token IDs.** Each GPU computes its vocabulary-shard contribution. One TP all-reduce creates the complete `[32, 7168]` hidden-state tensor on all ranks.
10. **Run MLA in all 61 layers.** Each layer writes one new 576-element compressed cache entry per request, attends over the paged cache through the absorbed decode path, applies `o_proj`, then performs a TP all-reduce.
11. **Run FFN or MoE.** Layer 0 executes the dense FFN. In layers 1–60 the router selects top-8 experts per token; every GPU computes its TP shard of those experts and the shared expert, followed by one TP all-reduce.
12. **Produce the next tokens.** Final RMSNorm and the TP-sharded `lm_head` produce vocabulary shards. vLLM gathers the logits, samples 32 tokens, appends them to the requests, advances each KV length by one and repeats decode.

## 4. Material corrections to a checkpoint-only graph

### Correction 1: MLA does not expand cached K and V during decode

A naive graph executes `kv_b_proj` over all cached positions. vLLM instead absorbs `W_UK_T` into the current-query side, attends directly over the compressed cache, and applies `W_UV` only to the final attention result. Expanding K and V for `32 × 8,192` cached positions would therefore be materially wrong.

### Correction 2: communication is 123 all-reduces, not 122

```text
1 embedding all-reduce
+ 61 attention all-reduces
+ 61 FFN/MoE all-reduces
= 123 TP all-reduces
+ 1 final logits gather
```

This corrects Part 1. Timing becomes `123 × L` plus the gather. The ranking thresholds barely move: `3.8 ms / 123 ≈ 31 µs` and `7.3 ms / 123 ≈ 59 µs`.

### Correction 3: FP8 KV changes more than memory traffic

A checkpoint-only graph may treat KV datatype as only a bytes-per-element multiplier. In vLLM it also changes backend eligibility: `FLASH_ATTN_MLA` accepts BF16/FP16 KV but not FP8, so FP8 moves selection to the next installed valid backend, expected to be `FLASHMLA` on H200.

### Correction 4: the MLA cache is not TP-sharded by eight

The MLA cache has one shared 576-element entry per token per layer. With TP8 and no context parallelism, each TP rank allocates that cache: 18.42 GB per GPU at BF16, not 2.30 GB.

## 5. Still not determined without runtime evidence

- The measured fixed collective latency `L` for the selected all-reduce path.
- Which MLA backends are installed in the operator's build and which one the startup selector chooses.
- Whether batch size 32 is present in the captured CUDA-graph shapes under the operator's settings.
- Final resident bytes after Marlin repacking.
- Whether `--language-model-only` keeps the vision weights off GPU memory in this exact launch path.
- The operator's real launch arguments. Part 3 should request them before any intervention.
