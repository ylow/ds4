# Phase 2a execution plan: F16 KV cache on CUDA (memory + long-context speed)

**Date:** 2026-06-25
**Goal:** cut KV-cache memory ~2× (toward the 600–800K-context goal) and reduce long-context
KV read bandwidth by storing the compressed attention KV (and later raw + indexer KV) as F16
instead of F32. Near-lossless. Precursor to FP8/TurboQuant (Phase 2b).

## What's already there vs the gap

- Macro **`DS4_GPU_ATTN_COMP_CACHE_F16`** (ds4.c:10296) = 1 on Apple/Metal (production), 0 on CUDA.
  ~25 sites in ds4.c (alloc, `attn_comp_stage` staging, save/load payload) already compile on CUDA.
- Probe result: enabling the macro for CUDA links after adding **one** helper
  (`ds4_gpu_tensor_copy_f32_to_f16` — trivial, reuse `f32_to_f16_kernel` at ds4_cuda.cu:3681),
  and **memory really drops** (ctx 4096 ctx-buffers 432→410 MiB; scales up a lot at long ctx).
- **The real gap:** NO CUDA attention kernel reads comp-KV as `__half`. The `comp_kv_f16` flag is
  threaded for Metal parity but every CUDA site bails (`if (comp_kv_f16) return 0;`). So the macro
  is **all-or-nothing**: it can't be enabled until every comp-KV reader supports `__half`.
  Confirmed failure with macro on: `attention batch encode failed` at layer 2 (first ratio-4 layer)
  prefill.

## The 7 kernels that read `comp_kv` (must gain a `__half` read path)

Scalar reads (`comp_kv[idx*head_dim + d]` — easy to template):
- `attention_prefill_mixed_kernel` (ds4_cuda.cu:4531) — reads at 4572, 4605
- `attention_decode_mixed_kernel` (4778) — 4854, 4875, 4930, 4940
- `attention_indexed_mixed_kernel` (4946) — 5040, 5090, 5100

float4-vectorized reads (`(const float4*)(comp_kv + ...)` — need a half-vectorized replacement,
e.g. load `int4`=8×`__half` and convert; these are the head_dim==512 hot path):
- `attention_indexed_mixed_heads8_rb4_kernel` (5106) — 5203, 5251
- `attention_indexed_mixed_heads8_online_kernel` (5282) — 5379
- `attention_static_mixed_heads8_online_kernel` (5447) — 5503
- `attention_decode_mixed_heads8_online_kernel` (5571) — 5668

Dispatch note: at SMALL n_comp the scalar `attention_decode_mixed_kernel` is used; the float4
heads8 kernels kick in at LARGE n_comp (`!cuda_attention_score_buffer_fits`) / head_dim==512. So the
scalar kernels alone let us validate F16 correctness at small context first.

## Approach

1. **Device load helper** (templated on comp dtype) so each kernel reads float or half uniformly:
   ```
   template<bool F16> __device__ __forceinline__ float ld1(const void* p, uint64_t i);
   template<bool F16> __device__ __forceinline__ void   ld4(const void* p, uint64_t i, float out[4]);
   ```
   F16 path: `__half2float`; float4 path reads `int4` (8 halves) → 8 floats (two ld4).
2. **Template each of the 7 kernels** on `bool COMP_F16` (and instantiate both). Change the
   `comp_kv` param to `const void*`. Dispatch picks the instantiation from `comp_kv_f16`.
   Keep raw_kv F32 for now (separate later step).
3. Add `ds4_gpu_tensor_copy_f32_to_f16` in ds4_cuda.cu (see probe; ~12 lines).
4. **Keep the macro OFF while editing** (all new code inert, repo keeps building/behaving
   identically). Flip `DS4_GPU_ATTN_COMP_CACHE_F16` for CUDA only once all 7 kernels are done:
   `#if defined(__APPLE__) || (!defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU))`.

## Validation (gate)

- **Correctness:** teacher-forced perplexity must stay within a small tolerance of the F32 ref
  `nll=317.843967992` (F16 is lossy, expect ~±0.0X, NOT bit-identical; flag if >~1 nll or NaN/crash).
  Test small ctx first (scalar kernels), then large ctx (heads8 float4 kernels), via
  `./ds4 --cuda --perplexity-file <txt> -n 256 -c <ctx>`.
- **Memory:** confirm ctx-buffer MiB drop; measure at 128K/256K with a long prompt.
- **Speed:** ds4-bench decode tok/s at long ctx — expect a *gain* (less KV bandwidth in
  attention; the indexer KV is still F32 until its own step).
- Regression: SSD-streaming + CPU build unaffected (macro gated to CUDA/Metal).

## Follow-on steps (separate, after comp-attn-KV lands)

- **Indexer KV (128-dim) → F16**: the long-context indexer-scoring bandwidth hotspot; the WMMA
  indexer kernels already `__float2half` internally, so storing F16 lets them read half directly.
  6 indexer kernels (indexer_scores_kernel + 4 WMMA variants + indexer_score_one_direct).
- **Raw KV → F16**: already F16-rounded (lossless); read by the same attention kernels (extend the
  template to raw_kv too).
- **Phase 2b — FP8 / TurboQuant (~3-bit)** on the 512-dim compressed rows: add parallel per-row
  scale buffers + dequant in the read kernels. Keep RoPE-key and 128-dim indexer higher precision.

## Status — Phase 2a LANDED (commit 08ad8e5)

Compressed attention KV is now F16 on CUDA, via **expand-on-read** (lower risk than the 7-kernel
in-kernel __half rewrite above): the 5 extern attention entry points dequantize the F16 comp rows
to a reused F32 scratch (`cuda_expand_comp_f16_to_f32` + `f16_to_f32_kernel`) and run the existing
F32 kernels unchanged; write side uses `ds4_gpu_tensor_copy_f32_to_f16`. Validated:
- Correctness: teacher-forced perplexity nll 317.843968 (F32) → 317.842979 (F16), ~3e-6 relative
  (pure F16 rounding). Near-lossless.
- Memory: ctx-buffers 432 → 410 MiB at ctx 4096 = exactly the ratio-4 comp cache halving
  (1026 rows × 512 × 2 bytes × 21 layers); scales to ~4.3 GB saved at 800K.
- Speed: ~neutral at normal context (13.36 → 13.22 tok/s @2048; 12.90 @8192). Expand overhead
  grows at extreme length.

### Remaining follow-ons (in priority order)
1. **gather-of-selected** expand (indexed layers): dequantize only the top-k selected rows + raw
   window instead of all n_comp — removes the long-context expand overhead so 800K decode stays
   fast. (Sequential ratio-128 layers can expand [0,n_comp) as today.)
2. **Indexer KV (128-dim) → F16**: the long-context indexer-scoring bandwidth hotspot; the WMMA
   indexer kernels already `__float2half` internally.
3. **Raw KV → F16**: already F16-rounded (lossless).
4. **In-kernel __half comp reads** (the templating plan above): recovers attention read bandwidth
   and removes the expand pass entirely.
5. **Phase 2b — FP8 / TurboQuant ~3-bit** on the 512-dim compressed rows (parallel scale buffers).
