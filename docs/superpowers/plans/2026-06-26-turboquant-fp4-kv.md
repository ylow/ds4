# Hadamard-FP4 (TurboQuant) Compressed-Attention KV — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store the 448 NoPE dims of the 512-dim compressed-attention KV rows as Hadamard-64-rotated E2M1 FP4 (4-bit) + per-64-group int8 exponent (RoPE-64 tail stays F16), cutting that cache from 584 → 360 B/row (5.69× vs F32) as the memory lever toward 600–800K context.

**Architecture:** Extends the Phase 2b FP8-split path. Storage adds a new `comp_kv_dtype == 3` (FP4). Read = expand-on-read: new kernels decode nibble→`e2m1·2^k`, re-apply the self-inverse normalized Hadamard-64 to un-rotate into the *same reused F32 scratch* (`g_comp_f32_expand_*`), so the trusted F32 attention kernels are byte-identical and untouched. Write = a new commit kernel (the only write change) Hadamard-rotates the staged F32, FP4-packs nibbles + frexp-exact exponent. RoPE-tail F16 buffer and per-64-group int8 scale buffer are reused verbatim from FP8. Gated by a new `DS4_GPU_ATTN_COMP_CACHE_FP4` macro that takes precedence over FP8 → F16 → F32.

**Tech Stack:** C (ds4.c), CUDA (ds4_cuda.cu), the model's own E2M1 codec (`dsv4_e2m1fn_value_*`) + a 64-wide Walsh–Hadamard transform. Build: `make ds4 ds4-bench CUDA_ARCH=` during iteration, `make cuda-spark` to finish.

## Global Constraints

- **Codec:** Hadamard-64 + E2M1 FP4 on the 448 NoPE dims; RoPE-64 tail F16; indexer 128-dim cache untouched. Copy exact magnitude table `{0,.5,1,1.5,2,3,4,6}`, max 6.0, nibble `(sign<<3)|index`.
- **Scale granularity:** per-64-group (one int8 exponent per Hadamard block, 7 used/row), reusing the existing 8/row `layer_attn_comp_scale` buffer.
- **Exact exponent:** `k = ceil(log2(amax/6))` computed via `frexpf` (host & device identical) — never `(int)ceilf(log2f(...))`. `int e2; float fr = frexpf(amax/6, &e2); k = (fr==0.5f)?(e2-1):e2;`
- **Hadamard normalization:** `1/sqrt(64) = 0.125f`; the normalized transform is self-inverse, used for both rotate (write) and un-rotate (read). Host and device must use the identical butterfly stride order (no FMA) so the value self-check is bit-exact.
- **FP4 amax floor:** `7.052966104933725e-38f` (the model's FP4 floor, cf. `dsv4_fp4_act_quantize_row_inplace_cpu`).
- **Row layout (FP4 mode):** `layer_attn_comp_cache` = 224 B/row nibbles (`n_nope/2`, 2 dims/byte: even local dim → low nibble, odd → high); `layer_attn_comp_scale` = 8 B/row; `layer_attn_comp_rope` = 128 B/row F16. Total 360 B/row.
- **Precedence:** every attn-comp site checks FP4 first, then FP8, then F16, then F32. `metal_graph_attn_comp_cache_dtype()` returns 3 in FP4 mode.
- **Correctness gate (per commit):** `./ds4 --cuda --perplexity-file doors-of-stone-chapter-1.md -n 256 -c 4096`. Baselines: F32=317.843968, current FP8/F16 build=**317.842979423** (context buffers 396.21 MiB @4096). FP4 accept gate (Task 4 only): **nll ≤ 319.4**, value self-check passes, no NaN, 16384 long-ctx stable.
- **Process hygiene:** ONE ds4 process at a time (instance lock); kill by exact PID, never `pkill -f ds4-server`. Don't `make cpu` (overwrites CUDA binaries). Model loads ~10s (86 GB) — run ds4 in the background and poll. Greedy decode is NOT a valid oracle; only teacher-forced perplexity is bit-deterministic.

---

## Task 1: Read path — FP4 expand/gather kernels + entry-point branches (CUDA only)

All changes in `ds4_cuda.cu`. Inert because nothing returns `comp_kv_dtype == 3` until Task 4 flips the macro, so the live path stays FP8 and nll must be unchanged. This is the bulk of the churn (mirrors Phase 2b commit `582c3ad`).

**Files:**
- Modify: `ds4_cuda.cu` — codec primitives (after `dsv4_e2m1fn_dequant_dev`, ~line 4563), expand/gather kernels (after `gather_comp_fp8_to_f32_kernel`, ~line 4451), launchers (after `cuda_gather_comp_fp8_to_f32`, ~line 9135), and `== 3u` branches in the 5 attention entry points.

**Interfaces:**
- Produces (file-internal): `__device__ dsv4_e2m1fn_decode_nibble_dev(uint8_t)`, `__device__ hadamard64_shared(float *vals, uint32_t tid)`, kernels `expand_comp_fp4_to_f32_kernel` / `gather_comp_fp4_to_f32_kernel`, launchers `cuda_expand_comp_fp4_to_f32(...)` / `cuda_gather_comp_fp4_to_f32(...)` returning `const ds4_gpu_tensor *` (the reused F32 scratch view) like their FP8 siblings.

- [ ] **Step 1: Add the device decode + Hadamard-64 primitives.** In `ds4_cuda.cu`, immediately after `dsv4_e2m1fn_dequant_dev` (ends ~line 4563), insert:

```c
/* Decode a stored E2M1FN nibble back to float: bit3 = sign, bits0..2 = magnitude
 * index into dsv4_e2m1fn_value_dev.  Sibling of dsv4_e4m3fn_decode_byte_dev. */
__device__ static float dsv4_e2m1fn_decode_nibble_dev(uint8_t nib) {
    float mag = dsv4_e2m1fn_value_dev((int)(nib & 0x7u));
    return (nib & 0x8u) ? -mag : mag;
}

/* In-place 64-wide normalized Walsh-Hadamard on shared vals[0..63] across exactly 64
 * threads (tid 0..63).  Normalized by 1/sqrt(64)=0.125, so (H64*0.125)^2 = I: the SAME
 * call rotates on write and un-rotates on read.  Butterfly stride order matches
 * dsv4_hadamard64_inplace_cpu bit-for-bit (plain add/sub, no FMA).  Ends syncthreaded. */
__device__ static void hadamard64_shared(float *vals, uint32_t tid) {
    for (uint32_t stride = 1u; stride < 64u; stride <<= 1u) {
        if ((tid & stride) == 0u) {
            uint32_t base = (tid & ~(2u * stride - 1u)) + (tid & (stride - 1u));
            float a = vals[base];
            float b = vals[base + stride];
            vals[base]          = a + b;
            vals[base + stride] = a - b;
        }
        __syncthreads();
    }
    vals[tid] *= 0.125f;
    __syncthreads();
}
```

- [ ] **Step 2: Add the expand + gather kernels.** In `ds4_cuda.cu`, after `gather_comp_fp8_to_f32_kernel` (ends ~line 4451), insert:

```c
/* Expand all n_comp Hadamard-FP4 rows to F32.  One block per row, 64 threads.  NoPE: per
 * 64-group g, unpack nibble -> dsv4_e2m1fn_decode_nibble_dev * 2^exp[g] (rotated domain),
 * then hadamard64_shared un-rotates into the original basis.  RoPE tail: F16 -> F32.
 * Writes the reused F32 scratch so the trusted F32 attention kernels are byte-identical. */
__global__ static void expand_comp_fp4_to_f32_kernel(
        float *out, const uint8_t *nope, const int8_t *expo, const __half *rope,
        uint32_t n_comp, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;            /* 0..63 */
    if (row >= n_comp) return;
    const uint32_t n_nope = head_dim - n_rot;    /* 448 */
    const uint32_t n_grp  = n_nope / 64u;        /* 7   */
    const uint8_t *nr = nope + (uint64_t)row * (n_nope / 2u);
    const int8_t  *er = expo + (uint64_t)row * 8u;
    const __half  *rr = rope + (uint64_t)row * n_rot;
    float *orow = out + (uint64_t)row * head_dim;
    __shared__ float vals[64];
    for (uint32_t g = 0; g < n_grp; g++) {
        const uint8_t byte = nr[g * 32u + (tid >> 1)];
        const uint8_t nib  = (tid & 1u) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0xfu);
        vals[tid] = dsv4_e2m1fn_decode_nibble_dev(nib) * exp2f((float)er[g]);
        __syncthreads();
        hadamard64_shared(vals, tid);
        orow[g * 64u + tid] = vals[tid];
        __syncthreads();
    }
    for (uint32_t d = tid; d < n_rot; d += 64u)
        orow[n_nope + d] = __half2float(rr[d]);
}

/* Gather-of-selected (decode path): dequantize only the n_sel topk rows of a Hadamard-FP4
 * cache into the reused F32 scratch; out-of-range indices -> zeroed row (cf.
 * gather_comp_fp8_to_f32_kernel).  src is uniform across the block, so the invalid-row
 * early return cannot deadlock hadamard64_shared. */
__global__ static void gather_comp_fp4_to_f32_kernel(
        float *out, const uint8_t *nope, const int8_t *expo, const __half *rope,
        const int32_t *topk, uint32_t n_sel, uint32_t n_comp,
        uint32_t head_dim, uint32_t n_rot) {
    const uint32_t orow = blockIdx.x;
    const uint32_t tid  = threadIdx.x;
    if (orow >= n_sel) return;
    const uint32_t n_nope = head_dim - n_rot;
    const uint32_t n_grp  = n_nope / 64u;
    float *od = out + (uint64_t)orow * head_dim;
    const int32_t src = topk[orow];
    if (src < 0 || (uint32_t)src >= n_comp) {
        for (uint32_t d = tid; d < head_dim; d += 64u) od[d] = 0.0f;
        return;
    }
    const uint32_t row = (uint32_t)src;
    const uint8_t *nr = nope + (uint64_t)row * (n_nope / 2u);
    const int8_t  *er = expo + (uint64_t)row * 8u;
    const __half  *rr = rope + (uint64_t)row * n_rot;
    __shared__ float vals[64];
    for (uint32_t g = 0; g < n_grp; g++) {
        const uint8_t byte = nr[g * 32u + (tid >> 1)];
        const uint8_t nib  = (tid & 1u) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0xfu);
        vals[tid] = dsv4_e2m1fn_decode_nibble_dev(nib) * exp2f((float)er[g]);
        __syncthreads();
        hadamard64_shared(vals, tid);
        od[g * 64u + tid] = vals[tid];
        __syncthreads();
    }
    for (uint32_t d = tid; d < n_rot; d += 64u)
        od[n_nope + d] = __half2float(rr[d]);
}
```

- [ ] **Step 3: Add the two launchers.** In `ds4_cuda.cu`, after `cuda_gather_comp_fp8_to_f32` (ends ~line 9136, just before the next function), insert:

```c
/* Hadamard-FP4 expand-all (prefill): mirrors cuda_expand_comp_fp8_to_f32 but one block/row. */
static const ds4_gpu_tensor *cuda_expand_comp_fp4_to_f32(const ds4_gpu_tensor *nope,
        const ds4_gpu_tensor *expo, const ds4_gpu_tensor *rope,
        uint32_t n_comp, uint32_t head_dim, uint32_t n_rot) {
    if (!nope || !expo || !rope || n_comp == 0u || head_dim == 0u) return NULL;
    const uint64_t count = (uint64_t)n_comp * head_dim;
    const uint64_t bytes = count * sizeof(float);
    if (g_comp_f32_expand_bytes < bytes) {
        if (g_comp_f32_expand_ptr) (void)cudaFree(g_comp_f32_expand_ptr);
        g_comp_f32_expand_ptr = NULL;
        g_comp_f32_expand_bytes = 0;
        if (cudaMalloc(&g_comp_f32_expand_ptr, (size_t)bytes) != cudaSuccess) {
            (void)cudaGetLastError();
            g_comp_f32_expand_ptr = NULL;
            return NULL;
        }
        g_comp_f32_expand_bytes = bytes;
    }
    expand_comp_fp4_to_f32_kernel<<<n_comp, 64>>>(
            (float *)g_comp_f32_expand_ptr, (const uint8_t *)nope->ptr,
            (const int8_t *)expo->ptr, (const __half *)rope->ptr, n_comp, head_dim, n_rot);
    if (!cuda_ok(cudaGetLastError(), "comp fp4->f32 expand")) return NULL;
    g_comp_f32_expand_view.ptr = g_comp_f32_expand_ptr;
    g_comp_f32_expand_view.bytes = g_comp_f32_expand_bytes;
    g_comp_f32_expand_view.owner = 0;
    return &g_comp_f32_expand_view;
}

/* Hadamard-FP4 gather-of-selected (decode): mirrors cuda_gather_comp_fp8_to_f32. */
static const ds4_gpu_tensor *cuda_gather_comp_fp4_to_f32(const ds4_gpu_tensor *nope,
        const ds4_gpu_tensor *expo, const ds4_gpu_tensor *rope, const ds4_gpu_tensor *topk,
        uint32_t n_sel, uint32_t n_comp, uint32_t head_dim, uint32_t n_rot) {
    if (!nope || !expo || !rope || !topk || n_sel == 0u || head_dim == 0u) return NULL;
    const uint64_t count = (uint64_t)n_sel * head_dim;
    const uint64_t bytes = count * sizeof(float);
    if (g_comp_f32_expand_bytes < bytes) {
        if (g_comp_f32_expand_ptr) (void)cudaFree(g_comp_f32_expand_ptr);
        g_comp_f32_expand_ptr = NULL;
        g_comp_f32_expand_bytes = 0;
        if (cudaMalloc(&g_comp_f32_expand_ptr, (size_t)bytes) != cudaSuccess) {
            (void)cudaGetLastError();
            g_comp_f32_expand_ptr = NULL;
            return NULL;
        }
        g_comp_f32_expand_bytes = bytes;
    }
    gather_comp_fp4_to_f32_kernel<<<n_sel, 64>>>(
            (float *)g_comp_f32_expand_ptr, (const uint8_t *)nope->ptr,
            (const int8_t *)expo->ptr, (const __half *)rope->ptr,
            (const int32_t *)topk->ptr, n_sel, n_comp, head_dim, n_rot);
    if (!cuda_ok(cudaGetLastError(), "comp fp4 gather")) return NULL;
    g_comp_f32_expand_view.ptr = g_comp_f32_expand_ptr;
    g_comp_f32_expand_view.bytes = g_comp_f32_expand_bytes;
    g_comp_f32_expand_view.owner = 0;
    return &g_comp_f32_expand_view;
}
```

- [ ] **Step 4: Add `== 3u` branches to the four expand-only entry points.** Each currently ends its `else if (comp_kv_dtype == 1u) { ... comp_kv_dtype = 0u; }` block. Insert a new `else if` BEFORE that block closes the dtype demux (i.e., after the `== 1u` block, add an `== 3u` block). The four sites are the functions containing lines ~9175, ~9481, ~9810, ~9850 (each has the identical `== 2u` / `== 1u` shape). For each, insert after the `== 1u` block:

```c
    } else if (comp_kv_dtype == 3u) {
        if (comp_kv && n_comp) {
            const ds4_gpu_tensor *cef = cuda_expand_comp_fp4_to_f32(
                    comp_kv, comp_scale, comp_rope, n_comp, head_dim, comp_n_rot);
            if (!cef) return 0;
            comp_kv = cef;
        }
        comp_kv_dtype = 0u;
    }
```

(Concretely: change each `} else if (comp_kv_dtype == 1u) { ...; comp_kv_dtype = 0u; }` so the trailing `}` becomes the start of the new `} else if (comp_kv_dtype == 3u) { ... }`.)

- [ ] **Step 5: Generalize the indexed entry point (gather + expand) for FP4.** In the function at ~line 9527 (`if (comp_kv_dtype) { const int is_fp8 = (comp_kv_dtype == 2u); ... }`), replace the body's dtype selection. Change the `is_fp8` line and the two selector expressions:

```c
    if (comp_kv_dtype) {
        const int is_fp8 = (comp_kv_dtype == 2u);
        const int is_fp4 = (comp_kv_dtype == 3u);
        if (comp_kv && n_comp && topk && n_tokens == 1u && ratio != 0u) {
            uint32_t visible = (pos0 + 1u) / ratio;
            if (visible > n_comp) visible = n_comp;
            uint32_t sel = top_k < visible ? top_k : visible;
            if (sel > 512u) sel = 512u;
            if (sel) {
                const ds4_gpu_tensor *gthr = is_fp4
                    ? cuda_gather_comp_fp4_to_f32(comp_kv, comp_scale, comp_rope, topk, sel, n_comp, head_dim, comp_n_rot)
                    : is_fp8
                    ? cuda_gather_comp_fp8_to_f32(comp_kv, comp_scale, comp_rope, topk, sel, n_comp, head_dim, comp_n_rot)
                    : cuda_gather_comp_f16_to_f32(comp_kv, topk, sel, n_comp, head_dim);
                const ds4_gpu_tensor *ident = cuda_identity_topk();
                if (!gthr || !ident) return 0;
                comp_kv = gthr;
                topk = ident;
                n_comp = sel;
            }
        } else if (comp_kv && n_comp) {
            const ds4_gpu_tensor *cef = is_fp4
                ? cuda_expand_comp_fp4_to_f32(comp_kv, comp_scale, comp_rope, n_comp, head_dim, comp_n_rot)
                : is_fp8
                ? cuda_expand_comp_fp8_to_f32(comp_kv, comp_scale, comp_rope, n_comp, head_dim, comp_n_rot)
                : cuda_expand_comp_f16_to_f32(comp_kv, n_comp, head_dim);
            if (!cef) return 0;
            comp_kv = cef;
        }
        comp_kv_dtype = 0u;
    }
```

- [ ] **Step 6: Build.** Run: `make ds4 ds4-bench CUDA_ARCH=`
Expected: clean build (ds4_cuda.cu recompiles via nvcc ~minutes, ds4.c relinks). No warnings-as-errors. The new kernels/launchers are compiled but unreachable (dtype is never 3 yet).

- [ ] **Step 7: Gate on the perplexity oracle (background, then poll).** Run in background: `./ds4 --cuda --perplexity-file doors-of-stone-chapter-1.md -n 256 -c 4096`
Expected: `nll=317.842979423` (bit-identical to the current FP8 build — the FP4 path is dead code) and "context buffers" still 396.21 MiB @4096. If nll differs, a shared kernel/launcher edit leaked into the live path — revert and isolate.

- [ ] **Step 8: Commit.**

```bash
git add ds4_cuda.cu
git commit -m "$(printf 'cuda: Hadamard-FP4 comp-attn KV read-side plumbing (inert)\n\nexpand/gather FP4 kernels (Hadamard-64 un-rotate + nibble decode) + launchers\n+ comp_kv_dtype==3 branches in the 5 attention entry points. Dead until the\nmacro flips; nll unchanged 317.842979423.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 2: Write path + buffers + estimators + save/load + trace + CPU codec (inert, macro OFF)

Adds the FP4 storage everywhere, behind a new `DS4_GPU_ATTN_COMP_CACHE_FP4` macro defined **0** so FP8 stays live. Mirrors Phase 2b commit `c8da47c`. Build must stay clean and nll unchanged.

**Files:**
- Modify: `ds4_cuda.cu` — `dsv4_e2m1fn_index_dev` (encode), `quantize_comp_f32_to_fp4split_kernel`, `ds4_gpu_tensor_quantize_f32_to_fp4split`.
- Modify: `ds4_gpu.h:44` — extern prototype.
- Modify: `ds4.c` — macro (~10336), CPU codec (~2585), `dtype()` (13587), `store_attn_comp_stage` (13619), the four `(F16||FP8)` guard helpers (13655–13700), alloc (11140), both estimators (10836, 21726) + staging guards (10872, 21748), cache-trace (21892), save/load helpers (23915) + 4 call sites (24193, 24415, 24921, 25278).

**Interfaces:**
- Consumes: `cuda_expand_comp_fp4_to_f32` (Task 1, only via runtime dtype — no compile dependency).
- Produces: `int ds4_gpu_tensor_quantize_f32_to_fp4split(ds4_gpu_tensor *nope, ds4_gpu_tensor *expo, ds4_gpu_tensor *rope, uint64_t first_row, const ds4_gpu_tensor *src_stage, uint32_t rows, uint32_t head_dim, uint32_t n_rot)`; macro `DS4_GPU_ATTN_COMP_CACHE_FP4`; host codec `dsv4_hadamard64_inplace_cpu(float*)`, `dsv4_e2m1fn_index_cpu(float)->uint8_t`, `dsv4_e2m1fn_decode_nibble_cpu(uint8_t)->float`.

- [ ] **Step 1: Add the device FP4 encoder + commit kernel.** In `ds4_cuda.cu`, after `dsv4_e4m3fn_encode_dev` / before `quantize_comp_f32_to_fp8split_kernel` (~line 4472), insert the encoder; and after `ds4_gpu_tensor_quantize_f32_to_fp8split` (~line 4535) insert the kernel + entry:

```c
/* Encode an already-scaled value (|x| clamped to 6) to an E2M1FN nibble: bit3 = sign,
 * bits0..2 = the magnitude index dsv4_e2m1fn_dequant_dev would pick (same tie-break). */
__device__ static uint8_t dsv4_e2m1fn_index_dev(float x) {
    uint8_t sign = (x < 0.0f) ? 0x8u : 0x0u;
    float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - dsv4_e2m1fn_value_dev(0));
    for (int i = 1; i < 8; i++) {
        float diff = fabsf(ax - dsv4_e2m1fn_value_dev(i));
        if (diff < best_diff || (diff == best_diff && ((i & 1) == 0) && ((best & 1) != 0))) {
            best = i;
            best_diff = diff;
        }
    }
    return sign | (uint8_t)best;
}
```

```c
/* Commit `rows` staged F32 comp rows to the Hadamard-FP4 cache.  One block/row, 64 threads.
 * Per 64-group: rotate (hadamard64_shared), amax-reduce, frexp-exact k, E2M1 nibble-pack +
 * int8 exponent; RoPE tail -> F16.  Lossy vs the staged E4M3 value (this is the 4-bit step),
 * but decode reproduces the codec's own output bit-for-bit (verified by the value self-check). */
__global__ static void quantize_comp_f32_to_fp4split_kernel(
        uint8_t *nope_out, int8_t *exp_out, __half *rope_out, uint64_t first_row,
        const float *src, uint32_t rows, uint32_t head_dim, uint32_t n_rot) {
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;                 /* 0..63 */
    if (row >= rows) return;
    const uint32_t n_nope = head_dim - n_rot;         /* 448 */
    const uint32_t n_grp  = n_nope / 64u;             /* 7   */
    const float *xr = src + (uint64_t)row * head_dim;
    uint8_t *nr = nope_out + ((uint64_t)first_row + row) * (n_nope / 2u);
    int8_t  *er = exp_out  + ((uint64_t)first_row + row) * 8u;
    __half  *rr = rope_out + ((uint64_t)first_row + row) * n_rot;
    __shared__ float   vals[64];
    __shared__ float   amaxbuf[64];
    __shared__ uint8_t nibs[64];
    for (uint32_t g = 0; g < n_grp; g++) {
        vals[tid] = xr[g * 64u + tid];
        __syncthreads();
        hadamard64_shared(vals, tid);                 /* rotate; ends syncthreaded */
        amaxbuf[tid] = fabsf(vals[tid]);
        __syncthreads();
        for (uint32_t s = 32; s > 0; s >>= 1) {
            if (tid < s) amaxbuf[tid] = fmaxf(amaxbuf[tid], amaxbuf[tid + s]);
            __syncthreads();
        }
        int e2;
        float fr = frexpf(fmaxf(amaxbuf[0], 7.052966104933725e-38f) / 6.0f, &e2);
        int k = (fr == 0.5f) ? (e2 - 1) : e2;
        float scale = exp2f((float)k);
        nibs[tid] = dsv4_e2m1fn_index_dev(fminf(6.0f, fmaxf(-6.0f, vals[tid] / scale)));
        __syncthreads();
        if (tid < 32u)
            nr[g * 32u + tid] = (uint8_t)(nibs[2u * tid] | (nibs[2u * tid + 1u] << 4));
        if (tid == 0) er[g] = (int8_t)k;
        __syncthreads();
    }
    for (uint32_t d = tid; d < n_rot; d += 64u)
        rr[d] = __float2half(xr[n_nope + d]);
}

/* Quantize `rows` staged F32 comp rows into the Hadamard-FP4 cache at row offset first_row. */
extern "C" int ds4_gpu_tensor_quantize_f32_to_fp4split(
        ds4_gpu_tensor *nope, ds4_gpu_tensor *expo, ds4_gpu_tensor *rope, uint64_t first_row,
        const ds4_gpu_tensor *src_stage, uint32_t rows, uint32_t head_dim, uint32_t n_rot) {
    if (!nope || !expo || !rope || !src_stage) return 0;
    if (rows == 0) return 1;
    if (n_rot >= head_dim) return 0;
    const uint32_t n_nope = head_dim - n_rot;
    const uint64_t need_nope = ((uint64_t)first_row + rows) * (n_nope / 2u);
    const uint64_t need_exp  = ((uint64_t)first_row + rows) * 8u;
    const uint64_t need_rope = ((uint64_t)first_row + rows) * n_rot * sizeof(__half);
    if (nope->bytes < need_nope || expo->bytes < need_exp || rope->bytes < need_rope ||
        src_stage->bytes < (uint64_t)rows * head_dim * sizeof(float)) return 0;
    quantize_comp_f32_to_fp4split_kernel<<<rows, 64>>>(
            (uint8_t *)nope->ptr, (int8_t *)expo->ptr, (__half *)rope->ptr, first_row,
            (const float *)src_stage->ptr, rows, head_dim, n_rot);
    return cuda_ok(cudaGetLastError(), "quantize f32->fp4split");
}
```

- [ ] **Step 2: Add the extern prototype.** In `ds4_gpu.h`, after the `ds4_gpu_tensor_quantize_f32_to_fp8split` prototype (~line 47), insert:

```c
/* Quantize staged F32 compressed-attention rows into the Hadamard-FP4 cache: NoPE dims ->
 * Hadamard-64 + E2M1 nibble (`nope`, 2/byte) + per-64-group int8 exponent (`expo`), RoPE
 * tail -> F16 (`rope`).  See DS4_GPU_ATTN_COMP_CACHE_FP4. */
int ds4_gpu_tensor_quantize_f32_to_fp4split(ds4_gpu_tensor *nope, ds4_gpu_tensor *expo,
                                            ds4_gpu_tensor *rope, uint64_t first_row,
                                            const ds4_gpu_tensor *src_stage, uint32_t rows,
                                            uint32_t head_dim, uint32_t n_rot);
```

- [ ] **Step 3: Add the host CPU codec.** In `ds4.c`, after `dsv4_fp4_act_quantize_row_inplace_cpu` (~line 2585), insert:

```c
/* 64-wide normalized Walsh-Hadamard (self-inverse: (H64*0.125)^2 = I), the half-width
 * sibling of dsv4_hadamard128_inplace_cpu.  Identical stride order to the device
 * hadamard64_shared so host and device agree bit-for-bit. */
static DS4_MAYBE_UNUSED void dsv4_hadamard64_inplace_cpu(float *x) {
    for (uint32_t stride = 1; stride < 64; stride <<= 1) {
        for (uint32_t base = 0; base < 64; base += 2u * stride) {
            for (uint32_t i = 0; i < stride; i++) {
                const float a = x[base + i];
                const float b = x[base + stride + i];
                x[base + i] = a + b;
                x[base + stride + i] = a - b;
            }
        }
    }
    const float scale = 0.125f;   /* 1/sqrt(64) */
    for (uint32_t i = 0; i < 64; i++) x[i] *= scale;
}

/* Encode an already-scaled value (|x| clamped to 6) to an E2M1FN nibble: bit3 = sign,
 * bits0..2 = the magnitude index dsv4_e2m1fn_dequant_cpu would pick. */
static DS4_MAYBE_UNUSED uint8_t dsv4_e2m1fn_index_cpu(float x) {
    const uint8_t sign = (x < 0.0f) ? 0x8u : 0x0u;
    const float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - dsv4_e2m1fn_value_cpu(0));
    for (int i = 1; i < 8; i++) {
        const float diff = fabsf(ax - dsv4_e2m1fn_value_cpu(i));
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }
    return sign | (uint8_t)best;
}

static DS4_MAYBE_UNUSED float dsv4_e2m1fn_decode_nibble_cpu(uint8_t nib) {
    const float mag = dsv4_e2m1fn_value_cpu((int)(nib & 0x7u));
    return (nib & 0x8u) ? -mag : mag;
}
```

- [ ] **Step 4: Add the macro (defined 0).** In `ds4.c`, after the FP8 macro block (ends ~line 10336), insert:

```c
/*
 * Hadamard-FP4 storage for the attention-compressed KV cache.  The NoPE 448 dims are
 * Hadamard-64 rotated (self-inverse) then stored as E2M1 FP4 nibbles (2/byte = 224 B/row)
 * plus a per-64-group int8 exponent; the 64-dim RoPE tail stays F16.  Genuinely lossy
 * (~4-bit; the model defines E4M3 not FP4 here), so gated by the perplexity oracle + a
 * value self-check.  Takes PRECEDENCE over _FP8 -> _F16 -> F32 when on.  Default OFF; flipped
 * on for CUDA in the final step so it can be reverted independently.
 */
#define DS4_GPU_ATTN_COMP_CACHE_FP4 0
```

- [ ] **Step 5: FP4-first in `metal_graph_attn_comp_cache_dtype()`.** In `ds4.c` (~13587), prepend the FP4 clause:

```c
static uint32_t metal_graph_attn_comp_cache_dtype(void) {
#if DS4_GPU_ATTN_COMP_CACHE_FP4
    return 3u;
#elif DS4_GPU_ATTN_COMP_CACHE_FP8
    return 2u;
#elif DS4_GPU_ATTN_COMP_CACHE_F16
    return 1u;
#else
    return 0u;
#endif
}
```

- [ ] **Step 6: FP4 branch in `metal_graph_store_attn_comp_stage`.** In `ds4.c` (~13619), prepend the FP4 commit branch before the FP8 one:

```c
#if DS4_GPU_ATTN_COMP_CACHE_FP4
    if (!g->layer_attn_comp_rope[il] || !g->layer_attn_comp_scale[il]) return false;
    return ds4_gpu_tensor_quantize_f32_to_fp4split(g->layer_attn_comp_cache[il],
                                                   g->layer_attn_comp_scale[il],
                                                   g->layer_attn_comp_rope[il],
                                                   first_row,
                                                   g->attn_comp_stage,
                                                   rows,
                                                   DS4_N_HEAD_DIM,
                                                   DS4_N_ROT) != 0;
#elif DS4_GPU_ATTN_COMP_CACHE_FP8
    /* (existing FP8 branch body unchanged) */
```

(Change the existing `#if DS4_GPU_ATTN_COMP_CACHE_FP8` on that block to `#elif DS4_GPU_ATTN_COMP_CACHE_FP8`.)

- [ ] **Step 7: Extend the four `(F16 || FP8)` guard helpers to include FP4.** In `ds4.c`, in `metal_graph_attn_comp_update_target` (13655), `_update_row` (13661), `_commit_attn_comp_stage` (13669), `_attn_comp_row_view` (13677), and `_attn_comp_prefill_target` (13692) + `_prefill_target_free` (13700), replace every `(DS4_GPU_ATTN_COMP_CACHE_F16 || DS4_GPU_ATTN_COMP_CACHE_FP8)` with `(DS4_GPU_ATTN_COMP_CACHE_F16 || DS4_GPU_ATTN_COMP_CACHE_FP8 || DS4_GPU_ATTN_COMP_CACHE_FP4)`. (Staging stays F32, so FP4 uses the staging path exactly like F16/FP8.)

- [ ] **Step 8: FP4 buffer allocation.** In `ds4.c` (~11140), make the FP8 alloc block FP4-first. Replace `if (DS4_GPU_ATTN_COMP_CACHE_FP8) {` with:

```c
            if (DS4_GPU_ATTN_COMP_CACHE_FP4) {
                /* Hadamard-FP4: cache holds 2 E2M1 nibbles/byte (n_nope/2); sidecars hold
                 * the F16 RoPE tail and per-64-group int8 exponents (8/row, 7 used). */
                g->layer_attn_comp_cache[il] = metal_graph_alloc_kv_cache_tensor(
                        managed_kv_cache,
                        (uint64_t)g->layer_comp_cap[il] * ((DS4_N_HEAD_DIM - DS4_N_ROT) / 2u));
                g->layer_attn_comp_rope[il] = metal_graph_alloc_kv_cache_tensor(
                        managed_kv_cache,
                        (uint64_t)g->layer_comp_cap[il] * DS4_N_ROT * sizeof(uint16_t));
                g->layer_attn_comp_scale[il] = metal_graph_alloc_kv_cache_tensor(
                        managed_kv_cache,
                        (uint64_t)g->layer_comp_cap[il] * 8u);
            } else if (DS4_GPU_ATTN_COMP_CACHE_FP8) {
```

(Append `else ` to the existing `if (DS4_GPU_ATTN_COMP_CACHE_FP8)` so it becomes `} else if (...FP8) {`.)

- [ ] **Step 9: FP4 term in the managed-KV estimator.** In `ds4.c` `metal_graph_kv_cache_bytes_for_context` (~10836), make it FP4-first:

```c
        if (DS4_GPU_ATTN_COMP_CACHE_FP4) {
            bytes += comp_cap * ((DS4_N_HEAD_DIM - DS4_N_ROT) / 2u + 8u +
                                 DS4_N_ROT * sizeof(uint16_t));
        } else if (DS4_GPU_ATTN_COMP_CACHE_FP8) {
            bytes += comp_cap * ((DS4_N_HEAD_DIM - DS4_N_ROT) + 8u +
                                 DS4_N_ROT * sizeof(uint16_t));
        } else {
            bytes += comp_cap * DS4_N_HEAD_DIM *
                     (DS4_GPU_ATTN_COMP_CACHE_F16 ? sizeof(uint16_t) : sizeof(float));
        }
```

- [ ] **Step 10: FP4 staging guard in the KV-policy estimator.** In `ds4.c` `metal_graph_context_bytes_for_kv_policy` (~10868–10873), staging stays F32, so extend the two guards to include FP4: change `if (DS4_GPU_ATTN_COMP_CACHE_F16 || DS4_GPU_INDEX_COMP_CACHE_F16 || DS4_GPU_ATTN_COMP_CACHE_FP8)` → add `|| DS4_GPU_ATTN_COMP_CACHE_FP4`, and `if (DS4_GPU_ATTN_COMP_CACHE_F16 || DS4_GPU_ATTN_COMP_CACHE_FP8)` (the attn-stage term, ~10872) → add `|| DS4_GPU_ATTN_COMP_CACHE_FP4`.

- [ ] **Step 11: FP4 term in the main memory estimate.** In `ds4.c` `ds4_context_memory_estimate_with_prefill` (~21726), make it FP4-first:

```c
            if (DS4_GPU_ATTN_COMP_CACHE_FP4) {
                /* E2M1 nibbles (n_nope/2) + int8 exponents (8/row) + F16 RoPE tail. */
                m.compressed_bytes += (uint64_t)layer_comp_cap *
                                      ((DS4_N_HEAD_DIM - DS4_N_ROT) / 2u + 8u +
                                       DS4_N_ROT * sizeof(uint16_t));
            } else if (DS4_GPU_ATTN_COMP_CACHE_FP8) {
                m.compressed_bytes += (uint64_t)layer_comp_cap *
                                      ((DS4_N_HEAD_DIM - DS4_N_ROT) + 8u +
                                       DS4_N_ROT * sizeof(uint16_t));
            } else {
                m.compressed_bytes += (uint64_t)layer_comp_cap *
                                      DS4_N_HEAD_DIM *
                                      (DS4_GPU_ATTN_COMP_CACHE_F16 ? sizeof(uint16_t) : sizeof(float));
            }
```

Then extend the `scratch_bytes` staging guard (~21748) `(DS4_GPU_ATTN_COMP_CACHE_F16 || DS4_GPU_ATTN_COMP_CACHE_FP8)` → add `|| DS4_GPU_ATTN_COMP_CACHE_FP4`.

- [ ] **Step 12: FP4 host-decode branch in the cache-trace.** In `ds4.c` (~21892), prepend an FP4 branch before the FP8 one (read nibbles + per-group exponent + F16 rope, Hadamard-64 un-rotate per group):

```c
                if (DS4_GPU_ATTN_COMP_CACHE_FP4) {
                    const uint32_t n_nope = DS4_N_HEAD_DIM - DS4_N_ROT;
                    uint8_t  *cb = xmalloc((size_t)n_comp * (n_nope / 2u));
                    int8_t   *xb = xmalloc((size_t)n_comp * 8u);
                    uint16_t *hb = xmalloc((size_t)n_comp * DS4_N_ROT * sizeof(uint16_t));
                    if (ds4_gpu_tensor_read(g.layer_attn_comp_cache[il], 0, cb,
                                            (size_t)n_comp * (n_nope / 2u)) != 0 &&
                        ds4_gpu_tensor_read(g.layer_attn_comp_scale[il], 0, xb,
                                            (size_t)n_comp * 8u) != 0 &&
                        ds4_gpu_tensor_read(g.layer_attn_comp_rope[il], 0, hb,
                                            (size_t)n_comp * DS4_N_ROT * sizeof(uint16_t)) != 0) {
                        for (uint32_t r = 0; r < n_comp; r++) {
                            for (uint32_t off = 0; off < n_nope; off += 64u) {
                                float grp[64];
                                const int k = (int)xb[(size_t)r * 8u + (off >> 6)];
                                for (uint32_t l = 0; l < 64u; l++) {
                                    const uint8_t byte = cb[(size_t)r * (n_nope / 2u) + (off + l) / 2u];
                                    const uint8_t nib = ((off + l) & 1u) ? (byte >> 4) : (byte & 0xfu);
                                    grp[l] = dsv4_e2m1fn_decode_nibble_cpu(nib) * ldexpf(1.0f, k);
                                }
                                dsv4_hadamard64_inplace_cpu(grp);
                                for (uint32_t l = 0; l < 64u; l++)
                                    gpu_comp[(size_t)r * DS4_N_HEAD_DIM + off + l] = grp[l];
                            }
                            for (uint32_t d = n_nope; d < DS4_N_HEAD_DIM; d++)
                                gpu_comp[(size_t)r * DS4_N_HEAD_DIM + d] =
                                    f16_to_f32(hb[(size_t)r * DS4_N_ROT + (d - n_nope)]);
                        }
                        comp_read = true;
                    }
                    free(cb); free(xb); free(hb);
                } else if (DS4_GPU_ATTN_COMP_CACHE_FP8) {
```

(Append `else ` to the existing `if (DS4_GPU_ATTN_COMP_CACHE_FP8)` at that site.)

- [ ] **Step 13: Add the FP4 save/load host codec helpers.** In `ds4.c`, after `payload_read_tensor_span_f32_as_fp8split` (ends ~line 24035), insert two siblings. They keep the stable F32 on-disk format (decode FP4→F32 on save, encode F32→FP4 via Hadamard-64 + frexp-exact-k on load):

```c
static DS4_MAYBE_UNUSED int payload_write_tensor_span_fp4split_as_f32(
        FILE *fp, const ds4_gpu_tensor *nope, const ds4_gpu_tensor *expo,
        const ds4_gpu_tensor *rope, uint64_t n_rows, uint8_t *buf, size_t cap,
        char *err, size_t errlen) {
    if (!nope || !expo || !rope) {
        payload_set_err(err, errlen, "missing FP4-split session tensor");
        return 1;
    }
    const uint32_t n_nope = DS4_N_HEAD_DIM - DS4_N_ROT;
    const size_t per_row = (size_t)(n_nope / 2u) + 8u + (size_t)DS4_N_ROT * sizeof(uint16_t) +
                           (size_t)DS4_N_HEAD_DIM * sizeof(float);
    const size_t rows_chunk = cap / per_row;
    if (rows_chunk == 0) {
        payload_set_err(err, errlen, "session conversion buffer too small for FP4-split");
        return 1;
    }
    uint8_t  *nb = buf;
    int8_t   *eb = (int8_t *)(buf + rows_chunk * (n_nope / 2u));
    uint16_t *rb = (uint16_t *)(void *)(buf + rows_chunk * ((size_t)(n_nope / 2u) + 8u));
    float    *fb = (float *)(void *)(buf + rows_chunk *
                   ((size_t)(n_nope / 2u) + 8u + (size_t)DS4_N_ROT * sizeof(uint16_t)));
    uint64_t done = 0;
    while (done < n_rows) {
        const size_t n = n_rows - done > (uint64_t)rows_chunk ? rows_chunk
                                                              : (size_t)(n_rows - done);
        if (ds4_gpu_tensor_read(nope, done * (n_nope / 2u), nb, n * (n_nope / 2u)) == 0 ||
            ds4_gpu_tensor_read(expo, done * 8u, eb, n * 8u) == 0 ||
            ds4_gpu_tensor_read(rope, done * DS4_N_ROT * sizeof(uint16_t), rb,
                                n * DS4_N_ROT * sizeof(uint16_t)) == 0) {
            payload_set_err(err, errlen, "failed to read FP4-split session tensor");
            return 1;
        }
        for (size_t r = 0; r < n; r++) {
            float          *frow = fb + r * DS4_N_HEAD_DIM;
            const uint8_t  *nrow = nb + r * (n_nope / 2u);
            const int8_t   *erow = eb + r * 8u;
            const uint16_t *rrow = rb + r * DS4_N_ROT;
            for (uint32_t off = 0; off < n_nope; off += 64u) {
                float grp[64];
                const int k = (int)erow[off >> 6];
                for (uint32_t l = 0; l < 64u; l++) {
                    const uint8_t byte = nrow[(off + l) / 2u];
                    const uint8_t nib = ((off + l) & 1u) ? (byte >> 4) : (byte & 0xfu);
                    grp[l] = dsv4_e2m1fn_decode_nibble_cpu(nib) * ldexpf(1.0f, k);
                }
                dsv4_hadamard64_inplace_cpu(grp);
                for (uint32_t l = 0; l < 64u; l++) frow[off + l] = grp[l];
            }
            for (uint32_t d = 0; d < DS4_N_ROT; d++)
                frow[n_nope + d] = f16_to_f32(rrow[d]);
        }
        if (payload_write_bytes(fp, fb, (uint64_t)n * DS4_N_HEAD_DIM * sizeof(float),
                                err, errlen) != 0) return 1;
        done += n;
    }
    return 0;
}

static DS4_MAYBE_UNUSED int payload_read_tensor_span_f32_as_fp4split(
        FILE *fp, ds4_gpu_tensor *nope, ds4_gpu_tensor *expo, ds4_gpu_tensor *rope,
        uint64_t n_rows, uint8_t *buf, size_t cap, uint64_t *remaining,
        char *err, size_t errlen) {
    if (!nope || !expo || !rope) {
        payload_set_err(err, errlen, "missing FP4-split session tensor");
        return 1;
    }
    const uint32_t n_nope = DS4_N_HEAD_DIM - DS4_N_ROT;
    const size_t per_row = (size_t)(n_nope / 2u) + 8u + (size_t)DS4_N_ROT * sizeof(uint16_t) +
                           (size_t)DS4_N_HEAD_DIM * sizeof(float);
    const size_t rows_chunk = cap / per_row;
    if (rows_chunk == 0) {
        payload_set_err(err, errlen, "session conversion buffer too small for FP4-split");
        return 1;
    }
    uint8_t  *nb = buf;
    int8_t   *eb = (int8_t *)(buf + rows_chunk * (n_nope / 2u));
    uint16_t *rb = (uint16_t *)(void *)(buf + rows_chunk * ((size_t)(n_nope / 2u) + 8u));
    float    *fb = (float *)(void *)(buf + rows_chunk *
                   ((size_t)(n_nope / 2u) + 8u + (size_t)DS4_N_ROT * sizeof(uint16_t)));
    uint64_t done = 0;
    while (done < n_rows) {
        const size_t n = n_rows - done > (uint64_t)rows_chunk ? rows_chunk
                                                              : (size_t)(n_rows - done);
        if (payload_read_bytes(fp, fb, (uint64_t)n * DS4_N_HEAD_DIM * sizeof(float),
                               remaining, err, errlen) != 0) return 1;
        for (size_t r = 0; r < n; r++) {
            const float *frow = fb + r * DS4_N_HEAD_DIM;
            uint8_t  *nrow = nb + r * (n_nope / 2u);
            int8_t   *erow = eb + r * 8u;
            uint16_t *rrow = rb + r * DS4_N_ROT;
            for (uint32_t off = 0; off < n_nope; off += 64u) {
                float grp[64];
                for (uint32_t l = 0; l < 64u; l++) grp[l] = frow[off + l];
                dsv4_hadamard64_inplace_cpu(grp);
                float amax = 7.052966104933725e-38f;
                for (uint32_t l = 0; l < 64u; l++) {
                    const float av = fabsf(grp[l]);
                    if (av > amax) amax = av;
                }
                int e2;
                const float fr = frexpf(amax / 6.0f, &e2);
                const int k = (fr == 0.5f) ? (e2 - 1) : e2;
                const float scale = ldexpf(1.0f, k);
                erow[off >> 6] = (int8_t)k;
                for (uint32_t l = 0; l < 64u; l += 2u) {
                    float v0 = grp[l] / scale, v1 = grp[l + 1] / scale;
                    if (v0 > 6.0f) v0 = 6.0f; if (v0 < -6.0f) v0 = -6.0f;
                    if (v1 > 6.0f) v1 = 6.0f; if (v1 < -6.0f) v1 = -6.0f;
                    nrow[(off + l) / 2u] =
                        (uint8_t)(dsv4_e2m1fn_index_cpu(v0) | (dsv4_e2m1fn_index_cpu(v1) << 4));
                }
            }
            for (uint32_t d = 0; d < DS4_N_ROT; d++)
                rrow[d] = f32_to_f16(frow[n_nope + d]);
        }
        if (ds4_gpu_tensor_write(nope, done * (n_nope / 2u), nb, n * (n_nope / 2u)) == 0 ||
            ds4_gpu_tensor_write(expo, done * 8u, eb, n * 8u) == 0 ||
            ds4_gpu_tensor_write(rope, done * DS4_N_ROT * sizeof(uint16_t), rb,
                                 n * DS4_N_ROT * sizeof(uint16_t)) == 0) {
            payload_set_err(err, errlen, "failed to restore FP4-split session tensor");
            return 1;
        }
        done += n;
    }
    return 0;
}
```

- [ ] **Step 14: Wire the four save/load call sites.** In `ds4.c` at the four sites (~24193 save-layer, ~24415 load-layer, ~24921 save, ~25278 load) that read `if (DS4_GPU_ATTN_COMP_CACHE_FP8) { rc = payload_..._fp8split...(...); } else if (DS4_GPU_ATTN_COMP_CACHE_F16) {`, prepend an FP4 branch. For the two **save** sites:

```c
        if (DS4_GPU_ATTN_COMP_CACHE_FP4) {
            rc = payload_write_tensor_span_fp4split_as_f32(fp,
                                                           g->layer_attn_comp_cache[il],
                                                           g->layer_attn_comp_scale[il],
                                                           g->layer_attn_comp_rope[il],
                                                           <n_rows-arg>, <buf-arg>, <cap-arg>,
                                                           <err-arg>, <errlen-arg>);
        } else if (DS4_GPU_ATTN_COMP_CACHE_FP8) {
```

For the two **load** sites use `payload_read_tensor_span_f32_as_fp4split(fp, ...cache, ...scale, ...rope, <n_rows>, <buf>, <cap>, <remaining>, <err>, <errlen>)`. Copy the exact argument expressions from the adjacent FP8 call at each site (they are identical except the function name); append `else ` to the existing `if (DS4_GPU_ATTN_COMP_CACHE_FP8)`.

- [ ] **Step 15: Build.** Run: `make ds4 ds4-bench CUDA_ARCH=`
Expected: clean build. All FP4 `#if` branches are preprocessed out (macro 0); the unconditionally-compiled additions (CPU codec, save/load helpers, quantize kernel/entry) carry `DS4_MAYBE_UNUSED` / are referenced only from the dead branches, so no unused-function errors.

- [ ] **Step 16: Gate on the perplexity oracle.** Run in background: `./ds4 --cuda --perplexity-file doors-of-stone-chapter-1.md -n 256 -c 4096`
Expected: `nll=317.842979423`, context buffers 396.21 MiB @4096 — unchanged (FP8 still live).

- [ ] **Step 17: Commit.**

```bash
git add ds4_cuda.cu ds4_gpu.h ds4.c
git commit -m "$(printf 'cuda: Hadamard-FP4 comp-attn KV write/buffers/save-load/trace (inert)\n\nFP4 commit kernel (Hadamard-64 rotate + frexp-exact k + E2M1 nibble pack),\nquantize entry + proto, host codec, DS4_GPU_ATTN_COMP_CACHE_FP4 macro (0),\ndtype()/store/alloc/estimators/save-load/trace FP4-first branches. Macro off\n=> FP8 still live; nll unchanged 317.842979423.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 3: Value-exactness self-check harness (env-gated, compiled but dormant until flip)

The correctness proof per the spec: separate intended FP4 loss from bugs by verifying the GPU cache decodes to exactly what the host codec produces from the same staged F32. Mirrors Phase 2b's selfcheck+readcheck tooling. It lives inside the FP4 store branch (so it only runs once the macro is on, in Task 4) and only when `DS4_FP4_SELFCHECK` is set.

**Files:**
- Modify: `ds4.c` — new `metal_graph_fp4_selfcheck(...)` (near `metal_graph_store_attn_comp_stage`, ~13606) + a guarded call inside the FP4 store branch.

**Interfaces:**
- Consumes: `dsv4_hadamard64_inplace_cpu`, `dsv4_e2m1fn_index_cpu`, `dsv4_e2m1fn_decode_nibble_cpu` (Task 2); `ds4_gpu_tensor_read`; `cuda_expand` is exercised indirectly via the trace, so the readcheck here recomputes host-decode from the read-back cache bytes.
- Produces: `static void metal_graph_fp4_selfcheck(ds4_gpu_graph *g, uint32_t il, uint32_t first_row, uint32_t rows)`.

- [ ] **Step 1: Add the self-check routine.** In `ds4.c`, immediately before `metal_graph_store_attn_comp_stage` (~13606), insert. It reads back the just-written cache bytes + the staged F32, recomputes the host codec's expected nibbles/exponent/rope, and asserts (a) stored bytes == host-encoded bytes (selfcheck) and (b) host-decode(stored) == host-decode(host-encode) bit-for-bit (readcheck), logging the worst mismatch:

```c
static DS4_MAYBE_UNUSED void metal_graph_fp4_selfcheck(
        ds4_gpu_graph *g, uint32_t il, uint32_t first_row, uint32_t rows) {
    if (!getenv("DS4_FP4_SELFCHECK") || rows == 0) return;
    const uint32_t n_nope = DS4_N_HEAD_DIM - DS4_N_ROT;
    const size_t   half   = n_nope / 2u;
    uint8_t *gb = xmalloc((size_t)rows * half);
    int8_t  *xe = xmalloc((size_t)rows * 8u);
    float   *st = xmalloc((size_t)rows * DS4_N_HEAD_DIM * sizeof(float));
    if (ds4_gpu_tensor_read(g->layer_attn_comp_cache[il], (uint64_t)first_row * half, gb,
                            (size_t)rows * half) == 0 ||
        ds4_gpu_tensor_read(g->layer_attn_comp_scale[il], (uint64_t)first_row * 8u, xe,
                            (size_t)rows * 8u) == 0 ||
        ds4_gpu_tensor_read(g->attn_comp_stage, 0, st,
                            (size_t)rows * DS4_N_HEAD_DIM * sizeof(float)) == 0) {
        free(gb); free(xe); free(st); return;
    }
    uint64_t byte_mismatch = 0, val_mismatch = 0;
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t off = 0; off < n_nope; off += 64u) {
            float grp[64];
            for (uint32_t l = 0; l < 64u; l++) grp[l] = st[(size_t)r * DS4_N_HEAD_DIM + off + l];
            dsv4_hadamard64_inplace_cpu(grp);
            float amax = 7.052966104933725e-38f;
            for (uint32_t l = 0; l < 64u; l++) { float a = fabsf(grp[l]); if (a > amax) amax = a; }
            int e2; const float fr = frexpf(amax / 6.0f, &e2);
            const int k = (fr == 0.5f) ? (e2 - 1) : e2;
            const float scale = ldexpf(1.0f, k);
            if ((int)xe[(size_t)r * 8u + (off >> 6)] != k) byte_mismatch++;
            for (uint32_t l = 0; l < 64u; l++) {
                float v = grp[l] / scale; if (v > 6.0f) v = 6.0f; if (v < -6.0f) v = -6.0f;
                const uint8_t want = dsv4_e2m1fn_index_cpu(v);
                const uint8_t byte = gb[(size_t)r * half + (off + l) / 2u];
                const uint8_t got  = ((off + l) & 1u) ? (byte >> 4) : (byte & 0xfu);
                if (got != want) byte_mismatch++;
                const float dgot  = dsv4_e2m1fn_decode_nibble_cpu(got)  * scale;
                const float dwant = dsv4_e2m1fn_decode_nibble_cpu(want) * scale;
                if (dgot != dwant) val_mismatch++;
            }
        }
    }
    fprintf(stderr, "ds4: FP4 selfcheck layer %u rows %u byte_mismatch=%llu val_mismatch=%llu\n",
            il, rows, (unsigned long long)byte_mismatch, (unsigned long long)val_mismatch);
    free(gb); free(xe); free(st);
}
```

- [ ] **Step 2: Call it from the FP4 store branch.** In `ds4.c` `metal_graph_store_attn_comp_stage`, inside the `#if DS4_GPU_ATTN_COMP_CACHE_FP4` branch added in Task 2, capture the quantize result and run the check before returning:

```c
#if DS4_GPU_ATTN_COMP_CACHE_FP4
    if (!g->layer_attn_comp_rope[il] || !g->layer_attn_comp_scale[il]) return false;
    {
        const bool ok = ds4_gpu_tensor_quantize_f32_to_fp4split(g->layer_attn_comp_cache[il],
                                                                g->layer_attn_comp_scale[il],
                                                                g->layer_attn_comp_rope[il],
                                                                first_row, g->attn_comp_stage,
                                                                rows, DS4_N_HEAD_DIM, DS4_N_ROT) != 0;
        if (ok) metal_graph_fp4_selfcheck(g, il, first_row, rows);
        return ok;
    }
#elif DS4_GPU_ATTN_COMP_CACHE_FP8
```

- [ ] **Step 3: Build.** Run: `make ds4 ds4-bench CUDA_ARCH=`
Expected: clean build (the `#if FP4` branch is still preprocessed out at macro 0; `metal_graph_fp4_selfcheck` is `DS4_MAYBE_UNUSED`).

- [ ] **Step 4: Gate.** Run in background: `./ds4 --cuda --perplexity-file doors-of-stone-chapter-1.md -n 256 -c 4096`
Expected: `nll=317.842979423` unchanged (still FP8; the check is dormant).

- [ ] **Step 5: Commit.**

```bash
git add ds4.c
git commit -m "$(printf 'ds4: FP4 comp-attn KV value-exactness self-check (env-gated, dormant)\n\nDS4_FP4_SELFCHECK recomputes the host Hadamard-64+E2M1 codec from the staged\nF32 and asserts stored bytes + decoded values match bit-for-bit, to separate\nintended 4-bit loss from bugs once the macro flips. Inert at macro=0.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Task 4: Flip the macro ON for CUDA + validate (perplexity, self-check, memory, long-ctx)

The only behavior change. Mirrors Phase 2b commit `bf7c814`/`53b8e7e`. Acceptance per the Global Constraints gate.

**Files:**
- Modify: `ds4.c` — the `DS4_GPU_ATTN_COMP_CACHE_FP4` macro definition.
- Modify: memory `ds4-optimization-findings.md` + the spec Status section (post-validation).

- [ ] **Step 1: Flip the macro.** In `ds4.c`, replace `#define DS4_GPU_ATTN_COMP_CACHE_FP4 0` (added in Task 2 Step 4) with the CUDA-only conditional:

```c
#if !defined(__APPLE__) && !defined(DS4_ROCM_BUILD) && !defined(DS4_NO_GPU)
#define DS4_GPU_ATTN_COMP_CACHE_FP4 1
#else
#define DS4_GPU_ATTN_COMP_CACHE_FP4 0
#endif
```

- [ ] **Step 2: Build.** Run: `make ds4 ds4-bench CUDA_ARCH=`
Expected: clean build. FP4 is now the live attn-comp dtype (precedence over FP8).

- [ ] **Step 3: Run the value self-check first (correctness before nll).** Run in background: `DS4_FP4_SELFCHECK=1 ./ds4 --cuda --perplexity-file doors-of-stone-chapter-1.md -n 256 -c 4096`
Expected: every `FP4 selfcheck ... byte_mismatch=0 val_mismatch=0` line. **Any nonzero mismatch is a codec/Hadamard/exponent bug — STOP and fix before trusting nll** (check: host vs device Hadamard stride order, frexp-exact `k`, nibble even/odd packing, the `0.125` normalization). Note the final `nll` too.

- [ ] **Step 4: Gate on the perplexity oracle + memory.** Run in background: `./ds4 --cuda --perplexity-file doors-of-stone-chapter-1.md -n 256 -c 4096`
Expected: **`nll ≤ 319.4`** (realistically a small drift above 317.843 — this is the intended 4-bit loss), no NaN/crash, and the "context buffers" line drops to **~391.6 MiB @4096** (comp cache 12.0 → ~7.4 MiB). If nll > 319.4 with a clean self-check, the codec is too lossy at per-64 scale → fall back to per-32 FP4 scale (14 exponents/row; widen the scale buffer alloc + estimators + save/load). If the buffers line is wrong, an estimator term is off.

- [ ] **Step 5: Long-context stability check.** Build the long input and run:

```bash
for i in $(seq 8); do cat doors-of-stone-chapter-1.md; done > /tmp/claude-1000/-home-ylow-deepseekflash/5aaec50a-9815-49b1-88e4-1b0c81ab21fd/scratchpad/doors-x8.md
./ds4 --cuda --perplexity-file /tmp/claude-1000/-home-ylow-deepseekflash/5aaec50a-9815-49b1-88e4-1b0c81ab21fd/scratchpad/doors-x8.md -n 256 -c 16384
```
Expected: completes with no NaN/blow-up, avg_nll stable (not diverging vs the 4096 run). Confirms the recurrence isn't chaotically amplifying a latent error over long context.

- [ ] **Step 6: Finish the build.** Run: `make cuda-spark`
Expected: clean; `ds4-server` rebuilt with FP4 on.

- [ ] **Step 7: Commit the code.**

```bash
git add ds4.c
git commit -m "$(printf 'cuda: enable Hadamard-FP4 comp-attn KV (DS4_GPU_ATTN_COMP_CACHE_FP4 on)\n\n360 B/row (5.69x vs F32, 1.62x vs FP8); context buffers 396.21 -> ~391.6 MiB\n@4096. Value self-check byte/val_mismatch=0; perplexity nll=<measured> (<=319.4);\n16384 long-ctx stable.\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

(Replace `<measured>` with the actual nll from Step 4.)

- [ ] **Step 8: Update the spec Status + memory.** Edit `docs/superpowers/specs/2026-06-26-turboquant-fp4-kv-design.md` "Status — PLANNED" → "LANDED" with the measured nll, buffers, self-check result, and commit hashes. Update `/home/ylow/.claude/projects/-home-ylow-deepseekflash/memory/ds4-optimization-findings.md` with a Phase 2c paragraph (codec, 360 B/row, measured nll/drift, the per-64-scale choice, the self-check methodology). Commit the spec change:

```bash
git add docs/superpowers/specs/2026-06-26-turboquant-fp4-kv-design.md
git commit -m "$(printf 'docs: record Hadamard-FP4 comp-attn KV result (Phase 2c)\n\nCo-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>')"
```

---

## Self-Review notes

- **Spec coverage:** codec (T2 S1,S3) ✓; Hadamard-64 self-inverse rotate/un-rotate (T1 S1) ✓; per-64 scale reusing the 8/row buffer (T2 S8) ✓; staging stays E4M3-QAT'd F32 — untouched, only the commit changes (T2 S6) ✓; RoPE F16 / indexer untouched (no indexer edits) ✓; expand-on-read with `==3` precedence (T1 S4–S5) ✓; gather-of-selected bounded decode (T1 S2 gather kernel) ✓; macro precedence FP4→FP8→F16→F32 (T2 S4–S11) ✓; 360 B/row memory in both estimators (T2 S9,S11) ✓; save/load F32 disk format (T2 S13–S14) ✓; cache-trace FP4 (T2 S12) ✓; value self-check (T3) ✓; tolerance ≤319.4 + long-ctx (T4 S3–S5) ✓; frexp-exact exponent everywhere (T2 S1 kernel, S13 load, T3 check) ✓.
- **Type consistency:** `dsv4_e2m1fn_index_dev/cpu` (encode→nibble), `dsv4_e2m1fn_decode_nibble_dev/cpu` (decode), `hadamard64_shared`(device)/`dsv4_hadamard64_inplace_cpu`(host), `ds4_gpu_tensor_quantize_f32_to_fp4split`, `payload_{write,read}_tensor_span_{fp4split_as_f32,f32_as_fp4split}`, `metal_graph_fp4_selfcheck` — names used consistently across tasks; the quantize entry signature matches the fp8split sibling exactly.
- **Line numbers are anchors** (pre-edit); each step names the enclosing function/symbol so they remain locatable as edits shift offsets.
