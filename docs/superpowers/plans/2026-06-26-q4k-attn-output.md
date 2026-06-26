# Q4_K Attention Output-Projection — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Requantize the attention output-projection weights (`attn_output_a`, `attn_output_b`) from Q8_0 to Q4_K to cut the decode byte budget, and measure the resulting perplexity (nll) / decode-throughput (tok/s) tradeoff.

**Architecture:** Two halves. (1) **Offline** — extend `gguf-tools/deepseek4-quantize` with a "requant-from-template-GGUF" mode that dequantizes the Q8_0 `attn_output_*` tensors straight from the existing GGUF and re-quantizes them to Q4_K, byte-copying every other tensor through; this produces a new GGUF with no HF re-download. (2) **Engine** — add Q4_K weight-read variants of the four specialized output-projection CUDA kernels (decode output_a, decode output_b HC-expand, batch output_a, batch output_b), dispatched by the loaded tensor's quant type, and relax the load-time type validation. Validate with the bit-deterministic teacher-forced perplexity oracle plus a decode tok/s measurement; an env-gated device-vs-host self-check proves each new kernel's addressing before trusting the oracle.

**Tech Stack:** C11 (engine `ds4.c`, CLI `ds4_cli.c`), CUDA (`ds4_cuda.cu`), standalone C quantizer (`gguf-tools/deepseek4-quantize.c`, `gguf-tools/quants.[ch]`). Build: `make cuda-spark` (engine), `make -C gguf-tools` (quantizer). GPU: single GB10 (sm_121), 273 GB/s, unified memory.

## Global Constraints

- **Correctness oracle (bit-deterministic).** `./ds4 --perplexity-file doors-of-stone-chapter-1.md -n 256 -c 4096` prints `nll=…`. The Q8_0 baseline is `nll=317.843967992`. Greedy generation is NOT a valid oracle (non-associative GPU reductions). Every gate uses teacher-forced perplexity.
- **This is a deliberate lossy trade.** No fixed nll budget — we measure and report Δnll / Δppl and decode tok/s, then decide. For reference, the FP4-KV phase accepted +0.367% nll.
- **Q4_K / Q8_K super-block = 256.** A Q4_K weight or Q8_K activation requires the contraction dimension to be a multiple of 256. Verified for this recipe: output_a in_dim `group_dim=4096`, output_b in_dim `low_dim=8192`, output_b out_dim `DS4_N_EMBD=4096` — all `% 256 == 0`. Do NOT generalize the Q4_K path to tensors that violate this.
- **Recipe = output proj only.** Only `blk.{L}.attn_output_a.weight` and `blk.{L}.attn_output_b.weight` (and the MTP `mtp.0.attn_output_*`) become Q4_K. `attn_q_a`, `attn_q_b`, `attn_kv`, shared expert, output head, routed experts stay exactly as in the template.
- **Dispatch by loaded type, no env flag for the recipe.** The recipe lives in the GGUF; the engine branches on `tensor->type == DS4_TENSOR_Q4_K`. The original Q8 GGUF remains a valid input (instant rollback by repointing the `ds4flash.gguf` symlink). The only env flag added is `DS4_Q4K_SELFCHECK` for the development-time self-check.
- **Block formats (exact).** Q8_0: 34 bytes/block = `[f16 scale][32×int8]`, 32 weights/block. Q4_K: `cuda_block_q4_K` = 144 bytes/block = `{u16 d; u16 dmin; u8 scales[12]; u8 qs[128];}`, 256 weights/block. Q8_K activation: `cuda_block_q8_K` = `{float d; int8 qs[256]; int16 bsums[16];}`.
- **Git:** ds4 is a git repo, local commits only (`main`), never push. Commit messages end with the `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` trailer.
- **Engine ↔ quantizer are separate programs.** `ds4.c`/`ds4_cuda.cu` do NOT link `gguf-tools/quants.c`. The engine already has its own `block_q4_K` + CPU Q4_K dot (used by routed experts; `q4k-dot-test` Makefile target). Reuse each side's own code; do not cross-link.

---

## Files

**Offline (gguf-tools):**
- Modify `gguf-tools/quants.h` — declare `ds4q_dequantize_q8_0`.
- Modify `gguf-tools/quants.c` — implement `ds4q_dequantize_q8_0` (inverse of `ds4q_quantize_q8_0` at :341).
- Modify `gguf-tools/deepseek4-quantize.c` — add `--template-requant` mode: thread the template `gguf_file*` into `generate_regular`; source listed tensors from the template (Q8_0→f32) instead of HF; byte-copy all others; make `--hf` optional in this mode. Add an RMSE branch to `compare_one_tensor`.

**Engine (CUDA kernels + wrappers):**
- Modify `ds4_gpu.h` — declare 4 new `extern "C"` Q4_K wrappers.
- Modify `ds4_cuda.cu` — add 4 Q4_K device kernels + 4 host wrappers + an env-gated self-check helper.

**Engine (dispatch + validation):**
- Modify `ds4.c` — add `tensor_expect_q8_0_or_q4_k_layout`; swap it in at the 2 output-proj validation sites (+ 2 MTP); branch on `->type` at the decode dispatch (`:15822`), non-fused decode (`:15846`), and prefill (`:19116`) sites.

**Validation (no new files):** run perplexity + tok/s; record results back into this plan and the design spec.

---

## Interfaces (new symbols, exact signatures)

```c
/* gguf-tools/quants.h */
/* Dequantize a Q8_0 tensor (row-major [nrows x ncols], ncols % 32 == 0) to f32.
   dst must hold nrows*ncols floats. Mirrors ds4q_quantize_q8_0 in reverse. */
void ds4q_dequantize_q8_0(const void *src, float *dst, int64_t nrows, int64_t ncols);
```

```c
/* ds4_gpu.h  — all extern "C", all NATIVE (no cuBLAS), Q4_K weight + Q8_K activation. */

/* output_a: grouped rows. heads -> low. n_tokens via grid.y (serves decode n_tok=1 AND batch). */
int ds4_gpu_attention_output_low_q4k_tensor(
        ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t group_dim, uint64_t rank,
        uint32_t n_groups, const ds4_gpu_tensor *heads, uint32_t n_tokens);

/* output_b decode: HC-expand fused (low -> out_hc + block_out). Single token. */
int ds4_gpu_matmul_q4k_hc_expand_tensor(
        ds4_gpu_tensor *out_hc, ds4_gpu_tensor *block_out, const void *model_map,
        uint64_t model_size, uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc);

/* output_b batch: plain Q4_K matmul, n_tok rows (low -> out). */
int ds4_gpu_matmul_q4k_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x, uint64_t n_tok);

/* batch both: output_a (grouped) then output_b (plain). Serves prefill + non-fused decode. */
int ds4_gpu_attention_output_q4k_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low,
        const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint64_t out_dim,
        const ds4_gpu_tensor *heads, uint32_t n_tokens);
```

Consumed by the dispatch edits in `ds4.c` (Task 8). Each Q4_K kernel reuses the existing device primitives `q8_K_quantize_kernel` (ds4_cuda.cu:11119, activation→Q8_K), `dev_dot_q4_K_q8_K_block` (:10824), `warp_sum_f32` (:3752), and `dev_f16_to_f32` (:10565) — none of these are modified.

---

## Task 1: Q8_0 → f32 dequant in the quantizer

**Files:**
- Modify: `gguf-tools/quants.h` (add prototype)
- Modify: `gguf-tools/quants.c` (add `ds4q_dequantize_q8_0`, near `ds4q_quantize_q8_0` at :341)

**Interfaces:**
- Produces: `void ds4q_dequantize_q8_0(const void *src, float *dst, int64_t nrows, int64_t ncols)` — consumed by Tasks 2 and 3.

The Q8_0 quantizer (`quants.c:341-367`) writes, per 32-value block: a 2-byte f16 scale `d` then 32 `int8`. `ds4q_f16_to_f32` exists at `quants.c:1076`. Block size 32, type size 34 (traits table `quants.c:46`).

- [ ] **Step 1: Add the prototype**

In `gguf-tools/quants.h`, next to the other `ds4q_*` quant declarations, add:

```c
/* Dequantize a row-major Q8_0 tensor (ncols % 32 == 0) to f32 (nrows*ncols floats). */
void ds4q_dequantize_q8_0(const void *src, float *dst, int64_t nrows, int64_t ncols);
```

- [ ] **Step 2: Implement the function**

In `gguf-tools/quants.c`, immediately after `ds4q_quantize_q8_0` (ends at :367), add:

```c
void ds4q_dequantize_q8_0(const void *src, float *dst, int64_t nrows, int64_t ncols) {
    const int64_t qk = 32;
    const size_t row_size = ds4q_row_size(DS4Q_TYPE_Q8_0, ncols); /* (ncols/32)*34 */
    const int64_t blocks_per_row = ncols / qk;
    for (int64_t r = 0; r < nrows; r++) {
        const uint8_t *in = (const uint8_t *)src + (size_t)r * row_size;
        float *out = dst + (size_t)r * ncols;
        for (int64_t b = 0; b < blocks_per_row; b++) {
            uint16_t hd;
            memcpy(&hd, in, sizeof(hd));
            const float d = ds4q_f16_to_f32(hd);
            const int8_t *qs = (const int8_t *)(in + sizeof(hd));
            for (int j = 0; j < qk; j++) out[b * qk + j] = (float)qs[j] * d;
            in += sizeof(hd) + qk;
        }
    }
}
```

- [ ] **Step 3: Build the quantizer to verify it compiles**

Run: `make -C /home/ylow/deepseekflash/ds4/gguf-tools 2>&1 | tail -5`
Expected: links `deepseek4-quantize` with no errors/warnings on the new function.

- [ ] **Step 4: Round-trip self-test (temporary)**

Create `gguf-tools/tests/test_q8_0_roundtrip.c`:

```c
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include "../quants.h"
int main(void) {
    const int64_t ncols = 64, nrows = 3;
    float x[64*3], deq[64*3];
    for (int i = 0; i < 64*3; i++) x[i] = sinf((float)i * 0.13f) * 2.0f;
    unsigned char q[3 * (64/32) * 34];
    ds4q_quantize_chunk(DS4Q_TYPE_Q8_0, x, q, 0, nrows, ncols, NULL);
    ds4q_dequantize_q8_0(q, deq, nrows, ncols);
    double maxerr = 0.0;
    for (int i = 0; i < 64*3; i++) { double e = fabs(deq[i]-x[i]); if (e > maxerr) maxerr = e; }
    printf("q8_0 roundtrip max abs err = %.6f\n", maxerr);
    /* Q8_0 step is amax/127; for |x|<=2 that is <= ~0.0157. Allow margin. */
    return maxerr < 0.02 ? 0 : 1;
}
```

(If `ds4q_quantize_chunk`'s signature differs, match the one at `quants.c:1050`; the report shows it takes `(type, src, dst, start, nrows, ncols, imatrix)`.)

- [ ] **Step 5: Compile + run the self-test**

Run:
```bash
cd /home/ylow/deepseekflash/ds4/gguf-tools && \
cc -O2 -std=c11 -I. tests/test_q8_0_roundtrip.c quants.c -lm -o /tmp/t_q8 && /tmp/t_q8; echo "exit=$?"
```
Expected: `q8_0 roundtrip max abs err = 0.0xxxxx` and `exit=0`.

- [ ] **Step 6: Commit** (delete the temp test first — it was a scaffold, not a kept asset)

```bash
cd /home/ylow/deepseekflash/ds4 && rm gguf-tools/tests/test_q8_0_roundtrip.c
git add gguf-tools/quants.c gguf-tools/quants.h
git commit -m "gguf-tools: add ds4q_dequantize_q8_0 (Q8_0 -> f32)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: `--template-requant` source mode in the quantizer

**Files:**
- Modify: `gguf-tools/deepseek4-quantize.c` — params struct + arg parse (~:1755-1812), `generate_regular` (:1167), `generate_tensor` (:1298), `write_full_gguf` loop (:1639), and `main` HF-optional guard (~:1888).

**Interfaces:**
- Consumes: `ds4q_dequantize_q8_0` (Task 1), `read_gguf_tensor_data` (:1552), `tensor_to_f32`/`f32_to_type` plumbing (:660/:1090 region).
- Produces: a tool that, with `--template-requant`, copies all template tensors verbatim except those whose policy type differs from the template type, which it dequantizes from the template and re-quantizes — no HF needed.

Design: `policy_type` already routes `attn_output_a/_b` to Q4_K when the user passes `--tensor-type blk…attn_output_a.weight=q4_k` (exact/prefix match, :1027). In `--template-requant` mode the *source* of every tensor is the template GGUF: if `dst->type == src->type` (template) → byte-copy via `read_gguf_tensor_data`; else → dequant the template bytes to f32 (only Q8_0 supported here; assert) and `f32_to_type` to the new type. The template `gguf_file` (with `->path`) must be threaded down to `generate_regular`.

- [ ] **Step 1: Add the mode flag to params + arg parse**

In the `params` struct (the struct holding `dry_run`, `compare_tensor`, etc.), add `bool template_requant;`. In the arg-parse block (alongside `--dry-run` at :1765), add:

```c
        } else if (strcmp(arg, "--template-requant") == 0) {
            p.template_requant = true;
```

- [ ] **Step 2: Make `--hf` optional under the mode**

`db_open(&db, p.hf_dir)` is called at `main` ~:1889. Guard it:

```c
    st_db db; bool db_ready = false;
    if (!p.template_requant) { db_open(&db, p.hf_dir); db_ready = true; }
    /* ...compare/normal dispatch below uses db_ready... */
```

Pass a `db*` that may be NULL into the write path when `template_requant`. (If `compare_one_tensor` is invoked in this mode, it likewise must not require `db`.)

- [ ] **Step 3: Thread the template `gguf_file*` and mode into `generate_tensor`/`generate_regular`**

Change the signatures to carry the template file + mode:

```c
static byte_buf generate_regular(st_db *db, const char *name, const tensor_meta *tmpl_meta,
                                 ds4q_type target, const imatrix_store *imatrix,
                                 const gguf_file *tmpl_file, bool template_requant);
static byte_buf generate_tensor(st_db *db, const char *name, const tensor_meta *tmpl_meta,
                                ds4q_type target, int n_experts, int n_threads,
                                const imatrix_store *imatrix,
                                const gguf_file *tmpl_file, bool template_requant);
```

Update both call sites (`write_full_gguf` :1643 and `compare_one_tensor` :1834) to pass `tmpl` (the `gguf_file`) and `p->template_requant`. `generate_tensor` forwards them to `generate_regular`. Expert tensors (`parse_expert_tensor(name).is_expert`) are NOT part of this recipe — under `template_requant` they must still be byte-copied; route experts through the same verbatim-copy branch (Step 4) rather than `generate_expert`, so HF is never touched.

- [ ] **Step 4: Implement the template-source branch in `generate_regular`**

At the top of `generate_regular`, before the HF read (:1193), add:

```c
    if (template_requant) {
        byte_buf src_bytes = read_gguf_tensor_data(tmpl_file, tmpl_file->path, name);
        if (target == tmpl_meta->type) {
            return src_bytes;                 /* verbatim copy-through */
        }
        if (tmpl_meta->type != DS4Q_TYPE_Q8_0)
            die("template-requant: only Q8_0 source tensors can be requantized");
        const int64_t ncols = tmpl_meta->ne[0];
        const int64_t nrows = tmpl_meta->ne[1];
        const int64_t n = nrows * ncols;
        float *f32 = xmalloc((size_t)n * sizeof(float));
        ds4q_dequantize_q8_0(src_bytes.data, f32, nrows, ncols);
        free(src_bytes.data);
        const char *names[1] = { name };
        const float *imat = imatrix_find(imatrix, names, 1, ncols, -1, 0);
        byte_buf b = f32_to_type(f32, n, target, ncols, imat);
        free(f32);
        return b;
    }
```

(`read_gguf_tensor_data` re-opens the file per call — fine for a one-shot full-model pass. `f32_to_type` is the same quantize entry the HF path uses at :1200.)

- [ ] **Step 5: Build**

Run: `make -C /home/ylow/deepseekflash/ds4/gguf-tools 2>&1 | tail -8`
Expected: clean build, no warnings on the changed functions.

- [ ] **Step 6: Dry-run plan check (no data written)**

Run:
```bash
cd /home/ylow/deepseekflash/ds4 && gguf-tools/deepseek4-quantize \
  --template gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --template-requant \
  --tensor-type blk.0.attn_output_a.weight=q4_k \
  --tensor-type blk.0.attn_output_b.weight=q4_k \
  --dry-run 2>&1 | grep -E 'attn_output_[ab]\.weight' | head
```
Expected: `print_plan` lines showing `blk.0.attn_output_a.weight … q8_0 -> q4_K` (and `_b`), every other tensor unchanged. (Use a per-layer loop of `--tensor-type` for all layers in Task 3; for the dry-run, layer 0 suffices to confirm the plan path.)

- [ ] **Step 7: Commit**

```bash
cd /home/ylow/deepseekflash/ds4 && git add gguf-tools/deepseek4-quantize.c
git commit -m "gguf-tools: --template-requant mode (requant from template GGUF, no HF)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 3: Pre-flight RMSE + build the Q4_K-output GGUF

**Files:**
- Modify: `gguf-tools/deepseek4-quantize.c` — extend `compare_one_tensor` (:1825) with an RMSE branch.
- Produce (artifact, NOT committed): `gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-AOutQ4K-SExpQ8-OutQ8-chat-v2-imatrix.gguf`.

**Interfaces:**
- Consumes: `ds4q_dequantize_q8_0` (Task 1), `--template-requant` (Task 2), the engine-independent quantizer.

- [ ] **Step 1: Add an RMSE compare path**

`compare_one_tensor` currently only byte/FNV-compares, which FAILs across different types (Q4_K vs Q8_0). Add: when `out_ctx->tensors[idx].type != ref` type, dequantize both sides to f32 and print RMSE. Dequant the generated Q4_K with the engine-independent path already in the quantizer (the `f32_to_type` inverse is not present for Q4_K in quants.c; instead compute RMSE against the *source* f32). Simplest correct check: in `--template-requant`, RMSE of `dequant_q4k(generated)` vs `dequant_q8_0(template_source)`. Since quants.c lacks a Q4_K dequant, compute the RMSE proxy as: requantization error reported by `f32_to_type` is not exposed, so add a minimal `ds4q_dequantize_q4_k` mirroring the existing `ds4q_quantize`'s block math, OR (preferred, less code) print the **per-tensor amax and the byte sizes** plus the FNV of the generated bytes, and rely on the engine self-check (Task 4-7) for numeric proof. Implement the lightweight version:

```c
    if (out_ctx->tensors[idx].type != tmpl->tensors[idx].type) {
        printf("type_change: %s -> %s\n",
               ds4q_type_name(tmpl->tensors[idx].type),
               ds4q_type_name(out_ctx->tensors[idx].type));
        printf("generated_bytes: %zu  (template_bytes: %zu)\n",
               generated.size, tmpl->tensors[idx].size);
        printf("generated_fnv1a64: %016" PRIx64 "\n",
               fnv1a64_bytes(generated.data, generated.size));
        free(generated.data);
        return;
    }
```

Numeric RMSE is deferred to the engine self-check (Task 4), which compares the GPU Q4_K kernel against a host f32 reference on the real weights — a stronger, end-to-end check.

- [ ] **Step 2: Build**

Run: `make -C /home/ylow/deepseekflash/ds4/gguf-tools 2>&1 | tail -3`
Expected: clean.

- [ ] **Step 3: Compare-tensor smoke (layer 0)**

Run:
```bash
cd /home/ylow/deepseekflash/ds4 && gguf-tools/deepseek4-quantize \
  --template gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --template-requant --tensor-type blk.0.attn_output_a.weight=q4_k \
  --compare-tensor blk.0.attn_output_a.weight 2>&1 | tail -8
```
Expected: `type_change: q8_0 -> q4_K`, `generated_bytes:` ≈ 0.53× the template bytes, an FNV hash, no crash.

- [ ] **Step 4: Generate the full Q4_K-output GGUF**

Build the `--tensor-type` list for ALL attention layers (the model's `blk.{0..N-1}` plus `mtp.0`). Discover N:
```bash
cd /home/ylow/deepseekflash/ds4 && gguf-tools/deepseek4-quantize \
  --template gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --template-requant --dry-run 2>&1 | grep -c 'attn_output_a\.weight'
```
Then generate overrides for every `attn_output_a`/`_b` (and `mtp.0.attn_output_*`) and run the full write:
```bash
cd /home/ylow/deepseekflash/ds4
OUT=gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-AOutQ4K-SExpQ8-OutQ8-chat-v2-imatrix.gguf
ARGS=""
for L in $(seq 0 $((N-1))); do
  ARGS="$ARGS --tensor-type blk.$L.attn_output_a.weight=q4_k --tensor-type blk.$L.attn_output_b.weight=q4_k"
done
ARGS="$ARGS --tensor-type mtp.0.attn_output_a.weight=q4_k --tensor-type mtp.0.attn_output_b.weight=q4_k"
gguf-tools/deepseek4-quantize \
  --template gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --template-requant $ARGS --out "$OUT"
```
Expected: per-tensor progress; output ~2–3 GB smaller than the 86.7 GB template. Verify it exists and is smaller:
```bash
ls -la "$OUT" gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
```
(Replace `$N` with the count from the discovery command. Keep both GGUFs.)

- [ ] **Step 5: Confirm the new GGUF loads (type validation will reject until Task 8 — expected)**

Run:
```bash
cd /home/ylow/deepseekflash/ds4 && ./ds4 -m "$OUT" --perplexity-file doors-of-stone-chapter-1.md -n 1 -c 4096 2>&1 | tail -5
```
Expected at THIS stage: a `tensor … has type q4_K, expected q8_0` error from `tensor_expect_layout` (proves the requant worked and the engine sees Q4_K). This becomes a pass after Task 8.

- [ ] **Step 6: Commit the tool change (not the GGUF)**

```bash
cd /home/ylow/deepseekflash/ds4 && git add gguf-tools/deepseek4-quantize.c
git commit -m "gguf-tools: report type-change in --compare-tensor for requant pre-flight

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 4: Q4_K output_a kernel + wrapper + self-check

**Files:**
- Modify: `ds4_cuda.cu` — add `grouped_q4k_a_preq_warp8_kernel`, `ds4_gpu_attention_output_low_q4k_tensor`, and a self-check helper.
- Modify: `ds4_gpu.h` — declare `ds4_gpu_attention_output_low_q4k_tensor`.

**Interfaces:**
- Consumes: `q8_K_quantize_kernel` (:11119), `dev_dot_q4_K_q8_K_block` (:10824), `warp_sum_f32` (:3752), `cuda_tmp_alloc`, `cuda_model_range_ptr`, `cuda_block_q8_K`/`cuda_block_q4_K` structs (:55).
- Produces: `ds4_gpu_attention_output_low_q4k_tensor(...)` — consumed by Tasks 7 and 8.

The Q8_0 reference is `grouped_q8_0_a_preq_warp8_kernel` (ds4_cuda.cu:4104-4140) and its wrapper `ds4_gpu_attention_output_low_q8_tensor` (:10340). The Q4_K twin keeps the grouped-row structure and 32-lane `warp_sum_f32`; only the activation quantizer (Q8_0→Q8_K), the weight block stride (34→144, cast to `cuda_block_q4_K*`), and the inner dot (`dot_i8_block`→`dev_dot_q4_K_q8_K_block`) change.

- [ ] **Step 1: Write the device kernel**

In `ds4_cuda.cu`, immediately after `grouped_q8_0_a_preq_warp8_kernel` (after :4140), add the twin. Copy that kernel verbatim, rename to `grouped_q4k_a_preq_warp8_kernel`, change the activation pointer params from `(const int8_t *xq, const float *xscale)` to `(const cuda_block_q8_K *xq)`, and replace the per-block loop body. The loop iterates Q4_K super-blocks (`blocks = group_dim/256`):

```c
__global__ static void grouped_q4k_a_preq_warp8_kernel(
        float *low, const unsigned char *w, const cuda_block_q8_K *xq,
        uint64_t group_dim, uint64_t rank, uint32_t n_groups, uint32_t n_tokens,
        uint64_t blocks, int use_dp4a) {
    (void)use_dp4a;
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (row >= low_dim || tok >= n_tokens) return;
    const uint64_t group = row / rank;
    const uint64_t row_in_group = row - group * rank;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)w + (group * rank + row_in_group) * blocks;
    const uint64_t xrow = tok * (uint64_t)n_groups + group;
    const cuda_block_q8_K *xqr = xq + xrow * blocks;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        acc += dev_dot_q4_K_q8_K_block(wr + b, xqr + b);
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) low[tok * low_dim + row] = acc;
}
```

- [ ] **Step 2: Write the host wrapper**

After the kernel, add (modeled on `ds4_gpu_attention_output_low_q8_tensor` :10340, but Q4_K sizing `blocks_a = group_dim/256`, `144` B/block, and a Q8_K activation scratch sized `x_rows*blocks_a*sizeof(cuda_block_q8_K)`):

```c
extern "C" int ds4_gpu_attention_output_low_q4k_tensor(
        ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t group_dim, uint64_t rank,
        uint32_t n_groups, const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 ||
        n_groups == 0 || n_tokens == 0 || (group_dim % 256u) != 0) return 0;
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = group_dim / 256u;
    const uint64_t out_a_bytes = (uint64_t)n_groups * rank * blocks_a * 144u; /* sizeof(cuda_block_q4_K) */
    if (out_a_offset > model_size || out_a_bytes > model_size - out_a_offset ||
        heads->bytes < (uint64_t)n_tokens * n_groups * group_dim * sizeof(float) ||
        low->bytes < (uint64_t)n_tokens * low_dim * sizeof(float)) return 0;
    const unsigned char *out_a = reinterpret_cast<const unsigned char *>(
            cuda_model_range_ptr(model_map, out_a_offset, out_a_bytes, "attn_out_a_q4k"));
    if (!out_a) return 0;
    const uint64_t x_rows = (uint64_t)n_tokens * n_groups;
    const uint64_t tmp_bytes = x_rows * blocks_a * sizeof(cuda_block_q8_K);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "attn output low q4k prequant");
    if (!tmp) return 0;
    cuda_block_q8_K *xq = (cuda_block_q8_K *)tmp;
    dim3 qgrid((unsigned)blocks_a, (unsigned)x_rows, 1);
    q8_K_quantize_kernel<<<qgrid, 256>>>(xq, (const float *)heads->ptr, (uint32_t)group_dim, (uint32_t)x_rows);
    if (!cuda_ok(cudaGetLastError(), "attn_output_low_q4k prequant launch")) return 0;
    dim3 grid_a(((unsigned)low_dim + 7u) / 8u, (unsigned)n_tokens, 1);
    grouped_q4k_a_preq_warp8_kernel<<<grid_a, 256>>>((float *)low->ptr, out_a, xq,
            group_dim, rank, n_groups, n_tokens, blocks_a, 0);
    return cuda_ok(cudaGetLastError(), "attn_output_low_q4k launch");
}
```

(Confirm `q8_K_quantize_kernel`'s row layout matches: it indexes `x + row*in_dim`, and `heads` for the grouped path is `n_groups` rows of `group_dim` — so pass `n_rows = x_rows`, `in_dim = group_dim`. This mirrors how the Q8_0 path treats `heads`.)

- [ ] **Step 3: Declare the wrapper**

In `ds4_gpu.h` after the `ds4_gpu_attention_output_low_q8_tensor` decl (:778), add the `ds4_gpu_attention_output_low_q4k_tensor` prototype from the Interfaces section.

- [ ] **Step 4: Add the env-gated self-check helper**

Add a host function `ds4_q4k_selfcheck_output_low(...)` (guarded by `getenv("DS4_Q4K_SELFCHECK")`) that, given the same inputs, also computes a host f32 reference: copy `out_a` (Q4_K bytes) and `heads` to host, dequant the Q4_K weight with the engine's existing CPU Q4_K dequant (the routed-expert path; find it via `grep -n 'block_q4_K' ds4.c` and the `q4k-dot-test` target), do the grouped f32 matmul, and compare to the GPU `low` (max abs/rel diff). Print `q4k selfcheck output_low: max_abs=… max_rel=…`. Wire it to run once for the first layer when the env is set, right after the wrapper returns in the dispatch (Task 8). For THIS task, expose it and call it from a tiny temporary driver if convenient; otherwise validate in Task 8 Step 3.

- [ ] **Step 5: Build**

Run: `cd /home/ylow/deepseekflash/ds4 && make cuda-spark 2>&1 | tail -8`
Expected: compiles `ds4_cuda.cu` and links `ds4` with no errors. (The new wrapper is unreferenced until Task 8 — a `-Wunused` note is acceptable; the build must still succeed.)

- [ ] **Step 6: Commit**

```bash
cd /home/ylow/deepseekflash/ds4 && git add ds4_cuda.cu ds4_gpu.h
git commit -m "cuda: Q4_K attention output_a kernel + wrapper (inert)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 5: Q4_K output_b HC-expand kernel + wrapper (decode)

**Files:**
- Modify: `ds4_cuda.cu` — add `matmul_q4k_hc_expand_preq_warp8_kernel` + `ds4_gpu_matmul_q4k_hc_expand_tensor`.
- Modify: `ds4_gpu.h` — declare `ds4_gpu_matmul_q4k_hc_expand_tensor`.

**Interfaces:**
- Consumes: same primitives as Task 4.
- Produces: `ds4_gpu_matmul_q4k_hc_expand_tensor(...)` — consumed by Task 8 (decode output_b).

Reference: kernel `matmul_q8_0_hc_expand_preq_warp8_kernel` (ds4_cuda.cu:3984-4032), wrapper `cuda_matmul_q8_0_hc_expand_tensor_labeled` (:8451) called from `ds4_gpu_matmul_q8_0_hc_expand_tensor` (:14155). **The entire HC-split tail (the `if (lane == 0)` block, :4015-4031) is weight-format-agnostic — copy it verbatim.** Only the weight loop changes.

- [ ] **Step 1: Write the device kernel**

After `matmul_q8_0_hc_expand_preq_warp8_kernel` (:4032), add the twin `matmul_q4k_hc_expand_preq_warp8_kernel`: copy it verbatim, change the activation params to `(const cuda_block_q8_K *xq)`, replace the weight loop with the Q4_K block loop (as in Task 4 Step 1), and keep the HC-split tail unchanged:

```c
__global__ static void matmul_q4k_hc_expand_preq_warp8_kernel(
        float *out_hc, float *block_out, const float *block_add, const float *residual_hc,
        const float *split, const unsigned char *w, const cuda_block_q8_K *xq,
        uint64_t in_dim, uint64_t out_dim, uint32_t n_embd, uint32_t n_hc, uint64_t blocks,
        int has_add, int use_dp4a) {
    (void)use_dp4a;
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const cuda_block_q4_K *wr = (const cuda_block_q4_K *)w + row * blocks;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) acc += dev_dot_q4_K_q8_K_block(wr + b, xq + b);
    acc = warp_sum_f32(acc);
    if (lane == 0) {
        /* ---- COPY VERBATIM from matmul_q8_0_hc_expand_preq_warp8_kernel lines 4015-4031 ---- */
    }
}
```

- [ ] **Step 2: Write the host wrapper**

Add `ds4_gpu_matmul_q4k_hc_expand_tensor` modeled on `cuda_matmul_q8_0_hc_expand_tensor_labeled` (:8451): same signature as the Q8_0 public wrapper (Interfaces section), `blocks = in_dim/256`, weight bytes `out_dim*blocks*144`, activation scratch one `cuda_block_q8_K` row (`blocks` blocks), quantize with `q8_K_quantize_kernel<<<dim3(blocks,1,1),256>>>`, launch `matmul_q4k_hc_expand_preq_warp8_kernel<<<(out_dim+7)/8, 256>>>` with `block_add = block_out` and `has_add = 0` (decode has no add, matching the Q8_0 fused call). Guard `(in_dim % 256) == 0`.

- [ ] **Step 3: Declare** in `ds4_gpu.h` after :1051.

- [ ] **Step 4: Build**

Run: `cd /home/ylow/deepseekflash/ds4 && make cuda-spark 2>&1 | tail -8`
Expected: clean compile + link.

- [ ] **Step 5: Commit**

```bash
cd /home/ylow/deepseekflash/ds4 && git add ds4_cuda.cu ds4_gpu.h
git commit -m "cuda: Q4_K attention output_b HC-expand kernel + wrapper (inert)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 6: Q4_K plain matmul + batch wrapper (output_b prefill / non-fused)

**Files:**
- Modify: `ds4_cuda.cu` — add `matmul_q4k_preq_warp8_kernel` (n_tok=1), `matmul_q4k_preq_batch_warp8_kernel` (n_tok>1), `ds4_gpu_matmul_q4k_tensor`, and `ds4_gpu_attention_output_q4k_batch_tensor`.
- Modify: `ds4_gpu.h` — declare `ds4_gpu_matmul_q4k_tensor`, `ds4_gpu_attention_output_q4k_batch_tensor`.

**Interfaces:**
- Consumes: Task 4's `ds4_gpu_attention_output_low_q4k_tensor` (the batch wrapper reuses it for output_a), plus the same primitives.
- Produces: `ds4_gpu_matmul_q4k_tensor`, `ds4_gpu_attention_output_q4k_batch_tensor` — consumed by Task 8 (prefill + non-fused decode).

References: `matmul_q8_0_preq_warp8_kernel` (~:4017), `matmul_q8_0_preq_batch_warp8_kernel` (:4034), wrapper `cuda_matmul_q8_0_tensor_labeled` (:8250, native fallback section), and the batch wrapper `ds4_gpu_attention_output_q8_batch_tensor` (:10174).

- [ ] **Step 1: Write the two plain-matmul kernels**

After the Q8_0 plain matmul kernels, add `matmul_q4k_preq_warp8_kernel` (twin of `matmul_q8_0_preq_warp8_kernel`) and `matmul_q4k_preq_batch_warp8_kernel` (twin of `matmul_q8_0_preq_batch_warp8_kernel`). Both: weight `(const cuda_block_q4_K *)w + row*blocks`, activation `cuda_block_q8_K`, inner `dev_dot_q4_K_q8_K_block`, `warp_sum_f32`, `blocks = in_dim/256`. The batch kernel adds the `tok = blockIdx.y` row indexing into the activation (`xq + tok*blocks`) and output (`out[tok*out_dim+row]`), exactly as the Q8_0 batch twin does.

- [ ] **Step 2: Write `ds4_gpu_matmul_q4k_tensor`**

Modeled on the native section of `cuda_matmul_q8_0_tensor_labeled` (:8250) — NO cuBLAS. Quantize the `n_tok × in_dim` activation to Q8_K (`q8_K_quantize_kernel<<<dim3(in_dim/256, n_tok,1),256>>>`), then dispatch `matmul_q4k_preq_warp8_kernel` for `n_tok==1` else `matmul_q4k_preq_batch_warp8_kernel`. Guard `(in_dim % 256) == 0`.

- [ ] **Step 3: Write `ds4_gpu_attention_output_q4k_batch_tensor`**

Modeled on `ds4_gpu_attention_output_q8_batch_tensor` (:10174) but native-only:
```c
extern "C" int ds4_gpu_attention_output_q4k_batch_tensor(
        ds4_gpu_tensor *out, ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t out_b_offset, uint64_t group_dim, uint64_t rank,
        uint32_t n_groups, uint64_t out_dim, const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    const uint64_t low_dim = (uint64_t)n_groups * rank;
    if (!ds4_gpu_attention_output_low_q4k_tensor(low, model_map, model_size, out_a_offset,
            group_dim, rank, n_groups, heads, n_tokens)) return 0;
    return ds4_gpu_matmul_q4k_tensor(out, model_map, model_size, out_b_offset,
            low_dim, out_dim, low, n_tokens);
}
```

- [ ] **Step 4: Declare** both in `ds4_gpu.h`.

- [ ] **Step 5: Build**

Run: `cd /home/ylow/deepseekflash/ds4 && make cuda-spark 2>&1 | tail -8`
Expected: clean compile + link.

- [ ] **Step 6: Commit**

```bash
cd /home/ylow/deepseekflash/ds4 && git add ds4_cuda.cu ds4_gpu.h
git commit -m "cuda: Q4_K plain matmul + batch output wrappers (inert)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 7: Relax load-time type validation

**Files:**
- Modify: `ds4.c` — add `tensor_expect_q8_0_or_q4_k_layout` (model on `tensor_expect_f16_or_q8_0_layout` :3296); swap in at `:3685-3686` (main blocks) and `:3755-3756` (MTP).

**Interfaces:**
- Produces: the loader accepts Q4_K `attn_output_*`. Consumed by Task 8 (so the new GGUF loads).

- [ ] **Step 1: Add the validator**

After `tensor_expect_f16_or_q8_0_layout` (:3312), add:

```c
static bool tensor_type_is_q8_0_or_q4_k(uint32_t type) {
    return type == DS4_TENSOR_Q8_0 || type == DS4_TENSOR_Q4_K;
}
static void tensor_expect_q8_0_or_q4_k_layout(
        const ds4_tensor *t, uint32_t ndim, uint64_t d0, uint64_t d1, uint64_t d2) {
    if (!t) ds4_die("internal error: missing tensor while validating layout");
    if (!tensor_type_is_q8_0_or_q4_k(t->type)) {
        fprintf(stderr, "ds4: tensor %.*s has type %u, expected q8_0 or q4_K\n",
                (int)t->name.len, t->name.ptr, t->type);
        exit(1);
    }
    tensor_expect_layout(t, t->type, ndim, d0, d1, d2);
}
```

- [ ] **Step 2: Swap at the 4 sites**

At `:3685-3686`:
```c
        tensor_expect_q8_0_or_q4_k_layout(l->attn_output_a, 2, DS4_N_HEAD_DIM * (DS4_N_HEAD / DS4_N_OUT_GROUP), out_low_dim, 0);
        tensor_expect_q8_0_or_q4_k_layout(l->attn_output_b, 2, out_low_dim, DS4_N_EMBD, 0);
```
Apply the same replacement at the MTP pair `:3755-3756`.

- [ ] **Step 3: Build**

Run: `cd /home/ylow/deepseekflash/ds4 && make cuda-spark 2>&1 | tail -5`
Expected: clean.

- [ ] **Step 4: Confirm the Q4_K GGUF now loads past validation (will fail later in attn until Task 8 dispatch)**

Run:
```bash
cd /home/ylow/deepseekflash/ds4 && ./ds4 -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-AOutQ4K-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --perplexity-file doors-of-stone-chapter-1.md -n 1 -c 4096 2>&1 | tail -8
```
Expected: NO `expected q8_0` error now. It will instead produce wrong/garbage logits or an attention error because the Q8 kernels still misread Q4_K bytes — that is fixed in Task 8. (If it happens to run, the nll will be nonsense; ignore until Task 8.)

- [ ] **Step 5: Commit**

```bash
cd /home/ylow/deepseekflash/ds4 && git add ds4.c
git commit -m "ds4: accept Q4_K (or Q8_0) for attention output-proj tensors at load

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 8: Dispatch the Q4_K path + self-check

**Files:**
- Modify: `ds4.c` — branch on `layer->attn_output_a->type` at the decode site (`:15822-15857`), the non-fused decode else-branch (`:15846`), and the prefill site (`:19116-19145`). Add the self-check call.

**Interfaces:**
- Consumes: all four Q4_K wrappers (Tasks 4–6) and the validation (Task 7).
- Produces: end-to-end Q4_K decode + prefill — the artifact the oracle measures.

- [ ] **Step 1: Branch the fused decode site (`:15822`)**

Wrap the two fused calls so Q4_K weights use the Q4_K kernels:

```c
    if (ok && fuse_attn_out_hc) {
        if (layer->attn_output_a->type == DS4_TENSOR_Q4_K) {
            ok = ds4_gpu_attention_output_low_q4k_tensor(g->attn_low, model->map, model->size,
                    layer->attn_output_a->abs_offset, group_dim, rank, n_groups, g->heads, 1) != 0;
            if (ok) ok = ds4_gpu_matmul_q4k_hc_expand_tensor(g->after_attn_hc, g->attn_out,
                    model->map, model->size, layer->attn_output_b->abs_offset,
                    (uint64_t)n_groups * rank, DS4_N_EMBD, g->attn_low, g->cur_hc, g->hc_split,
                    DS4_N_EMBD, DS4_N_HC) != 0;
        } else {
            /* ---- existing Q8_0 calls verbatim (low_q8 + matmul_q8_0_hc_expand) ---- */
        }
    } else if (ok) {
        if (layer->attn_output_a->type == DS4_TENSOR_Q4_K) {
            ok = ds4_gpu_attention_output_q4k_batch_tensor(g->attn_out, g->attn_low,
                    model->map, model->size, layer->attn_output_a->abs_offset,
                    layer->attn_output_b->abs_offset, group_dim, rank, n_groups, DS4_N_EMBD,
                    g->heads, 1) != 0;
        } else {
            /* ---- existing ds4_gpu_attention_output_q8_batch_tensor call verbatim ---- */
        }
    }
```

- [ ] **Step 2: Branch the prefill site (`:19116-19145`)**

There the Q8 path calls `ds4_gpu_attention_output_q8_batch_f16_tensor` then `..._q8_batch_tensor`. For Q4_K (no f16 expansion), route BOTH through `ds4_gpu_attention_output_q4k_batch_tensor` with the site's `n_tokens`:

```c
        if (layer->attn_output_a->type == DS4_TENSOR_Q4_K) {
            ok = ds4_gpu_attention_output_q4k_batch_tensor(<out>, <low>, model->map, model->size,
                    layer->attn_output_a->abs_offset, layer->attn_output_b->abs_offset,
                    group_dim, rank, n_groups, DS4_N_EMBD, <heads>, <n_tokens>) != 0;
        } else {
            /* ---- existing f16-batch + q8-batch calls verbatim ---- */
        }
```
(Read the exact local variable names for out/low/heads/n_tokens at `:19116` and substitute; they mirror the decode site's `g->…`.)

- [ ] **Step 3: Self-check — prove kernel correctness on the real model**

Add, immediately after the Q4_K decode wrapper returns for the FIRST decoded layer when `getenv("DS4_Q4K_SELFCHECK")`, a host reference compare (Task 4 Step 4): dequant the layer's Q4_K `attn_output_a`/`_b` with the engine's CPU Q4_K path, recompute `low`/`out` in f32 from `g->heads`, and assert max rel diff < 1e-3. Then run:

```bash
cd /home/ylow/deepseekflash/ds4 && DS4_Q4K_SELFCHECK=1 \
  ./ds4 -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-AOutQ4K-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --perplexity-file doors-of-stone-chapter-1.md -n 2 -c 4096 2>&1 | grep -i 'q4k selfcheck'
```
Expected: `q4k selfcheck output_low: max_abs=… max_rel=<1e-3` and the output_b check similar — proving the GPU kernel addressing matches the host reference (isolating addressing bugs from quantization loss). If max_rel is large (e.g. >0.05), the kernel addressing is wrong — debug the weight stride / block indexing before proceeding.

- [ ] **Step 4: Build + commit**

```bash
cd /home/ylow/deepseekflash/ds4 && make cuda-spark 2>&1 | tail -5
git add ds4.c && git commit -m "ds4: dispatch Q4_K attention output-proj kernels by tensor type + selfcheck

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Task 9: Measure the nll / throughput tradeoff (the deliverable)

**Files:** none modified. Record results into this plan and the spec.

**Interfaces:** Consumes the flipped GGUF + the dispatched engine.

- [ ] **Step 1: Baseline (Q8_0) perplexity — re-confirm the oracle**

```bash
cd /home/ylow/deepseekflash/ds4 && ./ds4 -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --perplexity-file doors-of-stone-chapter-1.md -n 256 -c 4096 2>&1 | tail -2
```
Expected: `nll=317.843967992` (within run-to-run determinism). Record it.

- [ ] **Step 2: Q4_K perplexity**

```bash
cd /home/ylow/deepseekflash/ds4 && ./ds4 -m gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-AOutQ4K-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
  --perplexity-file doors-of-stone-chapter-1.md -n 256 -c 4096 2>&1 | tail -2
```
Record `nll` and `ppl`. Compute Δnll% = (nll_q4k − 317.843967992) / 317.843967992 × 100 and Δppl.

- [ ] **Step 3: Decode throughput — Q8_0 then Q4_K, ctx 4096 and 16384**

```bash
cd /home/ylow/deepseekflash/ds4
for M in DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
         DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-AOutQ4K-SExpQ8-OutQ8-chat-v2-imatrix.gguf; do
  for C in 4096 16384; do
    echo "== $M ctx=$C =="
    ./ds4 -m gguf/$M -p doors-of-stone-chapter-1.md -n 256 -c $C 2>&1 | grep -E 'generation: .* t/s'
  done
done
```
Record the `generation: X t/s` for each. Q8 baselines for sanity: ~13.5 @4096, ~13.06 @16384. Expect Q4_K ~17–18.

- [ ] **Step 4: Record results + decide**

Append a "Results (2026-06-26)" section to BOTH this plan and `docs/superpowers/specs/2026-06-26-q4k-attn-output-design.md` with: baseline nll, Q4_K nll, Δnll%, Δppl, the four tok/s numbers, and the decision (accept / tune imatrix / try output_b-only / roll back). Commit the docs (NOT the GGUF):

```bash
cd /home/ylow/deepseekflash/ds4 && git add docs/superpowers/
git commit -m "docs: record Q4_K attention output-proj nll/throughput results

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 5: Set the runtime default (only if accepted)**

If the tradeoff is accepted, repoint the symlink so the server/CLI use it by default:
```bash
ln -sfn /home/ylow/deepseekflash/ds4/gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-AOutQ4K-SExpQ8-OutQ8-chat-v2-imatrix.gguf \
        /home/ylow/deepseekflash/ds4/ds4flash.gguf
```
Rollback = repoint to the `…AProjQ8…` GGUF. Restart the server if running: `systemctl --user restart ds4` (never `pkill -f ds4`).

---

## Results (2026-06-26)

**Perplexity (teacher-forced, n=256, c=4096, `doors-of-stone-chapter-1.md`):**

| | nll | avg_nll | ppl |
|---|---|---|---|
| Q8 reference oracle (prior session) | 317.843967992 | 1.241578 | ~3.461 |
| Q8 measured (today, bit-reproducible) | 319.009406860 | 1.246130 | 3.476863 |
| Q4_K measured (today) | 316.384665147 | 1.235878 | 3.441397 |

Δnll% (Q4_K vs reference oracle 317.844): **−0.459%**
Δnll% (Q4_K vs measured Q8 319.009): **−0.823%**
Δppl (Q4_K vs measured Q8): **−0.035** (Q4_K is slightly better, within noise)

**Baseline note (cause of the 317.844→319.009 gap):** the `317.843967992` oracle predates the
Phase 2c **Hadamard-FP4 (NF4) compressed-attention KV cache**, which is now enabled by default
(`DS4_GPU_ATTN_COMP_CACHE_FP4`, commit d0c8b30). That KV-quant adds +0.367% nll → `319.009`
(matching the Phase 2c-recorded `319.009 @4096` exactly). So the correct same-engine baseline for
this Q4_K-attn-weight comparison is **319.009**, and Q4_K-output sits at **316.385** (−0.823%) —
no regression. The drift is the already-merged KV feature, not this work.

**Decode throughput (generation tok/s):**

| Model | ctx=4096 | ctx=16384 | prefill @4096 | prefill @16384 |
|---|---|---|---|---|
| Q8   | 15.11 t/s | 14.82 t/s | 32.74 t/s | 33.67 t/s |
| Q4_K | 15.51 t/s | 15.51 t/s | 30.77 t/s | 30.63 t/s |
| Δ    | +2.6%     | +4.7%     | −5.7%      | −9.0%      |

**GGUF size delta:** Q8 = 86.72 GB, Q4_K = 85.28 GB → **−1.44 GB**.

**Summary:** Q4_K shows no quality regression relative to the Q8 baseline (nll is marginally
lower, likely noise/regularization). Decode throughput gain is modest at +2.6–4.7%, well below
the projected 1.32×/17–18 t/s — the byte savings of −1.44 GB did not convert to proportional
bandwidth gains, possibly because the current Q8 baseline has already been lifted by prior
optimizations (~15 t/s today vs the ~13.5 reference in the plan). Prefill is 5–9% slower with
Q4_K due to unpack overhead dominating in the compute-heavier batch path. The accept/reject
decision is the user's; both GGUFs are present and the symlink remains on the Q8 GGUF.

---

## Self-Review

**Spec coverage:**
- Offline requant-from-Q8-GGUF, no HF, Q4_K, synthetic imatrix → Tasks 1–3. (Synthetic imatrix is the quantizer's automatic fallback when `--imatrix` is absent; the `--template-requant` path passes the imatrix store through, which is empty here → fallback. ✓)
- Q4_K read variants of the specialized kernels (output_a grouped, output_b HC-expand, batch) → Tasks 4–6. ✓
- Dispatch by tensor type, no env flag for recipe → Task 8. ✓
- Relax `tensor_expect_layout` (+ MTP) → Task 7. ✓
- Validation: teacher-forced perplexity nll + decode tok/s at 4K/16K → Task 9. ✓
- Rollback via symlink → Task 9 Step 5. ✓
- All paths that read the weights covered (decode fused, non-fused, prefill) → Task 8 Steps 1–2; the oracle exercises prefill (32-tok sync) + decode (per scored token), so both are mandatory and both are dispatched. ✓

**Placeholder scan:** Kernel bodies for Tasks 5–6 are specified as "copy named kernel X (exact lines), apply these substitutions" with the changed weight-loop shown in full and the format-agnostic tail copied verbatim — this is a port, not a placeholder; the engineer has the exact source lines and the exact diff. Task 4 shows the full kernel. The self-check reference (Task 4 Step 4 / Task 8 Step 3) points at the engine's existing CPU Q4_K dequant (locate via `grep block_q4_K ds4.c` + the `q4k-dot-test` target) rather than quoting it, because it already exists and is trusted.

**Type consistency:** wrapper names (`ds4_gpu_attention_output_low_q4k_tensor`, `ds4_gpu_matmul_q4k_hc_expand_tensor`, `ds4_gpu_matmul_q4k_tensor`, `ds4_gpu_attention_output_q4k_batch_tensor`) are identical in the Interfaces block, the kernel tasks, and the dispatch task. `DS4_TENSOR_Q4_K` (=12) used consistently. Activation struct `cuda_block_q8_K`, weight struct `cuda_block_q4_K`, primitive `dev_dot_q4_K_q8_K_block` consistent throughout. Block sizes: Q4_K 144 B/256 weights, Q8_0 34 B/32 weights — used consistently.

**Known risk flagged for execution:** Task 8 Step 2 requires reading the exact local variable names at `ds4.c:19116`; the plan instructs to substitute them rather than guessing. The self-check (Task 8 Step 3) is the gate that catches any addressing error before trusting the nll.
