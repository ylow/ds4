# Setting up the DeepSeek-V4-Flash (ds4 / DwarfStar) service on the GB10

End-to-end setup of the local inference server from a fresh checkout, as deployed on
this NVIDIA GB10 (DGX Spark class). Serves Anthropic `/v1/messages` + OpenAI APIs on
`127.0.0.1:8000`; drive it with Claude Code via `claude-local.sh`.

> Files that ship with the repo: `download_model.sh`, the `Makefile`. Files that are
> **local helpers** (not in git — recreate them from the snippets below): `start-server.sh`,
> `claude-local.sh`, and the systemd unit `~/.config/systemd/user/ds4.service`.

---

## 0. Prerequisites

**Hardware:** NVIDIA GB10 Grace-Blackwell (DGX Spark class) — ARM64, single Blackwell GPU
(sm_121), ~119 GB unified memory. The default model needs ~81 GB resident, so ≥96 GB RAM.

**Disk:** ≥90 GB free for the model GGUF (81 GB) plus build artifacts.

**Software:**
- CUDA toolkit 13 (`nvcc` on `PATH`, typically `/usr/local/cuda/bin`).
- A C toolchain + `make` (`sudo apt install build-essential`).
- `curl` (used by `download_model.sh` for this model).
- [Claude Code CLI](https://claude.com/claude-code) (`claude`) for the client side.
- A Hugging Face account/token is **not** required for the public `antirez/deepseek-v4-gguf`
  repo, but if you hit rate limits, set `HF_TOKEN` or pass `--token`.

Verify CUDA is visible:

```bash
nvcc --version          # expect CUDA 13.x
nvidia-smi              # expect the GB10 / Blackwell GPU
```

---

## 1. Clone

```bash
git clone https://github.com/antirez/ds4.git ~/deepseekflash/ds4
cd ~/deepseekflash/ds4
```

(This deployment lives at `/home/ylow/deepseekflash/ds4`. If you clone elsewhere, adjust the
absolute paths in the systemd unit and helper scripts below.)

> **Fork note:** this box runs a fork (`ylow/ds4`) carrying **local commits on top of
> upstream `antirez/ds4`** that shrink the GPU KV cache — see
> [Local optimizations](#local-optimizations-kv-cache-local-commits-on-top-of-upstream). A
> clean upstream clone builds and runs identically, just without those optimizations.

---

## 2. Build for the GB10

`make cuda-spark` builds every binary (`ds4`, `ds4-server`, `ds4-bench`, `ds4-eval`,
`ds4-agent`) for the Spark/Blackwell target:

```bash
make cuda-spark
```

- Takes a few minutes (nvcc compiles `ds4_cuda.cu`).
- Do **not** run `make cpu` afterwards — it overwrites the CUDA binaries with CPU ones.
- For a faster relink during iteration on `ds4.c` only: `make ds4 ds4-server CUDA_ARCH=`.

---

## 3. Download the model (~81 GB)

```bash
./download_model.sh q2-imatrix
```

This curls `DeepSeek-V4-Flash-IQ2XXS-...-imatrix.gguf` (2-bit routed experts, ~81 GB) into
`./gguf/` and symlinks it to `./ds4flash.gguf` (the default model path). It resumes if
interrupted — just re-run. Other targets (`q4-imatrix`, `pro-*`, `mtp`) are listed by
`./download_model.sh --help`; `q2-imatrix` is the right one for a 96–128 GB machine.

Confirm:

```bash
ls -lh ds4flash.gguf        # -> gguf/DeepSeek-V4-Flash-IQ2XXS-...-imatrix.gguf
```

---

## 4. Smoke test

```bash
# one-shot generation (loads ~81 GB, ~10s, then generates)
./ds4 --cuda -p "Hello, who are you?"

# correctness gate (bit-deterministic teacher-forced perplexity)
./ds4 --cuda --perplexity-file doors-of-stone-chapter-1.md -n 256 -c 4096
```

Expect ~13–15 tok/s decode and ~360 tok/s prefill.

---

## 5. Run the server

### Option A — foreground (quick)

Create `start-server.sh` (chmod +x) and run it:

```bash
#!/usr/bin/env bash
# Start the DeepSeek-V4-Flash (ds4 / DwarfStar) inference server on the GB10.
# Serves Anthropic /v1/messages + OpenAI /v1/chat/completions on 127.0.0.1:8000.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p /tmp/ds4-kv
exec ./ds4-server \
  --cuda \
  -m ds4flash.gguf \
  --ctx 131072 \
  --kv-disk-dir /tmp/ds4-kv \
  --host 127.0.0.1 \
  --port 8000 \
  "$@"
```

```bash
./start-server.sh
```

### Option B — systemd user service (managed, recommended)

The repo ships the unit as `ds4.service`. Copy it into your user unit directory (edit the
absolute paths inside if your checkout isn't at `/home/ylow/deepseekflash/ds4`):

```bash
mkdir -p ~/.config/systemd/user
cp ds4.service ~/.config/systemd/user/ds4.service
```

For reference, the unit is:

```ini
[Unit]
Description=DeepSeek-V4-Flash inference server (ds4 / DwarfStar) on GB10
Documentation=https://github.com/antirez/ds4
After=network-online.target
StartLimitIntervalSec=60
StartLimitBurst=3

[Service]
Type=simple
WorkingDirectory=/home/ylow/deepseekflash/ds4
ExecStartPre=/bin/mkdir -p /tmp/ds4-kv
ExecStart=/home/ylow/deepseekflash/ds4/ds4-server \
    --cuda \
    -m /home/ylow/deepseekflash/ds4/ds4flash.gguf \
    --ctx 131072 \
    --kv-disk-dir /tmp/ds4-kv \
    --host 127.0.0.1 \
    --port 8000
Restart=on-failure
RestartSec=5
TimeoutStartSec=300
TimeoutStopSec=30

# NOTE: intentionally NO [Install] section, so the service is "static" and can never be
# 'enabled' / autostarted on boot. Start it manually.
```

Manage it:

```bash
systemctl --user daemon-reload
systemctl --user start ds4          # start (first start loads ~81 GB, up to ~1 min)
systemctl --user status ds4         # health
journalctl --user -u ds4 -f         # follow logs
systemctl --user stop ds4           # stop
```

It is **static** (no `[Install]` section) → it never autostarts on boot; start it by hand.
To let user services run without an active login session: `loginctl enable-linger $USER`.

> Only **one** ds4 process may hold the GPU at a time (instance lock). Don't run
> `start-server.sh` and the systemd service together. To kill a stray process, use its exact
> PID — never `pkill -f ds4-server` (it matches your own shell).

---

## 6. Point Claude Code at the local server

Create `claude-local.sh` (chmod +x):

```bash
#!/usr/bin/env bash
# Launch Claude Code against the local DeepSeek-V4-Flash (ds4 / DwarfStar) server.
# Usage: ./claude-local.sh [claude args...]
set -euo pipefail

export ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
export ANTHROPIC_AUTH_TOKEN="ds4-local"          # server ignores auth; any value works
export ANTHROPIC_MODEL="deepseek-v4-flash"
export ANTHROPIC_SMALL_FAST_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export DISABLE_TELEMETRY=1
export DISABLE_ERROR_REPORTING=1
export DISABLE_AUTOUPDATER=1
export DISABLE_COST_WARNINGS=1                    # count_tokens isn't implemented (cosmetic)

exec claude "$@"
```

```bash
./claude-local.sh                 # interactive Claude Code on the local model
./claude-local.sh -p "say hi"     # one-shot
```

---

## 7. Verify the API directly

```bash
curl -s http://127.0.0.1:8000/v1/messages \
  -H 'content-type: application/json' \
  -d '{"model":"deepseek-v4-flash","max_tokens":64,
       "messages":[{"role":"user","content":"Reply with just: ok"}]}'
```

Endpoints served: `/v1/messages` (Anthropic), `/v1/chat/completions`, `/v1/responses`,
`/v1/completions` (OpenAI-compatible). The model name is ignored — `deepseek-v4-flash` and
`deepseek-v4-pro` both serve the loaded GGUF.

---

## Local optimizations (KV-cache; local commits on top of upstream)

This deployment carries local commits (on top of upstream `antirez/ds4`) that shrink the
**live GPU KV cache** so the single GB10 can hold much longer context. They are CUDA-only,
gated behind compile-time macros in `ds4.c`, individually revertible, and each was validated
against the bit-deterministic teacher-forced perplexity oracle
(`./ds4 --cuda --perplexity-file doors-of-stone-chapter-1.md -n 256 -c 4096`). A clean
upstream clone does not include them.

**Why:** decode reads ~9.5 GB/token, dominated by the attention projections (~61%); the live
attention/indexer KV caches were F32 (~3 GB at 128K, scaling toward 15–20 GB at the 600–800K
context goal). The work narrows the dominant term — the 512-dim compressed-attention rows
(448 NoPE + 64 RoPE) — using **expand-on-read** (stored rows are dequantized into reused F32
scratch on read), so the trusted F32 attention kernels stay untouched.

| Compressed-attn KV storage | B/row | vs F32 | quality (teacher-forced nll @4096) |
|---|---|---|---|
| F32 (upstream)            | 2048  | 1.0×  | 317.843968 (reference) |
| F16                       | 1024  | 2.0×  | 317.842979 (~lossless) |
| **FP8-split** — E4M3 NoPE + per-64 int8 exp + F16 RoPE | 584 | 3.5× | 317.842979 (**bit-identical to F16**) |
| **Hadamard-FP4 / NF4** (current) | 360 | **5.7×** | 319.009 (**+0.367%**, within the 0.5% gate) |

Enabled on CUDA (macros; FP4 takes precedence over FP8 over F16):
- `DS4_GPU_ATTN_COMP_CACHE_FP4` — Hadamard-64 rotation + **NF4** (NormalFloat4) 4-bit codec on
  the 448 NoPE dims, per-64 power-of-2 scale; RoPE-64 tail F16. The rotation makes the per-dim
  distribution ~Gaussian so NF4's quantile levels fit it (plain FP4/E2M1 was ~6× worse:
  +2.1% vs +0.37%). Bit-exactly self-checkable with `DS4_FP4_SELFCHECK=1`.
- `DS4_GPU_INDEX_COMP_CACHE_F16` — the 128-dim indexer KV cache stored F16 (bit-identical; the
  indexer scores all visible rows, so this also halves that read bandwidth).

Net effect at ctx 4096: context buffers 405.5 (F16) → 396.2 (FP8) → **391.5 MiB** (FP4); the
saving is linear in context, reaching multiple GB toward 600–800K tokens — the actual goal.

Also implemented but **off by default**: CUDA Graphs capture/replay of the decode tape
(`DS4_CUDA_GRAPH=1`) — correct and bit-exact, but neutral here because single-token decode is
GPU-bound (launch overhead is already hidden); kept opt-in for when per-token GPU time drops
enough to expose it.

Design specs, implementation plans, and the full validation history live under
`docs/superpowers/specs/` and `docs/superpowers/plans/`.

## Notes & gotchas

- **No auth, localhost-only.** Bound to `127.0.0.1:8000`. Don't expose it without adding a
  proxy/auth.
- **Disk KV cache.** `--kv-disk-dir /tmp/ds4-kv` persists prompt prefixes. A cold start
  prefills Claude Code's ~20K-token system prompt in ~55 s; warm turns reuse the cache and
  are ~4 s. Wipe with `rm -rf /tmp/ds4-kv` if it gets stale.
- **Context.** Served at `--ctx 131072`. Larger contexts cost more KV memory.
- **`count_tokens` returns 404** — unimplemented, cosmetic only (the client gauge just can't
  show token counts).
- **Quality.** This is a 2-bit (IQ2_XXS) quant — some quality loss vs cloud models.
- **Performance.** ~13–15 tok/s decode, ~360 tok/s prefill on this GB10.
- **Optional speculative decoding (MTP).** `./download_model.sh mtp`, then pass
  `--mtp gguf/DeepSeek-V4-Flash-MTP-...gguf --mtp-draft 2` to `ds4`/`ds4-server`.
