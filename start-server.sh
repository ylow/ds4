#!/usr/bin/env bash
# Start the DeepSeek-V4-Flash (ds4 / DwarfStar) inference server on the GB10.
# Serves Anthropic /v1/messages + OpenAI /v1/chat/completions on 127.0.0.1:8000.
set -euo pipefail
cd "$(dirname "$0")"

mkdir -p /tmp/ds4-kv

# Model: low-ALU Q4_1 decode-optimized build (q_a/q_b/kv + output head -> Q4_1,
# attention output-proj -> Q4_K, shared expert + routed experts as-is). ~18 t/s
# decode on the GB10 (vs ~15.5 stock); teacher-forced nll 319.685 (~Q8 quality).
# The ds4flash.gguf symlink also points here, so override -m to switch models.
MODEL="${DS4_MODEL:-gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AQkvQ41-AOutQ4K-SExpQ8-OutHeadQ41-chat-v2-imatrix.gguf}"

exec ./ds4-server \
  --cuda \
  -m "$MODEL" \
  --ctx 131072 \
  --kv-disk-dir /tmp/ds4-kv \
  --host 127.0.0.1 \
  --port 8000 \
  "$@"
