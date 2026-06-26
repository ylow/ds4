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
