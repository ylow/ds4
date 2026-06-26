#!/usr/bin/env bash
# Launch Claude Code against the local DeepSeek-V4-Flash (ds4 / DwarfStar) server.
# Usage: ./claude-local.sh [claude args...]
#   ./claude-local.sh                 # interactive session
#   ./claude-local.sh -p "say hi"     # one-shot print mode
set -euo pipefail

# --- point Claude Code at the local ds4 server (Anthropic /v1/messages) ---
export ANTHROPIC_BASE_URL="http://127.0.0.1:8000"
export ANTHROPIC_AUTH_TOKEN="ds4-local"          # server ignores auth; any value works

# Force the model name the server serves (it ignores the name, but be explicit).
export ANTHROPIC_MODEL="deepseek-v4-flash"
export ANTHROPIC_SMALL_FAST_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_OPUS_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_SONNET_MODEL="deepseek-v4-flash"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="deepseek-v4-flash"

# Don't leak telemetry/background pings to Anthropic, and don't auto-update.
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1
export DISABLE_TELEMETRY=1
export DISABLE_ERROR_REPORTING=1
export DISABLE_AUTOUPDATER=1

# count_tokens isn't implemented by ds4; this keeps the context gauge from erroring.
export DISABLE_COST_WARNINGS=1

exec claude "$@"
