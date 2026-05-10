#!/usr/bin/env bash
# Start all backend servers required by Lingofluent.
#
# Servers started:
#   1. llama.cpp  — LLM server via Docker  (host port 8000)
#   2. CrispASR   — speech-to-text         (port 8080)
#   3. CrispASR   — text-to-speech         (port 8081)
#
# Usage:
#   bash scripts/start_servers.sh          # start all
#   bash scripts/start_servers.sh stop     # stop all
#   bash scripts/start_servers.sh status   # show running/stopped
#
# Customise via environment variables:
#   LLM_MODEL_DIR    — Gemma GGUF directory      (default: ~/llama/models/gemma-4-E2B-it)
#   CRISPASR_DIR     — CrispASR repo root         (default: ~/CrispASR)
#   LLM_PORT         — host port for llama.cpp    (default: 8000)
#   ASR_PORT         — CrispASR ASR port          (default: 8080)
#   TTS_PORT         — CrispASR TTS port          (default: 8081)
#   CONTEXT_SIZE     — llama.cpp context window   (default: 4096)
#   GPU_LAYERS       — layers offloaded to GPU    (default: 99)

set -uo pipefail

# Resolve project root from the script's own location — works regardless of
# where the script is called from.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

LLM_MODEL_DIR="${LLM_MODEL_DIR:-$HOME/llama/models/gemma-4-E2B-it}"
CRISPASR_DIR="${CRISPASR_DIR:-$HOME/CrispASR}"
LLM_PORT="${LLM_PORT:-8000}"
ASR_PORT="${ASR_PORT:-8080}"
TTS_PORT="${TTS_PORT:-8081}"
CONTEXT_SIZE="${CONTEXT_SIZE:-4096}"
GPU_LAYERS="${GPU_LAYERS:-99}"

CRISPASR_BIN="$CRISPASR_DIR/build/bin/crispasr"
ASR_MODEL="$CRISPASR_DIR/parakeet-tdt-0.6b-v3-q8_0.gguf"
LLM_MODEL="$LLM_MODEL_DIR/gemma-4-E2B-it-Q5_K_S.gguf"
MMPROJ="$LLM_MODEL_DIR/mmproj-F16.gguf"
VOICE_DIR="$PROJECT_DIR/voices"

LOG_DIR="$PROJECT_DIR/logs"
LLM_LOG="$LOG_DIR/llm_server.log"
ASR_LOG="$LOG_DIR/asr_server.log"
TTS_LOG="$LOG_DIR/tts_server.log"
PID_DIR="$PROJECT_DIR/.pids"

# ── helpers ───────────────────────────────────────────────────────────────────

info()    { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()      { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()    { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
die()     { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; exit 1; }
section() { echo -e "\n\033[1;37m── $* ──\033[0m"; }

require() {
    command -v "$1" &>/dev/null || die "'$1' not found. $2"
}

is_port_open() {
    # returns 0 if something is already listening on the port
    ss -tlnp 2>/dev/null | grep -q ":$1 " || \
    nc -z 127.0.0.1 "$1" 2>/dev/null
}

save_pid() { echo "$1" > "$PID_DIR/$2.pid"; }
load_pid() { cat "$PID_DIR/$1.pid" 2>/dev/null || echo ""; }

# ── stop ─────────────────────────────────────────────────────────────────────

do_stop() {
    section "Stopping servers"

    # llama.cpp Docker container
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^lingofluent-llm$"; then
        info "Stopping llama.cpp container..."
        docker stop lingofluent-llm &>/dev/null && ok "llama.cpp stopped"
    else
        warn "llama.cpp container not running"
    fi

    for name in asr tts; do
        pid=$(load_pid "$name")
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            info "Stopping CrispASR $name (PID $pid)..."
            kill "$pid" && ok "CrispASR $name stopped"
        else
            warn "CrispASR $name not running"
        fi
        rm -f "$PID_DIR/$name.pid"
    done
}

# ── status ────────────────────────────────────────────────────────────────────

do_status() {
    section "Server status"
    local llm_ok=false asr_ok=false tts_ok=false

    is_port_open "$LLM_PORT" && llm_ok=true
    is_port_open "$ASR_PORT" && asr_ok=true
    is_port_open "$TTS_PORT" && tts_ok=true

    printf "  llama.cpp  (:%s)  %s\n" "$LLM_PORT" "$($llm_ok && echo '✓ running' || echo '✗ stopped')"
    printf "  ASR        (:%s)  %s\n" "$ASR_PORT"  "$($asr_ok && echo '✓ running' || echo '✗ stopped')"
    printf "  TTS        (:%s)  %s\n" "$TTS_PORT"  "$($tts_ok && echo '✓ running' || echo '✗ stopped')"
}

# ── start ─────────────────────────────────────────────────────────────────────

do_start() {
    section "Pre-flight checks"
    require docker  "Install Docker and the NVIDIA Container Toolkit for GPU support."

    [[ -f "$LLM_MODEL" ]]   || die "LLM model not found: $LLM_MODEL\n  Run: bash scripts/download_models.sh"
    [[ -f "$MMPROJ" ]]      || die "Vision projector not found: $MMPROJ\n  Run: bash scripts/download_models.sh"
    [[ -f "$CRISPASR_BIN" ]] || die "CrispASR binary not found: $CRISPASR_BIN\n  Build CrispASR first (see README)."
    [[ -f "$ASR_MODEL" ]]   || die "ASR model not found: $ASR_MODEL\n  Run: bash scripts/download_models.sh"

    mkdir -p "$LOG_DIR" "$PID_DIR"
    ok "All prerequisites found"

    # ── 1. llama.cpp ─────────────────────────────────────────────────────────
    section "llama.cpp LLM server (port $LLM_PORT)"

    if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^lingofluent-llm$"; then
        ok "Already running"
    elif is_port_open "$LLM_PORT"; then
        warn "Port $LLM_PORT is occupied by another process — skipping"
    else
        # Remove stopped container with the same name if it exists
        docker rm lingofluent-llm 2>/dev/null || true
        info "Starting llama.cpp container..."
        if docker run -d \
            --name lingofluent-llm \
            --gpus all \
            -v "$LLM_MODEL_DIR:/models" \
            -p "${LLM_PORT}:8080" \
            ghcr.io/ggml-org/llama.cpp:server-cuda \
            -m /models/gemma-4-E2B-it-Q5_K_S.gguf \
            --mmproj /models/mmproj-F16.gguf \
            --host 0.0.0.0 \
            --port 8080 \
            --n-gpu-layers "$GPU_LAYERS" \
            -c "$CONTEXT_SIZE" \
            >> "$LLM_LOG" 2>&1; then
            ok "Started (logs: $LLM_LOG)"
        else
            warn "Failed to start llama.cpp — check $LLM_LOG"
        fi
    fi

    # ── 2. CrispASR ASR ──────────────────────────────────────────────────────
    section "CrispASR ASR server (port $ASR_PORT)"

    if is_port_open "$ASR_PORT"; then
        ok "Already running on port $ASR_PORT"
    else
        info "Starting ASR server..."
        info "  Command: $CRISPASR_BIN --server -m $ASR_MODEL --port $ASR_PORT"
        "$CRISPASR_BIN" --server \
            -m "$ASR_MODEL" \
            --port "$ASR_PORT" \
            >> "$ASR_LOG" 2>&1 &
        ASR_PID=$!
        save_pid "$ASR_PID" "asr"
        ok "Started (PID $ASR_PID, logs: $ASR_LOG)"
    fi

    # ── 3. CrispASR TTS ──────────────────────────────────────────────────────
    section "CrispASR TTS server (port $TTS_PORT)"

    if is_port_open "$TTS_PORT"; then
        ok "Already running on port $TTS_PORT"
    else
        info "Starting TTS server (first run downloads Qwen3-TTS weights)..."
        info "  Command: $CRISPASR_BIN --server --backend qwen3-tts -m auto --voice-dir $VOICE_DIR --port $TTS_PORT"
        "$CRISPASR_BIN" --server \
            --backend qwen3-tts -m auto \
            --voice-dir "$VOICE_DIR" \
            --port "$TTS_PORT" \
            >> "$TTS_LOG" 2>&1 &
        save_pid $! "tts"
        ok "Started (PID $!, logs: $TTS_LOG)"
    fi

    # ── summary ──────────────────────────────────────────────────────────────
    section "Done"
    echo ""
    echo "  LLM_BASE_URL=http://localhost:$LLM_PORT"
    echo "  ASR_BASE_URL=http://localhost:$ASR_PORT"
    echo "  TTS_BASE_URL=http://localhost:$TTS_PORT"
    echo ""
    echo "Check logs in $LOG_DIR/"
    echo "To stop all servers: bash scripts/start_servers.sh stop"
    echo ""
    echo "Start the bot:"
    echo "  python -m lingofluent"
}

# ── dispatch ──────────────────────────────────────────────────────────────────

case "${1:-start}" in
    start)  do_start  ;;
    stop)   do_stop   ;;
    status) do_status ;;
    *)      die "Unknown command: $1. Usage: $0 [start|stop|status]" ;;
esac
