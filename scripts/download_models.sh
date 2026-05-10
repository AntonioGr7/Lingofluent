#!/usr/bin/env bash
# Download all model weights required by Lingofluent.
#
# Usage:
#   bash scripts/download_models.sh
#
# Customise paths via environment variables before running:
#   LLM_MODEL_DIR   — where to save the Gemma GGUF files  (default: ~/llama/models/gemma-4-E2B-it)
#   ASR_MODEL_DIR   — where to save the CrispASR ASR model (default: ~/CrispASR)

set -euo pipefail

LLM_MODEL_DIR="${LLM_MODEL_DIR:-$HOME/llama/models/gemma-4-E2B-it}"
ASR_MODEL_DIR="${ASR_MODEL_DIR:-$HOME/CrispASR}"

# ── helpers ──────────────────────────────────────────────────────────────────

info()  { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m    $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
die()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; exit 1; }

require() {
    command -v "$1" &>/dev/null || die "'$1' not found. $2"
}

# ── prerequisites ─────────────────────────────────────────────────────────────

require hf "Install with: pip install huggingface_hub"

info "Checking Hugging Face authentication..."
if ! hf auth whoami &>/dev/null; then
    warn "Not logged in. Run: hf auth login"
    hf auth login
fi
ok "Authenticated as: $(hf auth whoami)"

# ── LLM: Gemma 4B (llama.cpp) ────────────────────────────────────────────────

info "Downloading Gemma 4B weights → $LLM_MODEL_DIR"
mkdir -p "$LLM_MODEL_DIR"

info "  Quantised model (Q5_K_S)..."
hf download unsloth/gemma-4-E2B-it-GGUF \
    --include "gemma-4-E2B-it-Q5_K_S.gguf" \
    --local-dir "$LLM_MODEL_DIR"

info "  Vision projector (mmproj)..."
hf download unsloth/gemma-4-E2B-it-GGUF \
    --include "mmproj-F16.gguf" \
    --local-dir "$LLM_MODEL_DIR"

ok "Gemma model ready at $LLM_MODEL_DIR"

# ── ASR: Parakeet (CrispASR) ─────────────────────────────────────────────────

info "Downloading Parakeet ASR model → $ASR_MODEL_DIR"
mkdir -p "$ASR_MODEL_DIR"

hf download cstr/parakeet-tdt-0.6b-v3-GGUF \
    --include "parakeet-tdt-0.6b-v3-q8_0.gguf" \
    --local-dir "$ASR_MODEL_DIR"

ok "Parakeet model ready at $ASR_MODEL_DIR"

# ── TTS: Qwen3-TTS ───────────────────────────────────────────────────────────

info "TTS (Qwen3-TTS) is downloaded automatically on first use by CrispASR."

# ── summary ──────────────────────────────────────────────────────────────────

echo ""
echo "All models downloaded. Next step:"
echo "  bash scripts/start_servers.sh"
