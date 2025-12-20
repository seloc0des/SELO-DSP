#!/bin/bash
set -euo pipefail

# SELO DSP Default Model Installation Script
# This script pulls all required models for the default configuration

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Determine Ollama binary location
OLLAMA_BIN="${OLLAMA_BIN:-$(command -v ollama || echo /usr/local/bin/ollama)}"

if [ ! -x "$OLLAMA_BIN" ]; then
    echo "Error: Ollama binary not found at $OLLAMA_BIN"
    exit 1
fi

# Check if Ollama service is running
if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    echo "Error: Ollama service is not responding at http://127.0.0.1:11434"
    echo "Please ensure Ollama is running before installing models."
    exit 1
fi

echo "========================================="
echo "  Installing Default SELO DSP Models"
echo "========================================="
echo ""

# Helper function to check if model exists
model_exists() {
    local model="$1"
    "$OLLAMA_BIN" show "$model" >/dev/null 2>&1
}

# Helper function to pull model with progress
pull_model() {
    local model="$1"
    local description="$2"
    
    echo "📦 $description"
    echo "   Model: $model"
    
    if model_exists "$model"; then
        echo "   ✓ Already installed, skipping"
        echo ""
        return 0
    fi
    
    echo "   Downloading..."
    if "$OLLAMA_BIN" pull "$model"; then
        echo "   ✓ Successfully installed"
        echo ""
        return 0
    else
        echo "   ✗ Failed to install"
        echo ""
        return 1
    fi
}

# Track failures
FAILED_MODELS=()

# 1. Conversational Model (Primary chat interface)
if ! pull_model "llama3:8b" "Conversational Model (LLaMA 3 8B)"; then
    FAILED_MODELS+=("llama3:8b")
fi

# 2. Analytical Model (Structured outputs, persona traits)
if ! pull_model "qwen2.5:3b" "Analytical Model (Qwen 2.5 3B)"; then
    FAILED_MODELS+=("qwen2.5:3b")
fi

# 3. Reflection Model (Inner monologue generation)
if ! pull_model "qwen2.5:3b" "Reflection Model (Qwen 2.5 3B)"; then
    FAILED_MODELS+=("qwen2.5:3b")
fi

# 4. Embedding Model (Vector search, semantic memory)
if ! pull_model "nomic-embed-text" "Embedding Model (Nomic Embed Text)"; then
    FAILED_MODELS+=("nomic-embed-text")
fi

echo "========================================="
echo "  Model Installation Summary"
echo "========================================="

if [ ${#FAILED_MODELS[@]} -eq 0 ]; then
    echo "✓ All models installed successfully!"
    echo ""
    echo "Installed models:"
    echo "  • llama3:8b          (Conversational)"
    echo "  • qwen2.5:3b         (Analytical)"
    echo "  • qwen2.5:3b         (Reflection)"
    echo "  • nomic-embed-text   (Embeddings)"
    echo ""
    exit 0
else
    echo "✗ Some models failed to install:"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  • $model"
    done
    echo ""
    echo "You can retry installation with:"
    echo "  bash $0"
    echo ""
    echo "Or manually install missing models with:"
    for model in "${FAILED_MODELS[@]}"; do
        echo "  ollama pull $model"
    done
    echo ""
    exit 1
fi
