#!/bin/bash
# Centralized hardware tier detection script
# UPDATED: Now enforces HIGH-TIER ONLY requirements for installation
# Standard tier installations are no longer supported due to reliability issues

# HIGH-TIER MINIMUM REQUIREMENTS:
# - GPU: 16GB VRAM (NVIDIA with CUDA support)
# - RAM: 32GB
# - Disk: 40GB free space
# - CPU: 8+ cores

# Auto-detect hardware performance tier (high-tier only)
detect_performance_tier() {
  local gpu_mem=0
  local tier="insufficient"
  
  # Try nvidia-smi first (NVIDIA GPUs)
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')
    
    # HIGH-TIER REQUIREMENT: 16GB+ VRAM
    if [ -n "$gpu_mem" ] && [ "$gpu_mem" -ge 16000 ]; then
      tier="high"
    else
      tier="insufficient"
    fi
  else
    # Try rocm-smi for AMD GPUs
    if command -v rocm-smi >/dev/null 2>&1; then
      gpu_mem=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | grep -oP '\d+' | head -n1)
      # HIGH-TIER REQUIREMENT: 16GB+ VRAM
      if [ -n "$gpu_mem" ] && [ "$gpu_mem" -ge 16000 ]; then
        tier="high"
      else
        tier="insufficient"
      fi
    else
      # No GPU detected
      tier="insufficient"
    fi
  fi
  
  echo "$tier"
  
  # Export GPU memory for detailed error messages
  export DETECTED_GPU_VRAM="$gpu_mem"
}

# Set tier-based configuration values
set_tier_values() {
  local tier="$1"
  
  if [ "$tier" = "high" ]; then
    # High-performance tier (16GB+ GPU) - ONLY SUPPORTED CONFIGURATION
    export TIER_REFLECTION_NUM_PREDICT=480
    export TIER_REFLECTION_MAX_TOKENS=480
    export TIER_REFLECTION_WORD_MAX=180
    export TIER_REFLECTION_WORD_MIN=80
    export TIER_ANALYTICAL_NUM_PREDICT=1536
    export TIER_CHAT_NUM_PREDICT=2048
    export TIER_CHAT_NUM_CTX=8192
  else
    # Insufficient hardware - installation will abort
    echo "ERROR: Hardware does not meet high-tier requirements" >&2
    return 1
  fi
  
  export PERFORMANCE_TIER="$tier"
}

# Cache tier detection result if requested
cache_tier() {
  local cache_file="${1:-/tmp/selo-tier-cache}"
  echo "$PERFORMANCE_TIER" > "$cache_file"
  # Also cache the values
  {
    echo "PERFORMANCE_TIER=$PERFORMANCE_TIER"
    echo "TIER_REFLECTION_NUM_PREDICT=$TIER_REFLECTION_NUM_PREDICT"
    echo "TIER_REFLECTION_MAX_TOKENS=$TIER_REFLECTION_MAX_TOKENS"
    echo "TIER_REFLECTION_WORD_MAX=$TIER_REFLECTION_WORD_MAX"
    echo "TIER_REFLECTION_WORD_MIN=$TIER_REFLECTION_WORD_MIN"
    echo "TIER_ANALYTICAL_NUM_PREDICT=$TIER_ANALYTICAL_NUM_PREDICT"
    echo "TIER_CHAT_NUM_PREDICT=$TIER_CHAT_NUM_PREDICT"
    echo "TIER_CHAT_NUM_CTX=$TIER_CHAT_NUM_CTX"
  } > "${cache_file}.env"
}

# Load cached tier if available and recent (within 1 hour)
load_cached_tier() {
  local cache_file="${1:-/tmp/selo-tier-cache}"
  
  if [ -f "${cache_file}.env" ]; then
    local age=$(( $(date +%s) - $(stat -c %Y "${cache_file}.env" 2>/dev/null || echo 0) ))
    if [ $age -lt 300 ]; then
      # Cache is less than 5 minutes old (reduced from 1 hour for installation accuracy)
      source "${cache_file}.env"
      return 0
    fi
  fi
  return 1
}

# Main execution if run directly
if [ "${BASH_SOURCE[0]}" = "${0}" ]; then
  # Check for cached value first
  if ! load_cached_tier; then
    # No valid cache, detect tier
    PERFORMANCE_TIER=$(detect_performance_tier)
    if ! set_tier_values "$PERFORMANCE_TIER"; then
      echo "Hardware requirements not met. Exiting." >&2
      exit 1
    fi
    cache_tier
  fi
  
  # Output tier information
  if [ -t 1 ]; then
    # Interactive terminal - show formatted output
    if [ "$PERFORMANCE_TIER" = "high" ]; then
      echo "✅ Performance Tier: $PERFORMANCE_TIER (REQUIREMENTS MET)"
      echo "Configuration:"
      echo "  TIER_REFLECTION_NUM_PREDICT=$TIER_REFLECTION_NUM_PREDICT"
      echo "  TIER_REFLECTION_MAX_TOKENS=$TIER_REFLECTION_MAX_TOKENS"
      echo "  TIER_REFLECTION_WORD_MAX=$TIER_REFLECTION_WORD_MAX"
      echo "  TIER_REFLECTION_WORD_MIN=$TIER_REFLECTION_WORD_MIN"
      echo "  TIER_ANALYTICAL_NUM_PREDICT=$TIER_ANALYTICAL_NUM_PREDICT"
      echo "  TIER_CHAT_NUM_PREDICT=$TIER_CHAT_NUM_PREDICT"
      echo "  TIER_CHAT_NUM_CTX=$TIER_CHAT_NUM_CTX"
    else
      echo "❌ Performance Tier: $PERFORMANCE_TIER (INSUFFICIENT)"
    fi
  else
    # Non-interactive - just output tier
    echo "$PERFORMANCE_TIER"
  fi
else
  # Being sourced - just load/detect without output
  if ! load_cached_tier; then
    PERFORMANCE_TIER=$(detect_performance_tier)
    if ! set_tier_values "$PERFORMANCE_TIER"; then
      return 1
    fi
    cache_tier
  fi
fi
