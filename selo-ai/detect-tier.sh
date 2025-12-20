#!/bin/bash
# Centralized hardware tier detection script
# This ensures consistent tier detection across all scripts

# Auto-detect hardware performance tier (2-tier system)
detect_performance_tier() {
  local gpu_mem=0
  local tier="standard"
  
  # Try nvidia-smi first (NVIDIA GPUs)
  if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_mem=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')
    
    if [ -n "$gpu_mem" ] && [ "$gpu_mem" -ge 12000 ]; then
      tier="high"
    else
      tier="standard"
    fi
  else
    # Try rocm-smi for AMD GPUs
    if command -v rocm-smi >/dev/null 2>&1; then
      gpu_mem=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | grep -oP '\d+' | head -n1)
      if [ -n "$gpu_mem" ] && [ "$gpu_mem" -ge 12000 ]; then
        tier="high"
      else
        tier="standard"
      fi
    else
      tier="standard"
    fi
  fi
  
  echo "$tier"
}

# Set tier-based configuration values
set_tier_values() {
  local tier="$1"
  
  if [ "$tier" = "high" ]; then
    # High-performance tier (12GB+ GPU)
    export TIER_REFLECTION_NUM_PREDICT=650
    export TIER_REFLECTION_MAX_TOKENS=650
    export TIER_REFLECTION_WORD_MAX=250
    export TIER_REFLECTION_WORD_MIN=80
    export TIER_ANALYTICAL_NUM_PREDICT=1536
    export TIER_CHAT_NUM_PREDICT=2048
    export TIER_CHAT_NUM_CTX=8192
  else
    # Standard tier (8GB GPU or unknown)
    export TIER_REFLECTION_NUM_PREDICT=640
    export TIER_REFLECTION_MAX_TOKENS=640
    export TIER_REFLECTION_WORD_MAX=250
    export TIER_REFLECTION_WORD_MIN=80
    export TIER_ANALYTICAL_NUM_PREDICT=640
    export TIER_CHAT_NUM_PREDICT=1024
    export TIER_CHAT_NUM_CTX=8192
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
    set_tier_values "$PERFORMANCE_TIER"
    cache_tier
  fi
  
  # Output tier information
  if [ -t 1 ]; then
    # Interactive terminal - show formatted output
    echo "Performance Tier: $PERFORMANCE_TIER"
    echo "Configuration:"
    echo "  TIER_REFLECTION_NUM_PREDICT=$TIER_REFLECTION_NUM_PREDICT"
    echo "  TIER_REFLECTION_MAX_TOKENS=$TIER_REFLECTION_MAX_TOKENS"
    echo "  TIER_REFLECTION_WORD_MAX=$TIER_REFLECTION_WORD_MAX"
    echo "  TIER_REFLECTION_WORD_MIN=$TIER_REFLECTION_WORD_MIN"
    echo "  TIER_ANALYTICAL_NUM_PREDICT=$TIER_ANALYTICAL_NUM_PREDICT"
    echo "  TIER_CHAT_NUM_PREDICT=$TIER_CHAT_NUM_PREDICT"
    echo "  TIER_CHAT_NUM_CTX=$TIER_CHAT_NUM_CTX"
  else
    # Non-interactive - just output tier
    echo "$PERFORMANCE_TIER"
  fi
else
  # Being sourced - just load/detect without output
  if ! load_cached_tier; then
    PERFORMANCE_TIER=$(detect_performance_tier)
    set_tier_values "$PERFORMANCE_TIER"
    cache_tier
  fi
fi
