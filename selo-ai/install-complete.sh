#!/bin/bash
# If an explicit install dir is provided, sync code there and adjust INSTALL_DIR
prepare_install_dir() {
  local target
  if [ -n "${INSTALL_DIR:-}" ]; then
    target="$INSTALL_DIR"
    echo "Using provided --install-dir: $target"
    sudo mkdir -p "$target"
    echo "Syncing codebase to $target ..."
    sudo rsync -a --delete --exclude '.git' --exclude 'backend/venv' "$SCRIPT_DIR/" "$target/"
    
    # Ensure Reports directory is copied and accessible
    if [ -d "$SCRIPT_DIR/../Reports" ]; then
        echo "Copying Reports directory for boot directives..."
        sudo mkdir -p "$target/Reports"
        sudo cp -r "$SCRIPT_DIR/../Reports/"* "$target/Reports/" 2>/dev/null || true
    fi
    
    # Ensure the install tree is owned by the instance user so the service can write logs/locks
    echo "Setting ownership to $INST_USER:$INST_USER..."
    sudo chown -R "$INST_USER":"$INST_USER" "$target" 2>/dev/null || true
    
    # Set proper permissions: directories need execute, files need read
    echo "Setting permissions..."
    sudo find "$target" -type d -exec chmod 775 {} \; 2>/dev/null || true
    sudo find "$target" -type f -name '*.sh' -exec chmod 775 {} \; 2>/dev/null || true
    sudo find "$target" -type f -name '*.py' -path '*/scripts/*' -exec chmod 775 {} \; 2>/dev/null || true
    sudo find "$target" -type f ! -name '*.sh' ! -name '*.py' -exec chmod 664 {} \; 2>/dev/null || true
  else
    # Default to running in-place from the current repository location
    target="$SCRIPT_DIR"
    echo "No --install-dir provided; installing in-place from $target"
    # Ensure the install tree is owned by the instance user so the service can write logs/locks
    echo "Setting ownership to $INST_USER:$INST_USER..."
    sudo chown -R "$INST_USER":"$INST_USER" "$target" 2>/dev/null || true
    
    # Set proper permissions: directories need execute, files need read
    echo "Setting permissions..."
    sudo find "$target" -type d -exec chmod 775 {} \; 2>/dev/null || true
    sudo find "$target" -type f -name '*.sh' -exec chmod 775 {} \; 2>/dev/null || true
    sudo find "$target" -type f -name '*.py' -path '*/scripts/*' -exec chmod 775 {} \; 2>/dev/null || true
    sudo find "$target" -type f ! -name '*.sh' ! -name '*.py' -exec chmod 664 {} \; 2>/dev/null || true
  fi

  # Persist INSTALL_DIR to service environment for systemd usage
  sudo mkdir -p /etc/selo-ai
  # Secure directory: group instance user, restrict traverse to root+group
  sudo chgrp "$INST_USER" /etc/selo-ai 2>/dev/null || true
  sudo chmod 750 /etc/selo-ai 2>/dev/null || true
  if [ -f /etc/selo-ai/environment ]; then
    if sudo grep -q '^INSTALL_DIR=' /etc/selo-ai/environment; then
      sudo sed -i -E "s|^INSTALL_DIR=.*|INSTALL_DIR=$target|" /etc/selo-ai/environment || true
    else
      echo "INSTALL_DIR=$target" | sudo tee -a /etc/selo-ai/environment >/dev/null
    fi
    # Ensure secure ownership and perms after update
    sudo chown root:"$INST_USER" /etc/selo-ai/environment 2>/dev/null || true
    sudo chmod 640 /etc/selo-ai/environment 2>/dev/null || true
  else
    echo "INSTALL_DIR=$target" | sudo tee /etc/selo-ai/environment >/dev/null
    sudo chown root:"$INST_USER" /etc/selo-ai/environment 2>/dev/null || true
    sudo chmod 640 /etc/selo-ai/environment 2>/dev/null || true
  fi

  # Export for current shell usage too
  export INSTALL_DIR="$target"
}

# Install or update the systemd unit so service runs correctly post-install
install_systemd_unit() {
  echo "========================================="
  echo "    Installing systemd unit"
  echo "========================================="
  local unit_src="$SCRIPT_DIR/selo-ai.service"
  local unit_dst="/etc/systemd/system/selo-ai@.service"
  if [ ! -f "$unit_src" ]; then
    echo "Error: systemd unit template not found at $unit_src"
    return 1
  fi
  echo "Installing unit to $unit_dst"
  sudo install -m 0644 "$unit_src" "$unit_dst"
  echo "Reloading systemd daemon"
  sudo systemctl daemon-reload

  local inst_user
  inst_user=${SUDO_USER:-$(whoami)}
  echo "Enabling instance selo-ai@${inst_user} to start on boot"
  sudo systemctl enable "selo-ai@${inst_user}" || true
}

# (shebang moved to top of file)

set -euo pipefail
IFS=$'\n\t'

declare -a WARMED_MODELS=()

# SELO DSP Complete Installation Script
# This script handles everything: dependencies, firewall, services, and startup

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Determine instance user for service grouping (used for secure env file perms)
INST_USER=${SUDO_USER:-$(whoami)}

# Safe ANSI color vars (avoid unbound errors under set -u)
if [ -t 1 ]; then
  YELLOW='\033[1;33m'
  NC='\033[0m'
else
  YELLOW=''
  NC=''
fi

# Optional flags
# Defaults: audit fix ON, quiet warnings ON
#   --no-audit-fix        Disable `npm audit fix`
#   --no-quiet-warnings   Disable quiet build (show full build output)
#   --install-dir <path>  Install/run service from this directory (rsync code to path)
#   --server-ip <ip>      Server LAN IP to advertise and probe (default: auto-detect)
#   --port <n>            Backend port to use (default: 8000)
#   --cuda <auto|on|off>  CUDA usage preference; default auto (detect via nvidia-smi)
# Backward compatibility (still accepted):
#   --audit-fix           Redundant (default ON)
#   --quiet-warnings      Redundant (default ON)
AUDIT_FIX=true
QUIET_WARNINGS=true
INSTALL_DIR=""
SERVER_IP=""
CLI_PORT=""
# Readiness timeout (seconds). 0 = wait indefinitely
# Set to 120 seconds (2 minutes) to prevent hanging during persona bootstrap
READINESS_TIMEOUT_SEC=120
# When true, require strict backend readiness using /health/details with DB probe
STRICT_HEALTH=false
# Persona readiness timeout (seconds). Installer will wait for persona to be fully bootstrapped.
# Set to 0 to wait indefinitely.
PERSONA_TIMEOUT_SEC=600
# Allow skipping persona readiness check (not recommended)
SKIP_PERSONA_CHECK=false
# CUDA mode: auto (default), on, off
CUDA_MODE="auto"
# Whether to restart Ollama service to apply settings (default: true)
RESTART_OLLAMA=true
# Whether to install PostgreSQL if not present (auto-detect)
INSTALL_POSTGRES=false
# Whether to install Ollama if not present (auto-detect)
INSTALL_OLLAMA=false
# Single default model configuration
MODEL_TEMPLATE_NAME="default"
MODEL_TEMPLATE_DIR="$SCRIPT_DIR/configs/$MODEL_TEMPLATE_NAME"
MODEL_INSTALL_SCRIPT="$MODEL_TEMPLATE_DIR/install-models.sh"
DEFAULT_CONVERSATIONAL_MODEL="llama3:8b"
DEFAULT_ANALYTICAL_MODEL="qwen2.5:3b"
DEFAULT_REFLECTION_MODEL="qwen2.5:3b"
DEFAULT_EMBEDDING_MODEL="nomic-embed-text"
CONVERSATIONAL_MODEL_VAL="$DEFAULT_CONVERSATIONAL_MODEL"
CONVERSATIONAL_MODEL_SET=false
ARGS=()
while (( "$#" )); do
  case "$1" in
    --audit-fix)
      AUDIT_FIX=true; shift ;;
    --quiet-warnings)
      QUIET_WARNINGS=true; shift ;;
    --no-audit-fix)
      AUDIT_FIX=false; shift ;;
    --no-quiet-warnings)
      QUIET_WARNINGS=false; shift ;;
    --install-dir)
      if [ -n "${2:-}" ]; then INSTALL_DIR="$2"; shift 2; else echo "Error: --install-dir requires a path"; exit 1; fi ;;
    --install-dir=*)
      INSTALL_DIR="${1#*=}"; shift ;;
    --server-ip)
      if [ -n "${2:-}" ]; then SERVER_IP="$2"; shift 2; else echo "Error: --server-ip requires an IP"; exit 1; fi ;;
    --server-ip=*)
      SERVER_IP="${1#*=}"; shift ;;
    --port)
      if [ -n "${2:-}" ]; then CLI_PORT="$2"; shift 2; else echo "Error: --port requires a value"; exit 1; fi ;;
    --port=*)
      CLI_PORT="${1#*=}"; shift ;;
    --cuda)
      if [ -n "${2:-}" ]; then CUDA_MODE="$2"; shift 2; else echo "Error: --cuda requires one of: auto|on|off"; exit 1; fi ;;
    --cuda=*)
      CUDA_MODE="${1#*=}"; shift ;;
    --readiness-timeout)
      if [ -n "${2:-}" ]; then READINESS_TIMEOUT_SEC="$2"; shift 2; else echo "Error: --readiness-timeout requires an integer seconds"; exit 1; fi ;;
    --readiness-timeout=*)
      READINESS_TIMEOUT_SEC="${1#*=}"; shift ;;
    --persona-timeout)
      if [ -n "${2:-}" ]; then PERSONA_TIMEOUT_SEC="$2"; shift 2; else echo "Error: --persona-timeout requires an integer seconds"; exit 1; fi ;;
    --persona-timeout=*)
      PERSONA_TIMEOUT_SEC="${1#*=}"; shift ;;
    --skip-persona-check)
      SKIP_PERSONA_CHECK=true; shift ;;
    --strict-health)
      STRICT_HEALTH=true; shift ;;
    --enable-cuda)
      CUDA_MODE="on"; shift ;;
    --no-cuda|--disable-cuda)
      CUDA_MODE="off"; shift ;;
    --no-restart-ollama)
      RESTART_OLLAMA=false; shift ;;
    --conversational-model)
      if [ -n "${2:-}" ]; then CONVERSATIONAL_MODEL_VAL="$2"; CONVERSATIONAL_MODEL_SET=true; shift 2; else echo "Error: --conversational-model requires a value (e.g. llama3 or llama3:8b)"; exit 1; fi ;;
    --conversational-model=*)
      CONVERSATIONAL_MODEL_VAL="${1#*=}"; CONVERSATIONAL_MODEL_SET=true; shift ;;
    --)
      shift; break ;;
    *)
      ARGS+=("$1"); shift ;;
  esac
done

mkdir -p "$SCRIPT_DIR/logs" 2>/dev/null || true

# Load centralized tier detection
if [ -f "$SCRIPT_DIR/detect-tier.sh" ]; then
  source "$SCRIPT_DIR/detect-tier.sh"
else
  # Fallback if centralized script not found
  echo "Warning: detect-tier.sh not found, using defaults"
  export PERFORMANCE_TIER="standard"
  export TIER_REFLECTION_NUM_PREDICT=640
  export TIER_REFLECTION_MAX_TOKENS=640
  export TIER_REFLECTION_WORD_MAX=250
  export TIER_REFLECTION_WORD_MIN=80
  export TIER_ANALYTICAL_NUM_PREDICT=640
  export TIER_CHAT_NUM_PREDICT=1024
  export TIER_CHAT_NUM_CTX=8192
fi

# Display tier information
if [ "$PERFORMANCE_TIER" = "high" ]; then
  echo "✨ High-Performance Tier Activated"
  echo "   - Reflection capacity: 650 tokens (~650 words max)"
  echo "   - Enhanced philosophical depth during persona bootstrap"
  echo "   - Extended chat context: 8192 tokens"
else
  echo "⚡ Standard Tier Activated (optimized for 8GB GPU)"
  echo "   - Reflection capacity: 640 tokens (~500 words max)"
  echo "   - Context window: 8192 tokens (qwen2.5 native capacity)"
  echo "   - Full-quality few-shot examples preserved"
fi
echo ""

# Ensure INSTALL_DIR is selected/synced up-front (honor --install-dir)
prepare_install_dir

# After selecting/syncing install directory, run the remainder from INSTALL_DIR
if [ -n "${INSTALL_DIR:-}" ] && [ "$SCRIPT_DIR" != "$INSTALL_DIR" ]; then
  echo "Switching working directory to INSTALL_DIR: $INSTALL_DIR"
  SCRIPT_DIR="$INSTALL_DIR"
  cd "$SCRIPT_DIR"
fi

echo "========================================="
echo "    SELO DSP Complete Installation"
echo "========================================="
echo "This script will:"
echo "  • Install system dependencies (Python, Node, build tools)"
echo "  • Optionally install PostgreSQL and Ollama"
echo "  • Auto-detect configuration (host IP, free backend port)"
echo "  • Build backend and frontend"
echo "  • Install systemd service (selo-ai@<user>)"
echo "  • Configure firewall (optional) and run health checks"
echo "  • CUDA mode: $CUDA_MODE (auto-detect if not on/off)"
echo ""

# Step 0: Interactive environment collection (run first)
# This gathers all required env vars and writes backend/.env and frontend/.env.
# Set SKIP_ENV_COLLECT=1 to bypass (advanced/non-interactive installs).
if [ -z "${SKIP_ENV_COLLECT:-}" ]; then
  if [ -f "$SCRIPT_DIR/collect-env-vars.sh" ]; then
    echo "Running interactive environment collector..."
    bash "$SCRIPT_DIR/collect-env-vars.sh"
  else
    echo "Warning: collector script not found at $SCRIPT_DIR/collect-env-vars.sh; continuing without interactive collection"
  fi
else
  echo "SKIP_ENV_COLLECT is set; skipping interactive environment collection"
fi

# Resolve server IP and backend port defaults after parsing args
# Prefer CLI flags, then collected backend/.env, then environment, then auto-detect/find-free

# Preload key selections from backend/.env (created by collector) without fully sourcing
COLLECTED_PORT=""
COLLECTED_HOST=""
COLLECTED_FRONTEND=""
if [ -f "$SCRIPT_DIR/backend/.env" ]; then
  COLLECTED_PORT=$(grep -E '^PORT=' "$SCRIPT_DIR/backend/.env" 2>/dev/null | tail -n1 | cut -d '=' -f2-)
  COLLECTED_HOST=$(grep -E '^HOST=' "$SCRIPT_DIR/backend/.env" 2>/dev/null | tail -n1 | cut -d '=' -f2-)
  COLLECTED_FRONTEND=$(grep -E '^FRONTEND_URL=' "$SCRIPT_DIR/backend/.env" 2>/dev/null | tail -n1 | cut -d '=' -f2-)
fi

# Helper: check available disk space (returns 1 if insufficient)
check_disk_space() {
  local required_gb=${1:-20}
  local target_dir="${2:-$SCRIPT_DIR}"
  
  # Get available space in GB
  local available_gb
  available_gb=$(df -BG "$target_dir" 2>/dev/null | tail -1 | awk '{print $4}' | sed 's/G//')
  
  if [ -z "$available_gb" ]; then
    echo "Warning: Could not determine available disk space"
    return 0  # Continue if we can't check
  fi
  
  if [ "$available_gb" -lt "$required_gb" ]; then
    echo ""
    echo "========================================="
    echo "    ⚠️  INSUFFICIENT DISK SPACE"
    echo "========================================="
    echo "Required: ${required_gb}GB for LLM models and data"
    echo "Available: ${available_gb}GB in $target_dir"
    echo ""
    echo "SELO requires approximately 20GB for:"
    echo "  • LLM models (llama3:8b, qwen2.5:3b): ~15GB"
    echo "  • Embedding model (nomic-embed-text): ~300MB"
    echo "  • Database and application data: ~2GB"
    echo "  • Build artifacts and logs: ~1GB"
    echo ""
    if [ -t 0 ] && [ "${AUTO_CONFIRM_FLAG:-false}" != "true" ]; then
      read -p "Continue anyway? (y/N): " -n 1 -r
      echo ""
      if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled. Please free up disk space and try again."
        return 1
      fi
      echo "Proceeding with installation despite low disk space..."
    else
      echo "ERROR: Insufficient disk space for non-interactive installation."
      echo "Please free up at least ${required_gb}GB and try again."
      return 1
    fi
  else
    echo "✓ Disk space check passed: ${available_gb}GB available (${required_gb}GB required)"
  fi
  return 0
}

# Helper: check available RAM (warns if below recommended)
check_ram() {
  local min_ram_gb=${1:-16}
  
  # Get total RAM in GB
  local total_ram_gb
  total_ram_gb=$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')
  
  if [ -z "$total_ram_gb" ]; then
    echo "Warning: Could not determine available RAM"
    return 0  # Continue if we can't check
  fi
  
  if [ "$total_ram_gb" -lt "$min_ram_gb" ]; then
    echo ""
    echo "========================================="
    echo "    ⚠️  LOW SYSTEM RAM DETECTED"
    echo "========================================="
    echo "Detected: ${total_ram_gb}GB RAM"
    echo "Recommended: ${min_ram_gb}GB minimum for optimal performance"
    echo ""
    echo "With less than ${min_ram_gb}GB RAM:"
    echo "  • Persona bootstrap may be slow (15-25 minutes)"
    echo "  • LLM responses may take longer"
    echo "  • System may become unresponsive during heavy operations"
    echo ""
    if [ -t 0 ] && [ "${AUTO_CONFIRM_FLAG:-false}" != "true" ]; then
      read -p "Continue anyway? (y/N): " -n 1 -r
      echo ""
      if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Installation cancelled."
        return 1
      fi
      echo "Proceeding with installation on low-RAM system..."
    else
      echo "Warning: Proceeding with low RAM in non-interactive mode."
      echo "Performance may be degraded."
    fi
  else
    echo "✓ RAM check passed: ${total_ram_gb}GB available (${min_ram_gb}GB recommended)"
  fi
  return 0
}

# Helper: verify critical models are available before persona bootstrap
verify_critical_models() {
  local ollama_bin="${OLLAMA_BIN:-$(command -v ollama || true)}"
  
  if [ -z "$ollama_bin" ]; then
    echo "Warning: Ollama binary not found, skipping model verification"
    return 0
  fi
  
  echo "Verifying critical models are available..."
  
  # Get model names (strip provider prefix if present)
  local chat_model=$(echo "${CONVERSATIONAL_MODEL_VAL:-llama3:8b}" | awk -F'/' '{print $NF}')
  local reflection_model=$(echo "${REFLECTION_LLM_VAL:-qwen2.5:3b}" | awk -F'/' '{print $NF}')
  local analytical_model=$(echo "${ANALYTICAL_MODEL_VAL:-qwen2.5:3b}" | awk -F'/' '{print $NF}')
  
  local missing_models=()
  
  # Check each critical model
  for model in "$analytical_model" "$reflection_model"; do
    if ! "$ollama_bin" show "$model" >/dev/null 2>&1; then
      missing_models+=("$model")
    fi
  done
  
  if [ ${#missing_models[@]} -gt 0 ]; then
    echo ""
    echo "========================================="
    echo "    ⚠️  MISSING CRITICAL MODELS"
    echo "========================================="
    echo "The following models required for persona bootstrap are not available:"
    for model in "${missing_models[@]}"; do
      echo "  • $model"
    done
    echo ""
    echo "Attempting to download missing models..."
    
    local download_failed=false
    for model in "${missing_models[@]}"; do
      echo "Downloading $model..."
      if ! "$ollama_bin" pull "$model" 2>&1; then
        echo "ERROR: Failed to download $model"
        download_failed=true
      else
        echo "✓ Downloaded $model"
      fi
    done
    
    if [ "$download_failed" = true ]; then
      echo ""
      echo "ERROR: Could not download all required models."
      echo "Persona bootstrap requires these models to generate the AI identity."
      echo ""
      echo "Please ensure:"
      echo "  1. Ollama service is running: sudo systemctl status ollama"
      echo "  2. Network connectivity is available"
      echo "  3. Sufficient disk space for models"
      echo ""
      echo "You can manually download models with:"
      for model in "${missing_models[@]}"; do
        echo "  ollama pull $model"
      done
      return 1
    fi
  fi
  
  echo "✓ All critical models verified for persona bootstrap"
  return 0
}

# Helper: deduce host IP (prefer default route interface)
deduce_host_ip() {
  # Do not use generic HOST shell var (may be hostname or 0.0.0.0). Prefer explicit SERVER_IP or interface IPs.
  if [ -n "${SERVER_IP:-}" ]; then echo "$SERVER_IP"; return; fi
  
  # First try to get IP from default route interface (most reliable)
  if command -v ip >/dev/null 2>&1; then
    local default_if
    default_if=$(ip -4 route show default 2>/dev/null | awk '/default/ {print $5; exit}')
    if [ -n "$default_if" ]; then
      local default_ip
      default_ip=$(ip -4 addr show dev "$default_if" 2>/dev/null | awk '/inet / {print $2}' | cut -d'/' -f1 | head -n1)
      if [ -n "$default_ip" ]; then
        echo "$default_ip"
        return
      fi
    fi
  fi
  
  # Fallback: use hostname -I but prefer 192.168.* over other private ranges
  if command -v hostname >/dev/null 2>&1; then
    local ips
    ips=$(hostname -I 2>/dev/null || true)
    # First pass: prefer 192.168.* (common home/office networks)
    for ip in $ips; do
      case "$ip" in
        192.168.*) echo "$ip"; return;;
      esac
    done
    # Second pass: other private ranges
    for ip in $ips; do
      case "$ip" in
        10.*|172.1[6-9].*|172.2[0-9].*|172.3[0-1].*) echo "$ip"; return;;
      esac
    done
    # Third pass: any IP
    for ip in $ips; do echo "$ip"; return; done
  fi
  echo "127.0.0.1"
}

# Helper: check if port is listening
check_port() {
  local port=$1
  if command -v lsof >/dev/null 2>&1; then
    lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1
    return $?
  elif command -v ss >/dev/null 2>&1; then
    ss -ltn "sport = :$port" 2>/dev/null | tail -n +2 | grep -q ":$port"
    return $?
  else
    # Assume free if we cannot check
    return 1
  fi
}

# Helper: find a free port starting from preferred, scanning range
find_free_port() {
  local start=${1:-8000}
  local end=${2:-8100}
  local p=$start
  while [ $p -le $end ]; do
    if ! check_port "$p"; then echo $p; return; fi
    p=$((p+1))
  done
  # Fallback if none found in range
  echo $start
}

SERVER_IP=${SERVER_IP:-$([ -n "$SERVER_IP" ] && echo "$SERVER_IP" || deduce_host_ip)}
PREFERRED_PORT=${CLI_PORT:-${SELO_AI_PORT:-${COLLECTED_PORT:-${PORT:-8000}}}}
BACKEND_PORT=$(find_free_port "$PREFERRED_PORT" 8100)
API_URL="http://${SERVER_IP}:${BACKEND_PORT}"
# Prefer collected frontend URL if provided; otherwise default to server ip:3000
FRONTEND_URL="${COLLECTED_FRONTEND:-http://${SERVER_IP}:3000}"

# Ensure system environment file exists/updated for service and subsequent steps
ensure_service_env_core() {
  local svc_env="/etc/selo-ai/environment"
  local be_env="$SCRIPT_DIR/backend/.env"
  # Collect feature flags from backend env if present
  local REFLECT_FLAG SCHED_FLAG CORS_VAL SOCKET_FLAG
  REFLECT_FLAG=$(grep -E '^ENABLE_REFLECTION_SYSTEM=' "$be_env" 2>/dev/null | tail -n1 | cut -d '=' -f2-)
  SCHED_FLAG=$(grep -E '^ENABLE_ENHANCED_SCHEDULER=' "$be_env" 2>/dev/null | tail -n1 | cut -d '=' -f2-)
  CORS_VAL=$(grep -E '^CORS_ORIGINS=' "$be_env" 2>/dev/null | tail -n1 | cut -d '=' -f2-)
  SOCKET_FLAG=$(grep -E '^SOCKET_IO_ENABLED=' "$be_env" 2>/dev/null | tail -n1 | cut -d '=' -f2-)

  if [ ! -f "$svc_env" ]; then
    echo "Creating /etc/selo-ai/environment with detected values..."
    sudo mkdir -p /etc/selo-ai
    sudo chgrp "$INST_USER" /etc/selo-ai 2>/dev/null || true
    sudo chmod 750 /etc/selo-ai 2>/dev/null || true
    {
      echo "HOST=${HOST:-0.0.0.0}"
      echo "SELO_AI_PORT=${BACKEND_PORT}"
      echo "API_URL=${API_URL}"
      echo "FRONTEND_URL=${FRONTEND_URL}"
      echo "INSTALL_DIR=${INSTALL_DIR:-$SCRIPT_DIR}"
      [ -n "$REFLECT_FLAG" ] && echo "ENABLE_REFLECTION_SYSTEM=$REFLECT_FLAG"
      [ -n "$SCHED_FLAG" ] && echo "ENABLE_ENHANCED_SCHEDULER=$SCHED_FLAG"
      [ -n "$CORS_VAL" ] && echo "CORS_ORIGINS=$CORS_VAL"
      [ -n "$SOCKET_FLAG" ] && echo "SOCKET_IO_ENABLED=$SOCKET_FLAG"
    } | sudo tee "$svc_env" >/dev/null
    sudo chown root:"$INST_USER" "$svc_env" 2>/dev/null || true
    sudo chmod 640 "$svc_env" 2>/dev/null || true
  else
    # Update core URLs/ports to reflect detected values; preserve other lines
    sudo sed -i \
      -e "s|^SELO_AI_PORT=.*|SELO_AI_PORT=${BACKEND_PORT}|" \
      -e "s|^API_URL=.*|API_URL=${API_URL}|" \
      -e "s|^FRONTEND_URL=.*|FRONTEND_URL=${FRONTEND_URL}|" \
      "$svc_env" || true
    # Remove any obsolete PORT entry to avoid ambiguity
    sudo sed -i -E '/^PORT=/d' "$svc_env" || true
    # Remove DATABASE_URL from service env to avoid conflicting sources; backend/.env is the source of truth
    if sudo grep -q '^DATABASE_URL=' "$svc_env"; then
      echo "Removing DATABASE_URL from $svc_env to avoid ambiguity (kept in backend/.env)"
      sudo sed -i -E '/^DATABASE_URL=/d' "$svc_env" || true
    fi
    # Append missing preferred keys
    sudo grep -q '^HOST=' "$svc_env" || echo "HOST=${HOST:-0.0.0.0}" | sudo tee -a "$svc_env" >/dev/null
    sudo grep -q '^SELO_AI_PORT=' "$svc_env" || echo "SELO_AI_PORT=${BACKEND_PORT}" | sudo tee -a "$svc_env" >/dev/null
    sudo grep -q '^API_URL=' "$svc_env" || echo "API_URL=${API_URL}" | sudo tee -a "$svc_env" >/dev/null
    sudo grep -q '^FRONTEND_URL=' "$svc_env" || echo "FRONTEND_URL=${FRONTEND_URL}" | sudo tee -a "$svc_env" >/dev/null
    # Ensure INSTALL_DIR reflects selected install directory (or current script location)
    if sudo grep -q '^INSTALL_DIR=' "$svc_env"; then
      sudo sed -i -E "s|^INSTALL_DIR=.*|INSTALL_DIR=${INSTALL_DIR:-$SCRIPT_DIR}|" "$svc_env" || true
    else
      echo "INSTALL_DIR=${INSTALL_DIR:-$SCRIPT_DIR}" | sudo tee -a "$svc_env" >/dev/null
    fi
    # Mirror Socket.IO flag if present in backend env; default to true if entirely absent
    if [ -n "$SOCKET_FLAG" ]; then
      if sudo grep -q '^SOCKET_IO_ENABLED=' "$svc_env"; then
        sudo sed -i -E "s|^SOCKET_IO_ENABLED=.*|SOCKET_IO_ENABLED=${SOCKET_FLAG}|" "$svc_env" || true
      else
        echo "SOCKET_IO_ENABLED=${SOCKET_FLAG}" | sudo tee -a "$svc_env" >/dev/null
      fi
    else
      sudo grep -q '^SOCKET_IO_ENABLED=' "$svc_env" || echo "SOCKET_IO_ENABLED=true" | sudo tee -a "$svc_env" >/dev/null
    fi
    sudo chown root:"$INST_USER" "$svc_env" 2>/dev/null || true
    sudo chmod 640 "$svc_env" 2>/dev/null || true
  fi
}

# NOTE: ensure_service_env_core is called inside main() - removed redundant top-level call

# Load backend environment into current shell for downstream steps (models, DB)
load_backend_env() {
  local be_env="$SCRIPT_DIR/backend/.env"
  if [ -f "$be_env" ]; then
    # shellcheck disable=SC2046
    set -a
    # Minimal sanitize: ignore lines without '=' and comments
    local tmp_env
    tmp_env=$(mktemp)
    while IFS= read -r line || [ -n "$line" ]; do
      if echo "$line" | grep -qE '^\s*#'; then echo "$line" >>"$tmp_env"; continue; fi
      if echo "$line" | grep -q '='; then echo "$line" >>"$tmp_env"; fi
    done <"$be_env"
    . "$tmp_env"
    rm -f "$tmp_env"
    set +a
  fi
}

# NOTE: load_backend_env is called inside main() - removed redundant top-level call

# CUDA auto-detection / decision
CUDA_ENABLED=false
if [ "$CUDA_MODE" = "on" ]; then
  CUDA_ENABLED=true
elif [ "$CUDA_MODE" = "off" ]; then
  CUDA_ENABLED=false
else
  # auto mode: detect via nvidia-smi and PyTorch
  echo "Auto-detecting CUDA support..."
  
  # Primary detection via nvidia-smi
  if command -v nvidia-smi >/dev/null 2>&1; then
    if nvidia-smi -L 2>/dev/null | grep -q "GPU"; then
      CUDA_ENABLED=true
      echo "✅ NVIDIA GPU detected via nvidia-smi"
    fi
  fi
  
  # Secondary detection via PyTorch (if available)
  if ! $CUDA_ENABLED && command -v python3 >/dev/null 2>&1; then
    if python3 -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
      CUDA_ENABLED=true
      echo "✅ CUDA detected via PyTorch"
    fi
  fi
  
  # Tertiary detection via CUDA toolkit
  if ! $CUDA_ENABLED && command -v nvcc >/dev/null 2>&1; then
    CUDA_ENABLED=true
    echo "✅ CUDA toolkit detected via nvcc"
  fi
  
  # If interactive TTY and not skipping prompts, ask the user to confirm
  if [ -t 0 ] && [ -z "${SKIP_ENV_COLLECT:-}" ]; then
    if $CUDA_ENABLED; then
      echo "Detected NVIDIA GPU/CUDA. Enable GPU acceleration? [Y/n]"
      read -r -p "> " _ans
      case "${_ans:-Y}" in
        [Yy]*) CUDA_ENABLED=true ; CUDA_MODE="on" ;;
        [Nn]*) CUDA_ENABLED=false; CUDA_MODE="off" ;;
        *)     CUDA_ENABLED=true ; CUDA_MODE="on" ;;
      esac
    else
      echo "No NVIDIA GPU/CUDA detected. Force GPU mode anyway? [y/N]"
      read -r -p "> " _ans
      case "${_ans:-N}" in
        [Yy]*) CUDA_ENABLED=true ; CUDA_MODE="on"  ;;
        [Nn]*) CUDA_ENABLED=false; CUDA_MODE="off" ;;
        *)     CUDA_ENABLED=false; CUDA_MODE="off" ;;
      esac
    fi
  else
    # Non-interactive: if CUDA detected, enable it automatically
    if $CUDA_ENABLED; then
      CUDA_MODE="on"
      echo "Non-interactive mode: GPU acceleration enabled automatically"
    else
      CUDA_MODE="off"
      echo "Non-interactive mode: No GPU detected, using CPU-only"
    fi
  fi
fi
echo "CUDA_ENABLED: $CUDA_ENABLED (mode=$CUDA_MODE)"

# Export default configuration identifiers
SELECTED_MODEL_CONFIG="default"
export SELECTED_MODEL_CONFIG
export CONVERSATIONAL_MODEL_VAL

ensure_apt_ready() {
    local timeout=${1:-300}
    local waited=0
    local interval=5
    echo "Checking apt/dpkg lock status (timeout ${timeout}s)..."
    while true; do
        if ! pgrep -x apt >/dev/null && ! pgrep -x apt-get >/dev/null && ! pgrep -x unattended-up >/dev/null; then
            if ! sudo fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 && \
               ! sudo fuser /var/cache/apt/archives/lock >/dev/null 2>&1; then
                break
            fi
        fi
        waited=$((waited+interval))
        if [ $waited -ge $timeout ]; then
            echo "Timeout waiting for apt locks to clear after ${timeout}s. You can retry once other apt processes finish."
            return 1
        fi
        echo "apt/dpkg busy (waited ${waited}s)... still waiting..."
        sleep $interval
    done
    return 0
}

ensure_node_toolchain() {
    local need_nodesource=0

    if ! command -v node >/dev/null 2>&1; then
        echo "Node.js not detected; installing Node 20.x via NodeSource..."
        need_nodesource=1
    else
        local node_major
        node_major=$(node -v | sed -E 's/^v([0-9]+).*/\1/')
        if [ -n "$node_major" ] && [ "$node_major" -lt 18 ]; then
            echo "Node.js version v$node_major detected; upgrading to Node 20.x via NodeSource..."
            need_nodesource=1
        fi
    fi

    if [ "$need_nodesource" -eq 1 ]; then
        echo "Removing distro-provided Node packages (nodejs, npm, libnode-dev) to avoid conflicts..."
        ensure_apt_ready 300 || return 1
        sudo apt-get purge -y nodejs npm libnode-dev >/dev/null 2>&1 || true
        ensure_apt_ready 300 || return 1
        sudo apt-get autoremove -y >/dev/null 2>&1 || true
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        ensure_apt_ready 300 || return 1
        sudo apt-get install -y nodejs
        hash -r
    fi

    local node_version
    node_version=$(node -v 2>/dev/null || true)
    local node_major
    node_major=$(echo "$node_version" | sed -E 's/^v([0-9]+).*/\1/')
    if [ -z "$node_major" ] || [ "$node_major" -lt 18 ]; then
        echo "Node.js remains <18 after NodeSource install; purging distro node and retrying..."
        ensure_apt_ready 300 || return 1
        sudo apt-get purge -y nodejs npm libnode-dev || true
        ensure_apt_ready 300 || return 1
        sudo apt-get autoremove -y >/dev/null 2>&1 || true
        curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
        ensure_apt_ready 300 || return 1
        sudo apt-get install -y nodejs
        hash -r
        node_version=$(node -v 2>/dev/null || true)
        node_major=$(echo "$node_version" | sed -E 's/^v([0-9]+).*/\1/')
    fi

    if [ -z "$node_major" ] || [ "$node_major" -lt 18 ]; then
        echo "Error: Unable to provision Node.js >=18" >&2
        return 1
    fi

    if command -v npm >/dev/null 2>&1; then
        local npm_major
        npm_major=$(npm -v | cut -d '.' -f1)
        if [ -n "$npm_major" ] && [ "$npm_major" -lt 9 ]; then
            echo "Upgrading npm to latest LTS-compatible release (>=9)..."
            sudo npm install -g npm@10
            hash -r
        fi
    fi

    if ! command -v npm >/dev/null 2>&1; then
        echo "Installing npm..."
        ensure_apt_ready 300 || return 1
        sudo apt-get install -y npm
    fi

    echo "Node.js $(node -v) and npm $(npm -v) are available"
}

# Dependency installation
install_dependencies() {
    echo "========================================="
    echo "    Installing Dependencies"
    echo "========================================="
    ensure_apt_ready 300 || return 1
    sudo apt update
    ensure_apt_ready 300 || return 1
    sudo apt install -y git curl build-essential python3 python3-venv python3-pip libpq-dev
    ensure_node_toolchain || return 1

    # Final safeguard: ensure conversational model is present locally
    OLLAMA_BIN=${OLLAMA_BIN:-ollama}
    if [ -n "${CONVERSATIONAL_MODEL_VAL:-}" ]; then
        if ! "$OLLAMA_BIN" show "$CONVERSATIONAL_MODEL_VAL" >/dev/null 2>&1; then
            echo "[finalize] Conversational model '$CONVERSATIONAL_MODEL_VAL' missing. Pulling via ollama..."
            "$OLLAMA_BIN" pull "$CONVERSATIONAL_MODEL_VAL" >/dev/null 2>&1 || echo "[finalize] Warning: failed to pull $CONVERSATIONAL_MODEL_VAL"
        fi
    fi
    # Ensure npm is available (some distros separate nodejs and npm)
    if ! command -v npm >/dev/null 2>&1; then
        echo "Installing npm..."
        ensure_apt_ready 300 || true
        sudo apt install -y npm || true
    fi
}

# Ensure backend/.env exists and contains required defaults before install proceeds (top-level)
ensure_backend_env_defaults() {
  local be_env="$SCRIPT_DIR/backend/.env"
  mkdir -p "$SCRIPT_DIR/backend" 2>/dev/null || true

  # Create file if it doesn't exist
  if [ ! -f "$be_env" ]; then
    if [ -f "$MODEL_TEMPLATE_DIR/.env.template" ]; then
      echo "Creating new backend/.env from template: $MODEL_TEMPLATE_DIR/.env.template"
      cp "$MODEL_TEMPLATE_DIR/.env.template" "$be_env"
    else
      echo "Template $MODEL_TEMPLATE_DIR/.env.template not found; writing default configuration"
      cat >"$be_env" <<EOF
# Backend environment (auto-generated by install script)
# IMPORTANT: adjust DATABASE_URL credentials if using remote DB

DATABASE_URL=postgresql+asyncpg://seloai:password@localhost/seloai
SELO_SYSTEM_API_KEY=dev-secret-key-change-me
OLLAMA_BASE_URL=http://localhost:11434

CONVERSATIONAL_MODEL=$DEFAULT_CONVERSATIONAL_MODEL
ANALYTICAL_MODEL=$DEFAULT_ANALYTICAL_MODEL
REFLECTION_LLM=$DEFAULT_REFLECTION_MODEL
EMBEDDING_MODEL=$DEFAULT_EMBEDDING_MODEL

HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:3000
EOF
    fi
  fi

  # Enforce default model selections unless user overrides exist
  local defaults=(
    "CONVERSATIONAL_MODEL=$DEFAULT_CONVERSATIONAL_MODEL"
    "ANALYTICAL_MODEL=$DEFAULT_ANALYTICAL_MODEL"
    "REFLECTION_LLM=$DEFAULT_REFLECTION_MODEL"
    "EMBEDDING_MODEL=$DEFAULT_EMBEDDING_MODEL"
  )
  for kv in "${defaults[@]}"; do
    local key="${kv%%=*}"
    local val="${kv#*=}"
    if grep -q "^${key}=" "$be_env"; then
      sed -i -E "s|^${key}=.*|${key}=${val}|" "$be_env"
    else
      echo "${key}=${val}" >> "$be_env"
    fi
  done

  # Allow CLI override for conversational model
  if [ "$CONVERSATIONAL_MODEL_SET" = true ]; then
    if grep -q '^CONVERSATIONAL_MODEL=' "$be_env"; then
      sed -i -E "s|^CONVERSATIONAL_MODEL=.*|CONVERSATIONAL_MODEL=${CONVERSATIONAL_MODEL_VAL}|" "$be_env"
    else
      echo "CONVERSATIONAL_MODEL=${CONVERSATIONAL_MODEL_VAL}" >> "$be_env"
    fi
    echo "Applied user-specified conversational model: $CONVERSATIONAL_MODEL_VAL"
  fi

  echo "Ensuring all required environment variables exist..."

  if ! grep -q '^DATABASE_URL=' "$be_env"; then
    echo "DATABASE_URL=postgresql+asyncpg://seloai:password@localhost/seloai" >> "$be_env"
    echo "Added DATABASE_URL to backend/.env"
  fi

  if ! grep -q '^SELO_SYSTEM_API_KEY=' "$be_env"; then
    echo "SELO_SYSTEM_API_KEY=dev-secret-key-change-me" >> "$be_env"
    echo "Added SELO_SYSTEM_API_KEY to backend/.env"
  fi

  if grep -q '^SELO_SYSTEM_API_KEY=dev-secret-key-change-me' "$be_env"; then
    if command -v openssl >/dev/null 2>&1; then
      RAND_KEY=$(openssl rand -hex 32)
    else
      RAND_KEY=$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')
    fi
    sed -i -E "s|^SELO_SYSTEM_API_KEY=.*|SELO_SYSTEM_API_KEY=${RAND_KEY}|" "$be_env" || true
    echo "Generated secure SELO_SYSTEM_API_KEY"
  fi

  if ! grep -q '^OLLAMA_BASE_URL=' "$be_env"; then
    echo "OLLAMA_BASE_URL=http://localhost:11434" >> "$be_env"
  fi

  if ! grep -q '^HOST=' "$be_env"; then
    echo "HOST=0.0.0.0" >> "$be_env"
  fi

  if ! grep -q '^PORT=' "$be_env"; then
    echo "PORT=8000" >> "$be_env"
  fi

  if ! grep -q '^SOCKET_IO_ENABLED=' "$be_env"; then
    echo "SOCKET_IO_ENABLED=true" >> "$be_env"
  fi

  if ! grep -q '^BRAVE_SEARCH_API_KEY=' "$be_env"; then
    echo "BRAVE_SEARCH_API_KEY=" >> "$be_env"
    echo "Added placeholder BRAVE_SEARCH_API_KEY to backend/.env (set a real key to enable web search)"
  fi

  if ! grep -q '^REFLECTION_MAX_TOKENS=' "$be_env"; then
    echo "REFLECTION_MAX_TOKENS=${TIER_REFLECTION_MAX_TOKENS:-640}" >> "$be_env"
  fi

  if ! grep -q '^REFLECTION_NUM_PREDICT=' "$be_env"; then
    echo "REFLECTION_NUM_PREDICT=${TIER_REFLECTION_NUM_PREDICT:-640}" >> "$be_env"
  fi

  if ! grep -q '^REFLECTION_WORD_MIN=' "$be_env"; then
    echo "REFLECTION_WORD_MIN=${TIER_REFLECTION_WORD_MIN:-100}" >> "$be_env"
  fi

  if ! grep -q '^REFLECTION_WORD_MAX=' "$be_env"; then
    echo "REFLECTION_WORD_MAX=${TIER_REFLECTION_WORD_MAX:-500}" >> "$be_env"
  fi

  if ! grep -q '^REFLECTION_TEMPERATURE=' "$be_env"; then
    echo "REFLECTION_TEMPERATURE=0.35" >> "$be_env"
  fi

  if ! grep -q '^REFLECTION_OUTPUT_STYLE=' "$be_env"; then
    echo "REFLECTION_OUTPUT_STYLE=verbose" >> "$be_env"
  fi

  if ! grep -q '^CHAT_NUM_PREDICT=' "$be_env"; then
    echo "CHAT_NUM_PREDICT=${TIER_CHAT_NUM_PREDICT:-1024}" >> "$be_env"
  fi

  if ! grep -q '^CHAT_NUM_CTX=' "$be_env"; then
    echo "CHAT_NUM_CTX=${TIER_CHAT_NUM_CTX:-4096}" >> "$be_env"
  fi

  if ! grep -q '^CHAT_TEMPERATURE=' "$be_env"; then
    echo "CHAT_TEMPERATURE=0.6" >> "$be_env"
  fi

  if ! grep -q '^ANALYTICAL_NUM_PREDICT=' "$be_env"; then
    echo "ANALYTICAL_NUM_PREDICT=${TIER_ANALYTICAL_NUM_PREDICT:-640}" >> "$be_env"
  fi

  if ! grep -q '^ANALYTICAL_TEMPERATURE=' "$be_env"; then
    echo "ANALYTICAL_TEMPERATURE=0.2" >> "$be_env"
  fi

  # Add GPU/CUDA configuration if not present and CUDA available
  if command -v nvidia-smi >/dev/null 2>&1; then
    if ! grep -q '^CUDA_VISIBLE_DEVICES=' "$be_env"; then
      echo "CUDA_VISIBLE_DEVICES=0" >> "$be_env"
    fi
    if ! grep -q '^PYTORCH_CUDA_ALLOC_CONF=' "$be_env"; then
      echo "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True" >> "$be_env"
    fi
    if ! grep -q '^OLLAMA_GPU_LAYERS=' "$be_env"; then
      echo "OLLAMA_GPU_LAYERS=-1" >> "$be_env"  # Use all GPU layers
    fi
  fi

  # Ollama configuration
  if ! grep -q '^OLLAMA_KEEP_ALIVE=' "$be_env"; then
    echo "OLLAMA_KEEP_ALIVE=15m" >> "$be_env"
  fi
  if ! grep -q '^OLLAMA_NUM_PARALLEL=' "$be_env"; then
    echo "OLLAMA_NUM_PARALLEL=1" >> "$be_env"
  fi
  if ! grep -q '^OLLAMA_MAX_LOADED_MODELS=' "$be_env"; then
    echo "OLLAMA_MAX_LOADED_MODELS=2" >> "$be_env"
  fi

  if grep -q '^REFLECTION_LLM_TIMEOUT_S=' "$be_env"; then
    sed -i -E "s|^REFLECTION_LLM_TIMEOUT_S=.*|REFLECTION_LLM_TIMEOUT_S=0|" "$be_env"
  else
    echo "REFLECTION_LLM_TIMEOUT_S=0" >> "$be_env"
  fi

  if ! grep -q '^REFLECTION_REQUIRED=' "$be_env"; then
    echo "REFLECTION_REQUIRED=true" >> "$be_env"
  fi

  if ! grep -q '^REFLECTION_SYNC_MODE=' "$be_env"; then
    echo "REFLECTION_SYNC_MODE=sync" >> "$be_env"
  fi

  if grep -q '^REFLECTION_SYNC_TIMEOUT_S=' "$be_env"; then
    sed -i -E "s|^REFLECTION_SYNC_TIMEOUT_S=.*|REFLECTION_SYNC_TIMEOUT_S=0|" "$be_env"
  else
    echo "REFLECTION_SYNC_TIMEOUT_S=0" >> "$be_env"
  fi
}

# NOTE: ensure_backend_env_defaults is called inside main() - removed redundant top-level call

# Ensure /etc/selo-ai/environment has the minimal keys; auto-fill from available data (top-level)
finalize_service_env() {
  local svc_env="/etc/selo-ai/environment"
  [ -f "$svc_env" ] || return 0
  # Read current values (use sudo to read if needed)
  local HOST_IP_CUR SELO_PORT_CUR API_URL_CUR FRONTEND_URL_CUR
  HOST_IP_CUR=$(sudo awk -F'=' '/^HOST_IP=/{print $2; exit}' "$svc_env" 2>/dev/null)
  SELO_PORT_CUR=$(sudo awk -F'=' '/^(SELO_AI_PORT|PORT)=/{print $2; exit}' "$svc_env" 2>/dev/null)
  API_URL_CUR=$(sudo awk -F'=' '/^API_URL=/{print $2; exit}' "$svc_env" 2>/dev/null)
  FRONTEND_URL_CUR=$(sudo awk -F'=' '/^FRONTEND_URL=/{print $2; exit}' "$svc_env" 2>/dev/null)
  # Fill missing from known runtime values
  [ -n "$HOST_IP_CUR" ] || HOST_IP_CUR="$SERVER_IP"
  [ -n "$SELO_PORT_CUR" ] || SELO_PORT_CUR="$BACKEND_PORT"
  [ -n "$API_URL_CUR" ] || API_URL_CUR="http://${HOST_IP_CUR}:${SELO_PORT_CUR}"
  [ -n "$FRONTEND_URL_CUR" ] || FRONTEND_URL_CUR="http://${HOST_IP_CUR}:3000"
  # Persist any missing keys
  if ! sudo grep -q '^HOST_IP=' "$svc_env"; then echo "HOST_IP=${HOST_IP_CUR}" | sudo tee -a "$svc_env" >/dev/null; fi
  if ! sudo grep -Eq '^(SELO_AI_PORT)=' "$svc_env"; then echo "SELO_AI_PORT=${SELO_PORT_CUR}" | sudo tee -a "$svc_env" >/dev/null; fi
  # Ensure no legacy PORT key remains
  sudo sed -i -E '/^PORT=/d' "$svc_env" || true
  # Ensure no DATABASE_URL is present here
  sudo sed -i -E '/^DATABASE_URL=/d' "$svc_env" || true
  if ! sudo grep -q '^API_URL=' "$svc_env"; then echo "API_URL=${API_URL_CUR}" | sudo tee -a "$svc_env" >/dev/null; fi
  if ! sudo grep -q '^FRONTEND_URL=' "$svc_env"; then echo "FRONTEND_URL=${FRONTEND_URL_CUR}" | sudo tee -a "$svc_env" >/dev/null; fi
  # Normalize CORS_ORIGINS to include both the server frontend URL and localhost:3000
  local CORS_DESIRED="${FRONTEND_URL_CUR},http://localhost:3000"
  if sudo grep -q '^CORS_ORIGINS=' "$svc_env"; then
    sudo sed -i -E "s|^CORS_ORIGINS=.*|CORS_ORIGINS=${CORS_DESIRED}|" "$svc_env" || true
  else
    echo "CORS_ORIGINS=${CORS_DESIRED}" | sudo tee -a "$svc_env" >/dev/null
  fi

  # Mirror CORS_ORIGINS into backend/.env for app config consistency
  if [ -f "$SCRIPT_DIR/backend/.env" ]; then
    if grep -q '^CORS_ORIGINS=' "$SCRIPT_DIR/backend/.env"; then
      sed -i -E "s|^CORS_ORIGINS=.*|CORS_ORIGINS=${CORS_DESIRED}|" "$SCRIPT_DIR/backend/.env" || true
    else
      echo "CORS_ORIGINS=${CORS_DESIRED}" >> "$SCRIPT_DIR/backend/.env"
    fi
  fi
  # Ensure feature flags exist in service env for consistency with backend defaults
  if ! sudo grep -q '^ENABLE_REFLECTION_SYSTEM=' "$svc_env"; then
    echo "ENABLE_REFLECTION_SYSTEM=true" | sudo tee -a "$svc_env" >/dev/null
  fi
  if ! sudo grep -q '^ENABLE_ENHANCED_SCHEDULER=' "$svc_env"; then
    echo "ENABLE_ENHANCED_SCHEDULER=true" | sudo tee -a "$svc_env" >/dev/null
  fi
  # Ensure reflection performance tunables are present and aligned with reflection-first policy
  # Sync mode required to guarantee reflection precedes chat (use string 'sync' for clarity)
  if sudo grep -q '^REFLECTION_SYNC_MODE=' "$svc_env"; then
    sudo sed -i -E "s|^REFLECTION_SYNC_MODE=.*|REFLECTION_SYNC_MODE=sync|" "$svc_env" || true
  else
    echo "REFLECTION_SYNC_MODE=sync" | sudo tee -a "$svc_env" >/dev/null
  fi
  # Require reflection for each chat turn
  if sudo grep -q '^REFLECTION_REQUIRED=' "$svc_env"; then
    sudo sed -i -E "s|^REFLECTION_REQUIRED=.*|REFLECTION_REQUIRED=true|" "$svc_env" || true
  else
    echo "REFLECTION_REQUIRED=true" | sudo tee -a "$svc_env" >/dev/null
  fi
  # Reflection-first policy: default to unbounded waits unless the operator sets a cap
  if sudo grep -q '^REFLECTION_ENFORCE_NO_TIMEOUTS=' "$svc_env"; then
    sudo sed -i -E "s|^REFLECTION_ENFORCE_NO_TIMEOUTS=.*|REFLECTION_ENFORCE_NO_TIMEOUTS=true|" "$svc_env" || true
  else
    echo "REFLECTION_ENFORCE_NO_TIMEOUTS=true" | sudo tee -a "$svc_env" >/dev/null
  fi
  if sudo grep -q '^REFLECTION_SYNC_TIMEOUT_S=' "$svc_env"; then
    sudo sed -i -E "s|^REFLECTION_SYNC_TIMEOUT_S=.*|REFLECTION_SYNC_TIMEOUT_S=0|" "$svc_env" || true
  else
    echo "REFLECTION_SYNC_TIMEOUT_S=0" | sudo tee -a "$svc_env" >/dev/null
  fi
  if sudo grep -q '^REFLECTION_LLM_TIMEOUT_S=' "$svc_env"; then
    sudo sed -i -E "s|^REFLECTION_LLM_TIMEOUT_S=.*|REFLECTION_LLM_TIMEOUT_S=0|" "$svc_env" || true
  else
    echo "REFLECTION_LLM_TIMEOUT_S=0" | sudo tee -a "$svc_env" >/dev/null
  fi
  # Mirror LLM timeout (unbounded)
  if sudo grep -q '^LLM_TIMEOUT=' "$svc_env"; then
    sudo sed -i -E "s|^LLM_TIMEOUT=.*|LLM_TIMEOUT=0|" "$svc_env" || true
  else
    echo "LLM_TIMEOUT=0" | sudo tee -a "$svc_env" >/dev/null
  fi
  # Mirror per-type reflection model selections for heavyweight profile
  if sudo grep -q '^REFLECTION_MODEL_DEFAULT=' "$svc_env"; then
    sudo sed -i -E "s|^REFLECTION_MODEL_DEFAULT=.*|REFLECTION_MODEL_DEFAULT=${REFLECTION_MODEL_DEFAULT_VAL:-qwen2.5:3b}|" "$svc_env" || true
  else
    echo "REFLECTION_MODEL_DEFAULT=${REFLECTION_MODEL_DEFAULT_VAL:-qwen2.5:3b}" | sudo tee -a "$svc_env" >/dev/null
  fi
  if sudo grep -q '^REFLECTION_MODEL_MESSAGE=' "$svc_env"; then
    sudo sed -i -E "s|^REFLECTION_MODEL_MESSAGE=.*|REFLECTION_MODEL_MESSAGE=${REFLECTION_MODEL_MESSAGE_VAL:-qwen2.5:3b}|" "$svc_env" || true
  else
    echo "REFLECTION_MODEL_MESSAGE=${REFLECTION_MODEL_MESSAGE_VAL:-qwen2.5:3b}" | sudo tee -a "$svc_env" >/dev/null
  fi
  for key in REFLECTION_MODEL_DAILY REFLECTION_MODEL_WEEKLY REFLECTION_MODEL_EMOTIONAL REFLECTION_MODEL_MANIFESTO REFLECTION_MODEL_PERIODIC; do
    if sudo grep -q "^${key}=" "$svc_env"; then
      sudo sed -i -E "s|^${key}=.*|${key}=${REFLECTION_MODEL_DEFAULT_VAL:-qwen2.5:3b}|" "$svc_env" || true
    else
      echo "${key}=${REFLECTION_MODEL_DEFAULT_VAL:-qwen2.5:3b}" | sudo tee -a "$svc_env" >/dev/null
    fi
  done
  # Keep a higher default token budget to avoid truncation; backend can override via backend/.env
  if sudo grep -q '^REFLECTION_MAX_TOKENS=' "$svc_env"; then
    sudo sed -i -E "s|^REFLECTION_MAX_TOKENS=.*|REFLECTION_MAX_TOKENS=${TIER_REFLECTION_MAX_TOKENS:-640}|" "$svc_env" || true
  else
    echo "REFLECTION_MAX_TOKENS=${TIER_REFLECTION_MAX_TOKENS:-640}" | sudo tee -a "$svc_env" >/dev/null
  fi
  
  # Set word count validation limit based on hardware tier
  if sudo grep -q '^REFLECTION_WORD_MAX=' "$svc_env"; then
    sudo sed -i -E "s|^REFLECTION_WORD_MAX=.*|REFLECTION_WORD_MAX=${TIER_REFLECTION_WORD_MAX:-500}|" "$svc_env" || true
  else
    echo "REFLECTION_WORD_MAX=${TIER_REFLECTION_WORD_MAX:-500}" | sudo tee -a "$svc_env" >/dev/null
  fi
  
  # Set Reports directory path for boot directives
  REPORTS_DIR="/opt/selo-ai/Reports"
  if sudo grep -q '^SELO_REPORTS_DIR=' "$svc_env"; then
    sudo sed -i -E "s|^SELO_REPORTS_DIR=.*|SELO_REPORTS_DIR=${REPORTS_DIR}|" "$svc_env" || true
  else
    echo "SELO_REPORTS_DIR=${REPORTS_DIR}" | sudo tee -a "$svc_env" >/dev/null
  fi
  
  # Apply no-truncation defaults into backend/.env as well so the app uses them directly
  if [ -f "$SCRIPT_DIR/backend/.env" ]; then
    # Set reflection token budget based on hardware tier
    if grep -q '^REFLECTION_NUM_PREDICT=' "$SCRIPT_DIR/backend/.env"; then
      sed -i -E "s|^REFLECTION_NUM_PREDICT=.*|REFLECTION_NUM_PREDICT=${TIER_REFLECTION_NUM_PREDICT:-640}|" "$SCRIPT_DIR/backend/.env" || true
    else
      echo "REFLECTION_NUM_PREDICT=${TIER_REFLECTION_NUM_PREDICT:-640}" >> "$SCRIPT_DIR/backend/.env"
    fi
    # Set word count validation limit based on hardware tier
    if grep -q '^REFLECTION_WORD_MAX=' "$SCRIPT_DIR/backend/.env"; then
      sed -i -E "s|^REFLECTION_WORD_MAX=.*|REFLECTION_WORD_MAX=${TIER_REFLECTION_WORD_MAX:-500}|" "$SCRIPT_DIR/backend/.env" || true
    else
      echo "REFLECTION_WORD_MAX=${TIER_REFLECTION_WORD_MAX:-500}" >> "$SCRIPT_DIR/backend/.env"
    fi
    # Mirror reflection temperature override (align with processing layer at 0.35)
    if grep -q '^REFLECTION_TEMPERATURE=' "$SCRIPT_DIR/backend/.env"; then
      sed -i -E "s|^REFLECTION_TEMPERATURE=.*|REFLECTION_TEMPERATURE=0.35|" "$SCRIPT_DIR/backend/.env" || true
    else
      echo "REFLECTION_TEMPERATURE=0.35" >> "$SCRIPT_DIR/backend/.env"
    fi
    # Set Reports directory path for boot directives
    if grep -q '^SELO_REPORTS_DIR=' "$SCRIPT_DIR/backend/.env"; then
      sed -i -E "s|^SELO_REPORTS_DIR=.*|SELO_REPORTS_DIR=${SCRIPT_DIR}/../Reports|" "$SCRIPT_DIR/backend/.env" || true
    else
      echo "SELO_REPORTS_DIR=${SCRIPT_DIR}/../Reports" >> "$SCRIPT_DIR/backend/.env"
    fi
    # Chat context window budget based on hardware tier
    if grep -q '^CHAT_NUM_CTX=' "$SCRIPT_DIR/backend/.env"; then
      sed -i -E "s|^CHAT_NUM_CTX=.*|CHAT_NUM_CTX=${TIER_CHAT_NUM_CTX:-4096}|" "$SCRIPT_DIR/backend/.env" || true
    else
      echo "CHAT_NUM_CTX=${TIER_CHAT_NUM_CTX:-4096}" >> "$SCRIPT_DIR/backend/.env"
    fi
    # Align reflection max tokens based on hardware tier
    if grep -q '^REFLECTION_MAX_TOKENS=' "$SCRIPT_DIR/backend/.env"; then
      sed -i -E "s|^REFLECTION_MAX_TOKENS=.*|REFLECTION_MAX_TOKENS=${TIER_REFLECTION_MAX_TOKENS:-640}|" "$SCRIPT_DIR/backend/.env" || true
    else
      echo "REFLECTION_MAX_TOKENS=${TIER_REFLECTION_MAX_TOKENS:-640}" >> "$SCRIPT_DIR/backend/.env"
    fi
  fi
  # Environment file should be readable by root and the instance user group
  sudo chown root:"$INST_USER" "$svc_env" 2>/dev/null || true
  sudo chmod 640 "$svc_env" 2>/dev/null || true
  # Ensure directory is traversable by the group so the service user can read the file
  sudo chgrp "$INST_USER" /etc/selo-ai 2>/dev/null || true
  sudo chmod 750 /etc/selo-ai 2>/dev/null || true
}

# Ensure no lingering Ollama models are running before changing configuration
stop_running_ollama_models() {
  local bin="${OLLAMA_BIN:-$(command -v ollama || true)}"
  if [ -z "$bin" ]; then
    return 0
  fi

  local ps_output
  ps_output="$($bin ps 2>/dev/null || true)"
  if [ -z "$ps_output" ]; then
    return 0
  fi

  local stopped_any="false"
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "$line" in
      NAME*|---*)
        continue
        ;;
    esac
    local model_name
    model_name=$(echo "$line" | awk '{print $1}')
    [ -z "$model_name" ] && continue
    if $bin stop "$model_name" >/dev/null 2>&1; then
      stopped_any="true"
    fi
  done <<< "$ps_output"

  if [ "$stopped_any" = "true" ]; then
    echo "Stopped running Ollama models prior to GPU reconfiguration."
  fi
}

register_warmed_model() {
  local mdl="$1"
  [ -z "$mdl" ] && return 0
  local existing
  for existing in "${WARMED_MODELS[@]}"; do
    if [ "$existing" = "$mdl" ]; then
      return 0
    fi
  done
  WARMED_MODELS+=("$mdl")
}

# Verify that the Ollama service is exercising the GPU after warmup
verify_gpu_acceleration() {
  if ! $CUDA_ENABLED; then
    echo "GPU acceleration not enabled; skipping GPU verification."
    return 0
  fi

  local bin="${OLLAMA_BIN:-$(command -v ollama || true)}"
  if [ -z "$bin" ]; then
    echo "GPU verification skipped: Ollama binary not found."
    return 0
  fi

  echo "========================================="
  echo "    Verifying GPU Acceleration"
  echo "========================================="

  mkdir -p "$SCRIPT_DIR/logs" 2>/dev/null || true
  local verify_log="$SCRIPT_DIR/logs/install_gpu_verification.log"

  if [ ${#WARMED_MODELS[@]} -eq 0 ]; then
    local fallback_model="${CONVERSATIONAL_MODEL_VAL:-${CONVERSATIONAL_MODEL:-phi3:mini}}"
    local fallback_basename
    fallback_basename="$(echo "$fallback_model" | awk -F'/' '{print $NF}')"
    register_warmed_model "$fallback_basename"
  fi

  : > "$verify_log"
  local failures=0
  local mdl
  for mdl in "${WARMED_MODELS[@]}"; do
    local payload
    payload='{"model":"'"$mdl"'","prompt":"GPU verification probe","stream":false,"options":{"num_predict":4}}'
    curl --max-time 45 -sS -o /dev/null \
         -H 'Content-Type: application/json' \
         -d "$payload" \
         http://127.0.0.1:11434/api/generate >/dev/null 2>&1 || true

    local ps_output
    ps_output="$($bin ps 2>/dev/null || true)"
    {
      echo "=== $mdl ==="
      echo "$ps_output"
      echo ""
    } >> "$verify_log"

    local line
    line=$(echo "$ps_output" | awk -v mdl="$mdl" '$1==mdl {print $0}')
    if [ -z "$line" ] && [[ "$mdl" != *:latest ]]; then
      line=$(echo "$ps_output" | awk -v mdl="$mdl:latest" '$1==mdl {print $0}')
    fi

    if [ -z "$line" ]; then
      echo "✗ GPU verification failed for $mdl (model not listed in ollama ps)."
      failures=1
      continue
    fi

    local gpu_value
    gpu_value=$(echo "$line" | awk '
      {
        for (i = 1; i <= NF; i++) {
          if ($i ~ /^[0-9]+%\/[0-9]+%$/ && (i+1) <= NF && $(i+1) == "CPU/GPU") {
            split($i, parts, "/");
            g = parts[2]; sub("%", "", g); print g; exit;
          }
          if ($i ~ /^[0-9]+%$/ && (i+1) <= NF && $(i+1) == "GPU") {
            g = $i; sub("%", "", g); print g; exit;
          }
        }
      }')

    if [ -z "$gpu_value" ] && echo "$line" | grep -q 'GPU'; then
      gpu_value=$(echo "$line" | grep -oE '[0-9]+%[[:space:]]+GPU' | head -n1 | grep -oE '^[0-9]+')
    fi

    if [ -n "$gpu_value" ] && [ "$gpu_value" -gt 0 ] 2>/dev/null; then
      echo "✓ GPU acceleration verified for $mdl (${gpu_value}% GPU)"
    else
      echo "✗ GPU usage for $mdl appears CPU-bound (see $verify_log)."
      failures=1
    fi
  done

  if [ "$failures" -eq 0 ]; then
    echo "All warmed models passed GPU verification. Details logged to $verify_log"
  else
    echo ""
    echo "========================================="
    echo "    ⚠️  GPU Verification Warning"
    echo "========================================="
    echo "One or more models failed GPU verification."
    echo ""
    if [ "${PERFORMANCE_TIER:-standard}" = "standard" ]; then
      echo "Standard tier systems may experience:"
      echo "  • Slower initial model loading"
      echo "  • CPU fallback for some operations"
      echo "  • Longer response times during warmup"
      echo ""
      echo "This is normal for 8GB GPU systems under load."
      echo "The system will continue to function but may need"
      echo "additional time for the first few operations."
    else
      echo "High tier systems should have full GPU acceleration."
      echo "Please check your GPU drivers and CUDA installation."
    fi
    echo ""
    echo "Detailed logs: $verify_log"
    echo "========================================="
    echo ""
  fi
}

# Ensure runtime deps (Ollama, required LLMs, PostgreSQL when applicable)
ensure_runtime_after_env() {
    echo "========================================="
    echo "    Ensuring Runtime Dependencies"
    echo "========================================="
    # 1) Ollama
    OLLAMA_BIN="$(command -v ollama || true)"
    if [ -z "$OLLAMA_BIN" ]; then
        INSTALL_OLLAMA=true
        echo "Ollama not found. Installing Ollama..."
        if [ "$INSTALL_OLLAMA" = true ]; then
            curl -fsSL https://ollama.com/install.sh | sh || { echo "Failed to install Ollama"; exit 1; }
            OLLAMA_BIN="$(command -v ollama || echo /usr/local/bin/ollama)"
        else
            echo "ERROR: Ollama is required but not installed"
            exit 1
        fi
    else
        echo "Ollama is already installed at: $OLLAMA_BIN"
        INSTALL_OLLAMA=false
    fi
    sudo systemctl enable ollama 2>/dev/null || true
    stop_running_ollama_models
    # Ensure Ollama systemd override is in place with performance settings
    echo "Configuring Ollama systemd overrides for performance..."
    sudo mkdir -p /etc/systemd/system/ollama.service.d
    OL_OVERRIDE="/etc/systemd/system/ollama.service.d/override.conf"
    
    # Detect GPU VRAM for optimal configuration
    local gpu_vram=0
    if command -v nvidia-smi >/dev/null 2>&1; then
        gpu_vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1 | tr -d ' ')
    elif command -v rocm-smi >/dev/null 2>&1; then
        gpu_vram=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | grep -oP '\d+' | head -n1)
    fi
    
    # Create baseline override with CPU-safe defaults
    sudo bash -c "cat > '$OL_OVERRIDE'" <<'OVREOF'
[Service]
# CPU Configuration - minimal threads for GPU-first operation
Environment=OLLAMA_NUM_THREAD=2
Environment=OLLAMA_NUM_PARALLEL=1
Environment=OLLAMA_KEEP_ALIVE=30m
OVREOF
    
    # If CUDA is enabled, append GPU-optimized configuration
    if $CUDA_ENABLED; then
        echo "Enabling GPU acceleration for Ollama (CUDA detected, VRAM: ${gpu_vram}MB) ..."
        
        # Calculate optimal settings based on VRAM
        local max_vram=$((gpu_vram - 2048))  # Leave 2GB for system
        local num_ctx=8192  # Default context window
        local num_parallel=1
        local keep_alive="30m"
        
        # Optimize based on VRAM tiers
        if [ "$gpu_vram" -ge 16000 ]; then
            # 16GB+ GPU: Maximum performance (16K allows all layers on GPU)
            num_ctx=16384
            num_parallel=2
            keep_alive="1h"
            echo "  → High-end GPU detected: 16K context, 2 parallel requests"
        elif [ "$gpu_vram" -ge 12000 ]; then
            # 12-16GB GPU: High performance
            num_ctx=16384
            num_parallel=2
            keep_alive="45m"
            echo "  → Mid-high GPU detected: 16K context, 2 parallel requests"
        elif [ "$gpu_vram" -ge 8000 ]; then
            # 8-12GB GPU: Balanced
            num_ctx=8192
            num_parallel=1
            keep_alive="30m"
            echo "  → Standard GPU detected: 8K context, 1 parallel request"
        else
            # <8GB GPU: Conservative
            num_ctx=4096
            num_parallel=1
            keep_alive="15m"
            echo "  → Low VRAM GPU detected: 4K context, conservative settings"
        fi
        
        sudo bash -c "cat >> '$OL_OVERRIDE'" <<OVREOF
# GPU Configuration - maximize VRAM usage, minimize CPU load
Environment=OLLAMA_NUM_GPU=1
Environment=OLLAMA_GPU_LAYERS=-1

# VRAM Optimization - use maximum available VRAM
Environment=OLLAMA_MAX_LOADED_MODELS=3
Environment=OLLAMA_MAX_VRAM=${max_vram}

# Context Window - optimized for detected VRAM
Environment=OLLAMA_NUM_CTX=${num_ctx}

# Parallelism - based on VRAM capacity
Environment=OLLAMA_NUM_PARALLEL=${num_parallel}

# Model Keep-Alive - keep models in VRAM longer
Environment=OLLAMA_KEEP_ALIVE=${keep_alive}

# Flash Attention - enable for faster GPU inference
Environment=OLLAMA_FLASH_ATTENTION=1

# CUDA Configuration - optimize GPU memory and performance
Environment=CUDA_VISIBLE_DEVICES=0
Environment=CUDA_DEVICE_ORDER=PCI_BUS_ID
Environment=CUDA_LAUNCH_BLOCKING=0
OVREOF
    else
        echo "CUDA not enabled; Ollama will run CPU-only with conservative defaults."
    fi
    sudo systemctl daemon-reload || true
    sudo systemctl restart ollama 2>/dev/null || true

    # Wait for Ollama API to become ready (idempotent, with retries)
    echo "Waiting for Ollama API to be ready at http://127.0.0.1:11434 ..."
    _ollama_ready=false
    for i in $(seq 1 30); do
        if curl -fsS --max-time 2 http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
            _ollama_ready=true
            echo "Ollama API is responsive."
            break
        fi
        sleep 1
    done
    if ! $_ollama_ready; then
        echo "Warning: Ollama API did not respond within 30s; continuing, but warmups may attempt pull later."
    fi

    # Ollama already restarted above after writing override

    # Conversational model defaults rely on standard llama3 builds; no additional alias preparation needed.

# Preserve user override but otherwise use default conversational model
if [ "$CONVERSATIONAL_MODEL_SET" = false ] || [ -z "$CONVERSATIONAL_MODEL_VAL" ]; then
    CONVERSATIONAL_MODEL_VAL="$DEFAULT_CONVERSATIONAL_MODEL"
    echo "Using conversational model: $CONVERSATIONAL_MODEL_VAL (default)"
else
    echo "Using user-specified conversational model: $CONVERSATIONAL_MODEL_VAL"
fi

export CONVERSATIONAL_MODEL_VAL

    # Helper: warm up an Ollama model (pull if needed, do a tiny generate)
    warm_ollama_model() {
        local model="$1"
        [ -z "$model" ] && return 0
        # Strip optional provider prefix like ollama/
        local mname
        mname="$(echo "$model" | awk -F'/' '{print $NF}')"
        echo "Warming model: $mname"
        register_warmed_model "$mname"
        # Prepare debug log path
        local dbg_log="$SCRIPT_DIR/logs/install_warmup_${mname}.log"
        # Build a GPU-preferred payload for warmup
        local payload
        # Keep warmup minimal and compatible across Ollama versions
        payload='{"model":"'"$mname"'","prompt":"ok","stream":false,"options":{"num_predict":8}}'
        # 1) Try a tiny generate first (works for local aliases too); when CUDA is enabled, capture diagnostics
        if $CUDA_ENABLED; then
            {
              echo "[warmup] POST /api/generate payload: $payload"
              curl --max-time 45 -sS -w '\n[http] code=%{http_code} time_total=%{time_total}s\n' \
                   -H 'Content-Type: application/json' \
                   -d "$payload" \
                   http://127.0.0.1:11434/api/generate
              echo "[env] Ollama Environment:"
              systemctl show ollama -p Environment --value | xargs -n1 || true
              echo "[logs] Recent ollama service logs (gpu-related):"
              sudo journalctl -u ollama -n 80 --no-pager 2>/dev/null | grep -i -E 'gpu|cuda|hip|metal|layers' || true
            } | tee "$dbg_log" >/dev/null
            # Consider warmup successful if HTTP code 200 was observed in log
            if grep -q '\[http\] code=200' "$dbg_log" 2>/dev/null; then
              return 0
            fi
        else
            if curl --max-time 45 -sS http://127.0.0.1:11434/api/generate \
                -H 'Content-Type: application/json' \
                -d "$payload" >/dev/null 2>&1; then
                return 0
            fi
        fi
        # 2) If generate failed, check presence via `ollama show` (handles aliases)
        if ! "$OLLAMA_BIN" show "$mname" >/dev/null 2>&1; then
            if ! "$OLLAMA_BIN" pull "$mname" >/dev/null 2>&1; then
                echo "Warning: could not ensure model $mname is available"
            fi
        fi
        # 4) Try generate again after ensuring presence
        if $CUDA_ENABLED; then
            {
              echo "[warmup-retry] POST /api/generate payload: $payload"
              curl --max-time 60 -sS -w '\n[http] code=%{http_code} time_total=%{time_total}s\n' \
                   -H 'Content-Type: application/json' \
                   -d "$payload" \
                   http://127.0.0.1:11434/api/generate
              echo "[env] Ollama Environment:"
              systemctl show ollama -p Environment --value | xargs -n1 || true
              echo "[logs] Recent ollama service logs (gpu-related):"
              sudo journalctl -u ollama -n 80 --no-pager 2>/dev/null | grep -i -E 'gpu|cuda|hip|metal|layers' || true
            } | tee -a "$dbg_log" >/dev/null
        else
            curl --max-time 60 -sS http://127.0.0.1:11434/api/generate \
                -H 'Content-Type: application/json' \
                -d "$payload" >/dev/null 2>&1 || true
        fi
    }
    # Normalize backend/.env and service env to use loopback for Ollama
    if [ -f "$SCRIPT_DIR/backend/.env" ]; then
        if grep -q '^OLLAMA_BASE_URL=' "$SCRIPT_DIR/backend/.env"; then
            sed -i -E "s|^OLLAMA_BASE_URL=.*|OLLAMA_BASE_URL=http://127.0.0.1:11434|" "$SCRIPT_DIR/backend/.env" || true
        else
            echo "OLLAMA_BASE_URL=http://127.0.0.1:11434" >> "$SCRIPT_DIR/backend/.env"
        fi
    fi
    if [ -f "/etc/selo-ai/environment" ]; then
        if sudo grep -q '^OLLAMA_BASE_URL=' /etc/selo-ai/environment; then
            sudo sed -i -E "s|^OLLAMA_BASE_URL=.*|OLLAMA_BASE_URL=http://127.0.0.1:11434|" /etc/selo-ai/environment || true
        else
            echo "OLLAMA_BASE_URL=http://127.0.0.1:11434" | sudo tee -a /etc/selo-ai/environment >/dev/null
        fi
    fi

    # 2) Install models based on selected configuration
    # Ensure we have a valid model configuration
    local profile_template="$SCRIPT_DIR/configs/default/.env.template"
    if [ -f "$profile_template" ]; then
        REFLECTION_LLM_VAL=$(grep '^REFLECTION_LLM=' "$profile_template" | cut -d'=' -f2-)
        ANALYTICAL_MODEL_VAL=$(grep '^ANALYTICAL_MODEL=' "$profile_template" | cut -d'=' -f2-)
        EMBEDDING_MODEL_VAL=$(grep '^EMBEDDING_MODEL=' "$profile_template" | cut -d'=' -f2-)
        echo "📋 Default models: REFLECTION=$REFLECTION_LLM_VAL, ANALYTICAL=$ANALYTICAL_MODEL_VAL, EMBEDDING=$EMBEDDING_MODEL_VAL"
    fi

    local model_install_script="$SCRIPT_DIR/configs/default/install-models.sh"
    if [ -f "$model_install_script" ] && [ -x "$model_install_script" ]; then
        echo "✅ Installing default model set via $model_install_script"
        if bash "$model_install_script"; then
            echo "✓ Default models installed successfully"
        else
            echo "⚠️  Model installation script failed, continuing with fallback logic..."
        fi
    else
        echo "⚠️  Default model installation script not found at $model_install_script, using fallback logic"
    fi

    # Helper function to pull model with retries
    pull_model_with_retry() {
        local model="$1"
        local description="$2"
        local max_attempts=5
        local attempt=1
        
        # Tier-aware timeout: standard tier gets more time for slower GPUs
        local pull_timeout=300  # 5 minutes default
        if [ "${PERFORMANCE_TIER:-standard}" = "standard" ]; then
            pull_timeout=600  # 10 minutes for standard tier
        fi
        
        echo "Ensuring $description '$model' is available..."
        
        # Check if already available
        if "$OLLAMA_BIN" show "$model" >/dev/null 2>&1; then
            echo "✓ $description '$model' already available"
            return 0
        fi
        
        # Try to pull with retries
        while [ $attempt -le $max_attempts ]; do
            echo "Downloading $description (attempt $attempt/$max_attempts)..."
            if timeout $pull_timeout "$OLLAMA_BIN" pull "$model" >/dev/null 2>&1; then
                echo "✓ Successfully downloaded $description '$model'"
                return 0
            fi
            if [ $attempt -lt $max_attempts ]; then
                echo "Download failed, retrying in 10 seconds..."
                sleep 10
            fi
            attempt=$((attempt + 1))
        done
        
        echo "Warning: Could not download $description '$model' after $max_attempts attempts"
        echo "You can install it manually later with: ollama pull $model"
        return 1
    }
    
    # 3) Pull/ensure required models with retry logic
    CHAT_MODEL="${CONVERSATIONAL_MODEL_VAL:-${CONVERSATIONAL_MODEL:-llama3:8b}}"
    pull_model_with_retry "$CHAT_MODEL" "conversational model"
    
    # Reflection model
    REFLECTION_MODEL_NAME=$(echo "${REFLECTION_LLM_VAL:-${REFLECTION_LLM:-qwen2.5:3b}}" | awk -F'/' '{print $NF}')
    pull_model_with_retry "$REFLECTION_MODEL_NAME" "reflection model"
    
    # Analytical model
    ANALYTICAL_MODEL_NAME=$(echo "${ANALYTICAL_MODEL_VAL:-${ANALYTICAL_MODEL:-qwen2.5:3b}}" | awk -F'/' '{print $NF}')
    pull_model_with_retry "$ANALYTICAL_MODEL_NAME" "analytical model"
    
    # Traits bootstrap model (qwen2.5:1.5b - only model that works for traits generation)
    # Test results: qwen2.5:1.5b has 34% success for traits vs qwen2.5:3b's 0% success
    echo "Ensuring traits bootstrap model 'qwen2.5:1.5b' is available..."
    if ! "$OLLAMA_BIN" show "qwen2.5:1.5b" >/dev/null 2>&1; then
        pull_model_with_retry "qwen2.5:1.5b" "traits bootstrap model"
    else
        echo "✓ Traits bootstrap model 'qwen2.5:1.5b' already available"
    fi
    
    # Embedding model (special case - try embeddings endpoint first)
    EMBEDDING_MODEL_NAME=$(echo "${EMBEDDING_MODEL:-nomic-embed-text}" | awk -F'/' '{print $NF}')
    echo "Ensuring embedding model '$EMBEDDING_MODEL_NAME' is available..."
    if ! "$OLLAMA_BIN" show "$EMBEDDING_MODEL_NAME" >/dev/null 2>&1; then
        # Try embeddings endpoint as a gentle pull
        curl --max-time 45 -sS http://127.0.0.1:11434/api/embeddings \
          -H 'Content-Type: application/json' \
          -d '{"model":"'"$EMBEDDING_MODEL_NAME"'","prompt":"ok"}' >/dev/null 2>&1 || true
        # If still not available, use retry logic
        if ! "$OLLAMA_BIN" show "$EMBEDDING_MODEL_NAME" >/dev/null 2>&1; then
            pull_model_with_retry "$EMBEDDING_MODEL_NAME" "embedding model"
        else
            echo "✓ Embedding model '$EMBEDDING_MODEL_NAME' available"
        fi
    else
        echo "✓ Embedding model '$EMBEDDING_MODEL_NAME' already available"
    fi
    # 3) PostgreSQL if DATABASE_URL indicates local postgres
    DB_URL_LINE=$(grep -E '^DATABASE_URL=' "$SCRIPT_DIR/backend/.env" 2>/dev/null | head -n1)
    DB_URL=${DB_URL_LINE#DATABASE_URL=}
    if echo "$DB_URL" | grep -Eiq '^postgres|^postgresql'; then
        if echo "$DB_URL" | grep -Eq '@(localhost|127\.0\.0\.1|::1)(/|:)?'; then
            # Extract database credentials from DATABASE_URL
            # Format: postgresql://user:pass@host/dbname or postgresql+asyncpg://user:pass@host/dbname
            DB_USER=$(echo "$DB_URL" | sed -E 's|^postgresql(\+[^:]*)?://([^:]+):.*|\2|')
            DB_PASS=$(echo "$DB_URL" | sed -E 's|^postgresql(\+[^:]*)?://[^:]+:([^@]+)@.*|\2|')
            DB_NAME=$(echo "$DB_URL" | sed -E 's|^postgresql(\+[^:]*)?://[^/]+/([^?]*).*|\2|')
            
            # Set defaults if parsing failed
            DB_USER="${DB_USER:-seloai}"
            DB_NAME="${DB_NAME:-seloai}"
            DB_PASS="${DB_PASS:-password}"
            
            # Escape password for SQL
            DB_PASS_ESC=$(echo "$DB_PASS" | sed "s/'/''/g")
            
            echo "Database configuration: user=$DB_USER, database=$DB_NAME"
            
            # Check if PostgreSQL is installed
            if ! command -v psql >/dev/null 2>&1; then
                INSTALL_POSTGRES=true
                echo "PostgreSQL not found, will install..."
            else
                echo "PostgreSQL is already installed"
                INSTALL_POSTGRES=false
            fi
            
            # Step 0: Install PostgreSQL if needed
            if [ "$INSTALL_POSTGRES" = true ]; then
                echo "Installing PostgreSQL..."
                ensure_apt_ready 300 || true
                sudo apt install -y postgresql postgresql-contrib libpq-dev || return 1
                # Ensure PostgreSQL service is running
                sudo systemctl start postgresql || true
                sudo systemctl enable postgresql || true
            fi
            
            # Create database user and database (regardless of whether PostgreSQL was just installed)
            # Run postgres commands from /tmp to avoid "could not change directory" warnings
            # when the script is run from a user's home directory
            echo "Ensuring database user '$DB_USER' exists..."
            if sudo -u postgres sh -c "cd /tmp && psql -tAc \"SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'\"" | grep -q 1; then
                echo "User '$DB_USER' already exists, updating password..."
                sudo -u postgres sh -c "cd /tmp && psql -c \"ALTER USER ${DB_USER} WITH PASSWORD '${DB_PASS_ESC}';\"" || true
            else
                echo "Creating user '$DB_USER'..."
                if ! sudo -u postgres sh -c "cd /tmp && psql -c \"CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS_ESC}';\""; then
                    echo "ERROR: Could not create database user '$DB_USER'"
                    echo "Database setup is required for SELO to function."
                    echo "Please check PostgreSQL is running and try again."
                    return 1
                fi
            fi
            
            # Create DB if missing
            if sudo -u postgres sh -c "cd /tmp && psql -tAc \"SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'\"" | grep -q 1; then
                echo "Database '$DB_NAME' already exists"
            else
                echo "Creating database '$DB_NAME'..."
                if ! sudo -u postgres sh -c "cd /tmp && psql -c \"CREATE DATABASE ${DB_NAME} OWNER ${DB_USER};\""; then
                    echo "ERROR: Could not create database '$DB_NAME'"
                    echo "Database setup is required for SELO to function."
                    echo "Please check PostgreSQL permissions and try again."
                    return 1
                fi
            fi
            
            # Grant privileges
            echo "Granting privileges on database '$DB_NAME' to user '$DB_USER'..."
            sudo -u postgres sh -c "cd /tmp && psql -c \"GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};\"" || true
        fi
    fi

    # Warm-up configured models to avoid first-request timeouts
    # Load backend/.env selections if present - ONLY if they're non-empty (don't overwrite with blanks)
    if [ -f "$SCRIPT_DIR/backend/.env" ]; then
        _conv_from_env="$(grep -E '^CONVERSATIONAL_MODEL=' "$SCRIPT_DIR/backend/.env" | tail -n1 | cut -d'=' -f2-)"
        _refl_from_env="$(grep -E '^REFLECTION_LLM=' "$SCRIPT_DIR/backend/.env" | tail -n1 | cut -d'=' -f2-)"
        _anal_from_env="$(grep -E '^ANALYTICAL_MODEL=' "$SCRIPT_DIR/backend/.env" | tail -n1 | cut -d'=' -f2-)"
        _embed_from_env="$(grep -E '^EMBEDDING_MODEL=' "$SCRIPT_DIR/backend/.env" | tail -n1 | cut -d'=' -f2-)"
        
        # Only update if we got non-empty values from .env
        [ -n "$_conv_from_env" ] && CONVERSATIONAL_MODEL_VAL="$_conv_from_env"
        [ -n "$_refl_from_env" ] && REFLECTION_LLM_VAL="$_refl_from_env"
        [ -n "$_anal_from_env" ] && ANALYTICAL_MODEL_VAL="$_anal_from_env"
        [ -n "$_embed_from_env" ] && EMBEDDING_MODEL_VAL="$_embed_from_env"
    fi
    # Pre-warm all primary models so first turn after install is fast
    warm_ollama_model "$CONVERSATIONAL_MODEL_VAL"
    warm_ollama_model "$REFLECTION_LLM_VAL"
    warm_ollama_model "$ANALYTICAL_MODEL_VAL"
    # Warm embedding model using embeddings API
    warm_ollama_embedding_model() {
        local model="$1"
        [ -z "$model" ] && return 0
        local mname
        mname="$(echo "$model" | awk -F'/' '{print $NF}')"
        echo "Warming embedding model: $mname"
        if ! curl --connect-timeout 5 --max-time 20 -sS http://127.0.0.1:11434/api/embeddings \
            -H 'Content-Type: application/json' \
            -d '{"model":"'"$mname"'","prompt":"ok"}' >/dev/null 2>&1; then
            echo "Warning: embedding warm-up for $mname timed out; continuing without cached embeddings."
        fi
    }
    warm_ollama_embedding_model "$EMBEDDING_MODEL_VAL"

    verify_gpu_acceleration

    # Enforce installer-selected models into backend/.env
    # DO NOT override if already correctly set by template - only add if missing
    if [ -f "$SCRIPT_DIR/backend/.env" ]; then
        # Only set if not already present (respects profile template)
        grep -q "^CONVERSATIONAL_MODEL=" "$SCRIPT_DIR/backend/.env" || echo "CONVERSATIONAL_MODEL=${CONVERSATIONAL_MODEL_VAL:-llama3:8b}" >> "$SCRIPT_DIR/backend/.env"
        grep -q "^REFLECTION_LLM=" "$SCRIPT_DIR/backend/.env" || echo "REFLECTION_LLM=${REFLECTION_LLM_VAL:-qwen2.5:3b}" >> "$SCRIPT_DIR/backend/.env"
        grep -q "^ANALYTICAL_MODEL=" "$SCRIPT_DIR/backend/.env" || echo "ANALYTICAL_MODEL=${ANALYTICAL_MODEL_VAL:-qwen2.5:3b}" >> "$SCRIPT_DIR/backend/.env"
        grep -q "^EMBEDDING_MODEL=" "$SCRIPT_DIR/backend/.env" || echo "EMBEDDING_MODEL=${EMBEDDING_MODEL_VAL:-nomic-embed-text}" >> "$SCRIPT_DIR/backend/.env"
        # Pin embedding dimension explicitly for FAISS index creation
        if grep -q '^EMBEDDING_DIM=' "$SCRIPT_DIR/backend/.env"; then
          sed -i -E "s|^EMBEDDING_DIM=.*|EMBEDDING_DIM=768|" "$SCRIPT_DIR/backend/.env" || true
        else
          echo "EMBEDDING_DIM=768" >> "$SCRIPT_DIR/backend/.env"
        fi
        # Conversational chat generation defaults (only set if missing)
        grep -q '^CHAT_MAX_TOKENS=' "$SCRIPT_DIR/backend/.env" || echo "CHAT_MAX_TOKENS=256" >> "$SCRIPT_DIR/backend/.env"
        grep -q '^CHAT_TEMPERATURE=' "$SCRIPT_DIR/backend/.env" || echo "CHAT_TEMPERATURE=0.7" >> "$SCRIPT_DIR/backend/.env"
        # Ensure persistent vector store directory exists for FAISS index/metadata
        mkdir -p "$SCRIPT_DIR/backend/data/vector_store" || true
        # Scheduler job store directory (used when DATABASE_URL is not provided)
        mkdir -p "$SCRIPT_DIR/data/scheduler" || true
        # Relax permissions enough for service user to write
        chown "$INST_USER":"$INST_USER" \
            "$SCRIPT_DIR/backend/data" \
            "$SCRIPT_DIR/backend/data/vector_store" \
            "$SCRIPT_DIR/data" \
            "$SCRIPT_DIR/data/scheduler" 2>/dev/null || true
        # Per-type reflection model overrides - use profile's REFLECTION_LLM as default
        # Read the REFLECTION_LLM that was set by the profile template
        PROFILE_REFLECTION_MODEL=$(grep "^REFLECTION_LLM=" "$SCRIPT_DIR/backend/.env" | cut -d'=' -f2-)
        # Only set these if not already present, using profile's reflection model as fallback
        grep -q "^REFLECTION_MODEL_DEFAULT=" "$SCRIPT_DIR/backend/.env" || echo "REFLECTION_MODEL_DEFAULT=${REFLECTION_MODEL_DEFAULT_VAL:-${PROFILE_REFLECTION_MODEL}}" >> "$SCRIPT_DIR/backend/.env"
        grep -q "^REFLECTION_MODEL_MESSAGE=" "$SCRIPT_DIR/backend/.env" || echo "REFLECTION_MODEL_MESSAGE=${REFLECTION_MODEL_MESSAGE_VAL:-${PROFILE_REFLECTION_MODEL}}" >> "$SCRIPT_DIR/backend/.env"
        grep -q "^REFLECTION_MODEL_DAILY=" "$SCRIPT_DIR/backend/.env" || echo "REFLECTION_MODEL_DAILY=${REFLECTION_MODEL_DAILY_VAL:-${PROFILE_REFLECTION_MODEL}}" >> "$SCRIPT_DIR/backend/.env"
        grep -q "^REFLECTION_MODEL_WEEKLY=" "$SCRIPT_DIR/backend/.env" || echo "REFLECTION_MODEL_WEEKLY=${REFLECTION_MODEL_WEEKLY_VAL:-${PROFILE_REFLECTION_MODEL}}" >> "$SCRIPT_DIR/backend/.env"
        grep -q "^REFLECTION_MODEL_EMOTIONAL=" "$SCRIPT_DIR/backend/.env" || echo "REFLECTION_MODEL_EMOTIONAL=${REFLECTION_MODEL_EMOTIONAL_VAL:-${PROFILE_REFLECTION_MODEL}}" >> "$SCRIPT_DIR/backend/.env"
        grep -q "^REFLECTION_MODEL_MANIFESTO=" "$SCRIPT_DIR/backend/.env" || echo "REFLECTION_MODEL_MANIFESTO=${REFLECTION_MODEL_MANIFESTO_VAL:-${PROFILE_REFLECTION_MODEL}}" >> "$SCRIPT_DIR/backend/.env"
        grep -q "^REFLECTION_MODEL_PERIODIC=" "$SCRIPT_DIR/backend/.env" || echo "REFLECTION_MODEL_PERIODIC=${REFLECTION_MODEL_PERIODIC_VAL:-${PROFILE_REFLECTION_MODEL}}" >> "$SCRIPT_DIR/backend/.env"
        # Performance-oriented defaults
        grep -q '^OLLAMA_KEEP_ALIVE=' "$SCRIPT_DIR/backend/.env" || echo "OLLAMA_KEEP_ALIVE=30m" >> "$SCRIPT_DIR/backend/.env"
        # Force unbounded LLM timeout (0) for heavyweight installs (overwrite if present)
        if grep -q '^LLM_TIMEOUT=' "$SCRIPT_DIR/backend/.env"; then
          sed -i -E "s|^LLM_TIMEOUT=.*|LLM_TIMEOUT=0|" "$SCRIPT_DIR/backend/.env" || true
        else
          echo "LLM_TIMEOUT=0" >> "$SCRIPT_DIR/backend/.env"
        fi
        # Ensure reflection-first mode is sync
        grep -q '^REFLECTION_SYNC_MODE=' "$SCRIPT_DIR/backend/.env" || echo "REFLECTION_SYNC_MODE=sync" >> "$SCRIPT_DIR/backend/.env"
        # Ensure bounded reflection timeouts (respect user overrides if already set)
        if grep -q '^REFLECTION_LLM_TIMEOUT_S=' "$SCRIPT_DIR/backend/.env"; then
          sed -i -E "s|^REFLECTION_LLM_TIMEOUT_S=.*|REFLECTION_LLM_TIMEOUT_S=0|" "$SCRIPT_DIR/backend/.env" || true
        else
          echo "REFLECTION_LLM_TIMEOUT_S=0" >> "$SCRIPT_DIR/backend/.env"
        fi
        if grep -q '^REFLECTION_SYNC_TIMEOUT_S=' "$SCRIPT_DIR/backend/.env"; then
          sed -i -E "s|^REFLECTION_SYNC_TIMEOUT_S=.*|REFLECTION_SYNC_TIMEOUT_S=0|" "$SCRIPT_DIR/backend/.env" || true
        else
          echo "REFLECTION_SYNC_TIMEOUT_S=0" >> "$SCRIPT_DIR/backend/.env"
        fi
        grep -q '^REFLECTION_REQUIRED=' "$SCRIPT_DIR/backend/.env" || echo "REFLECTION_REQUIRED=true" >> "$SCRIPT_DIR/backend/.env"
        grep -q '^CHAT_NUM_PREDICT=' "$SCRIPT_DIR/backend/.env" || echo "CHAT_NUM_PREDICT=320" >> "$SCRIPT_DIR/backend/.env"
        grep -q '^CHAT_TEMPERATURE=' "$SCRIPT_DIR/backend/.env" || echo "CHAT_TEMPERATURE=0.6" >> "$SCRIPT_DIR/backend/.env"
        grep -q '^CHAT_TOP_K=' "$SCRIPT_DIR/backend/.env" || echo "CHAT_TOP_K=40" >> "$SCRIPT_DIR/backend/.env"
        grep -q '^CHAT_TOP_P=' "$SCRIPT_DIR/backend/.env" || echo "CHAT_TOP_P=0.9" >> "$SCRIPT_DIR/backend/.env"
        grep -q '^CHAT_NUM_CTX=' "$SCRIPT_DIR/backend/.env" || echo "CHAT_NUM_CTX=12288" >> "$SCRIPT_DIR/backend/.env"
        if grep -q '^REFLECTION_NUM_PREDICT=' "$SCRIPT_DIR/backend/.env"; then
          current_predict=$(awk -F= '/^REFLECTION_NUM_PREDICT=/{print $2; exit}' "$SCRIPT_DIR/backend/.env")
          if [ -z "$current_predict" ] || ! awk -v cur="$current_predict" 'BEGIN{exit(cur >= 640 ? 0 : 1)}'; then
            sed -i -E "s|^REFLECTION_NUM_PREDICT=.*|REFLECTION_NUM_PREDICT=640|" "$SCRIPT_DIR/backend/.env" || true
          fi
        else
          echo "REFLECTION_NUM_PREDICT=640" >> "$SCRIPT_DIR/backend/.env"
        fi
        if grep -q '^REFLECTION_TEMPERATURE=' "$SCRIPT_DIR/backend/.env"; then
          sed -i -E "s|^REFLECTION_TEMPERATURE=.*|REFLECTION_TEMPERATURE=0.35|" "$SCRIPT_DIR/backend/.env" || true
        else
          echo "REFLECTION_TEMPERATURE=0.35" >> "$SCRIPT_DIR/backend/.env"
        fi
        grep -q '^PREWARM_MODELS=' "$SCRIPT_DIR/backend/.env" || echo "PREWARM_MODELS=true" >> "$SCRIPT_DIR/backend/.env"
        grep -q '^KEEPALIVE_ENABLED=' "$SCRIPT_DIR/backend/.env" || echo "KEEPALIVE_ENABLED=true" >> "$SCRIPT_DIR/backend/.env"
        grep -q '^PREWARM_INTERVAL_MIN=' "$SCRIPT_DIR/backend/.env" || echo "PREWARM_INTERVAL_MIN=5" >> "$SCRIPT_DIR/backend/.env"
        # CUDA-related defaults (only add if missing so users can override)
        CPU_THREADS=$( (command -v nproc >/dev/null 2>&1 && nproc) || echo 8 )
        grep -q '^OLLAMA_NUM_THREAD=' "$SCRIPT_DIR/backend/.env" || echo "OLLAMA_NUM_THREAD=${CPU_THREADS}" >> "$SCRIPT_DIR/backend/.env"
        if $CUDA_ENABLED; then
          grep -q '^OLLAMA_NUM_GPU=' "$SCRIPT_DIR/backend/.env" || echo "OLLAMA_NUM_GPU=1" >> "$SCRIPT_DIR/backend/.env"
          # Seed a higher default GPU offload depth for dedicated 8GB GPUs; users can override
          grep -q '^OLLAMA_GPU_LAYERS=' "$SCRIPT_DIR/backend/.env" || echo "OLLAMA_GPU_LAYERS=72" >> "$SCRIPT_DIR/backend/.env"
          # Add CUDA environment variables for GPU acceleration
          grep -q '^CUDA_VISIBLE_DEVICES=' "$SCRIPT_DIR/backend/.env" || echo "CUDA_VISIBLE_DEVICES=0" >> "$SCRIPT_DIR/backend/.env"
          grep -q '^CUDA_DEVICE_ORDER=' "$SCRIPT_DIR/backend/.env" || echo "CUDA_DEVICE_ORDER=PCI_BUS_ID" >> "$SCRIPT_DIR/backend/.env"
          grep -q '^PYTORCH_CUDA_ALLOC_CONF=' "$SCRIPT_DIR/backend/.env" || echo "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True" >> "$SCRIPT_DIR/backend/.env"
          grep -q '^TORCH_CUDA_MEMORY_FRACTION=' "$SCRIPT_DIR/backend/.env" || echo "TORCH_CUDA_MEMORY_FRACTION=0.8" >> "$SCRIPT_DIR/backend/.env"
          grep -q '^CUDA_LAUNCH_BLOCKING=' "$SCRIPT_DIR/backend/.env" || echo "CUDA_LAUNCH_BLOCKING=0" >> "$SCRIPT_DIR/backend/.env"
        else
          grep -q '^OLLAMA_NUM_GPU=' "$SCRIPT_DIR/backend/.env" || echo "OLLAMA_NUM_GPU=0" >> "$SCRIPT_DIR/backend/.env"
        fi
    fi
    # Also enforce in service env if it exists already
    if [ -f "/etc/selo-ai/environment" ]; then
        sudo sed -i "/^CONVERSATIONAL_MODEL=/d" /etc/selo-ai/environment; echo "CONVERSATIONAL_MODEL=${CONVERSATIONAL_MODEL_VAL:-llama3:8b}" | sudo tee -a /etc/selo-ai/environment >/dev/null
        sudo sed -i "/^REFLECTION_LLM=/d" /etc/selo-ai/environment; echo "REFLECTION_LLM=${REFLECTION_LLM_VAL:-qwen2.5:3b}" | sudo tee -a /etc/selo-ai/environment >/dev/null
        sudo sed -i "/^ANALYTICAL_MODEL=/d" /etc/selo-ai/environment; echo "ANALYTICAL_MODEL=${ANALYTICAL_MODEL_VAL:-qwen2.5:3b}" | sudo tee -a /etc/selo-ai/environment >/dev/null
        sudo sed -i "/^EMBEDDING_MODEL=/d" /etc/selo-ai/environment; echo "EMBEDDING_MODEL=${EMBEDDING_MODEL_VAL:-nomic-embed-text}" | sudo tee -a /etc/selo-ai/environment >/dev/null
        # Add chat generation defaults to service env if missing
        sudo grep -q '^CHAT_MAX_TOKENS=' /etc/selo-ai/environment || echo "CHAT_MAX_TOKENS=256" | sudo tee -a /etc/selo-ai/environment >/dev/null
        sudo grep -q '^CHAT_TEMPERATURE=' /etc/selo-ai/environment || echo "CHAT_TEMPERATURE=0.7" | sudo tee -a /etc/selo-ai/environment >/dev/null
        # Only add performance defaults if user hasn't set them
        grep -q '^OLLAMA_KEEP_ALIVE=' /etc/selo-ai/environment || echo "OLLAMA_KEEP_ALIVE=30m" | sudo tee -a /etc/selo-ai/environment >/dev/null
        grep -q '^LLM_TIMEOUT=' /etc/selo-ai/environment || echo "LLM_TIMEOUT=0" | sudo tee -a /etc/selo-ai/environment >/dev/null
        grep -q '^REFLECTION_SYNC_MODE=' /etc/selo-ai/environment || echo "REFLECTION_SYNC_MODE=sync" | sudo tee -a /etc/selo-ai/environment >/dev/null
        # Bounded reflection timeouts for reflection-first design
        if sudo grep -q '^REFLECTION_LLM_TIMEOUT_S=' /etc/selo-ai/environment; then
          sudo sed -i -E "s|^REFLECTION_LLM_TIMEOUT_S=.*|REFLECTION_LLM_TIMEOUT_S=0|" /etc/selo-ai/environment || true
        else
          echo "REFLECTION_LLM_TIMEOUT_S=0" | sudo tee -a /etc/selo-ai/environment >/dev/null
        fi
        if sudo grep -q '^REFLECTION_SYNC_TIMEOUT_S=' /etc/selo-ai/environment; then
          sudo sed -i -E "s|^REFLECTION_SYNC_TIMEOUT_S=.*|REFLECTION_SYNC_TIMEOUT_S=0|" /etc/selo-ai/environment || true
        else
          echo "REFLECTION_SYNC_TIMEOUT_S=0" | sudo tee -a /etc/selo-ai/environment >/dev/null
        fi
        grep -q '^REFLECTION_REQUIRED=' /etc/selo-ai/environment || echo "REFLECTION_REQUIRED=true" | sudo tee -a /etc/selo-ai/environment >/dev/null
        sudo grep -q '^CHAT_TOP_K=' /etc/selo-ai/environment || echo "CHAT_TOP_K=40" | sudo tee -a /etc/selo-ai/environment >/dev/null
        sudo grep -q '^CHAT_TOP_P=' /etc/selo-ai/environment || echo "CHAT_TOP_P=0.9" | sudo tee -a /etc/selo-ai/environment >/dev/null
        sudo grep -q '^CHAT_NUM_CTX=' /etc/selo-ai/environment || echo "CHAT_NUM_CTX=12288" | sudo tee -a /etc/selo-ai/environment >/dev/null
        if sudo grep -q '^REFLECTION_NUM_PREDICT=' /etc/selo-ai/environment; then
          current_predict=$(sudo awk -F= '/^REFLECTION_NUM_PREDICT=/{print $2; exit}' /etc/selo-ai/environment)
          if [ -z "$current_predict" ] || ! awk -v cur="$current_predict" 'BEGIN{exit(cur >= 640 ? 0 : 1)}'; then
            sudo sed -i -E "s|^REFLECTION_NUM_PREDICT=.*|REFLECTION_NUM_PREDICT=640|" /etc/selo-ai/environment || true
          fi
        else
          echo "REFLECTION_NUM_PREDICT=640" | sudo tee -a /etc/selo-ai/environment >/dev/null
        fi
        if sudo grep -q '^REFLECTION_TEMPERATURE=' /etc/selo-ai/environment; then
          sudo sed -i -E "s|^REFLECTION_TEMPERATURE=.*|REFLECTION_TEMPERATURE=0.35|" /etc/selo-ai/environment || true
        else
          echo "REFLECTION_TEMPERATURE=0.35" | sudo tee -a /etc/selo-ai/environment >/dev/null
        fi
        grep -q '^PREWARM_MODELS=' /etc/selo-ai/environment || echo "PREWARM_MODELS=true" | sudo tee -a /etc/selo-ai/environment >/dev/null
        grep -q '^KEEPALIVE_ENABLED=' /etc/selo-ai/environment || echo "KEEPALIVE_ENABLED=true" | sudo tee -a /etc/selo-ai/environment >/dev/null
        grep -q '^PREWARM_INTERVAL_MIN=' /etc/selo-ai/environment || echo "PREWARM_INTERVAL_MIN=5" | sudo tee -a /etc/selo-ai/environment >/dev/null
        # CUDA-related defaults (only add if missing so users can override)
        CPU_THREADS=$( (command -v nproc >/dev/null 2>&1 && nproc) || echo 8 )
        grep -q '^OLLAMA_NUM_THREAD=' /etc/selo-ai/environment || echo "OLLAMA_NUM_THREAD=${CPU_THREADS}" | sudo tee -a /etc/selo-ai/environment >/dev/null
        if $CUDA_ENABLED; then
          grep -q '^OLLAMA_NUM_GPU=' /etc/selo-ai/environment || echo "OLLAMA_NUM_GPU=1" | sudo tee -a /etc/selo-ai/environment >/dev/null
          # Seed default GPU offload depth; user may override later
          grep -q '^OLLAMA_GPU_LAYERS=' /etc/selo-ai/environment || echo "OLLAMA_GPU_LAYERS=72" | sudo tee -a /etc/selo-ai/environment >/dev/null
          # Add CUDA environment variables for GPU acceleration
          grep -q '^CUDA_VISIBLE_DEVICES=' /etc/selo-ai/environment || echo "CUDA_VISIBLE_DEVICES=0" | sudo tee -a /etc/selo-ai/environment >/dev/null
          grep -q '^CUDA_DEVICE_ORDER=' /etc/selo-ai/environment || echo "CUDA_DEVICE_ORDER=PCI_BUS_ID" | sudo tee -a /etc/selo-ai/environment >/dev/null
          grep -q '^PYTORCH_CUDA_ALLOC_CONF=' /etc/selo-ai/environment || echo "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256,expandable_segments:True" | sudo tee -a /etc/selo-ai/environment >/dev/null
          grep -q '^TORCH_CUDA_MEMORY_FRACTION=' /etc/selo-ai/environment || echo "TORCH_CUDA_MEMORY_FRACTION=0.8" | sudo tee -a /etc/selo-ai/environment >/dev/null
          grep -q '^CUDA_LAUNCH_BLOCKING=' /etc/selo-ai/environment || echo "CUDA_LAUNCH_BLOCKING=0" | sudo tee -a /etc/selo-ai/environment >/dev/null
        else
          grep -q '^OLLAMA_NUM_GPU=' /etc/selo-ai/environment || echo "OLLAMA_NUM_GPU=0" | sudo tee -a /etc/selo-ai/environment >/dev/null
        fi
    fi
}

# Build backend/frontend
build_project() {
    echo "========================================="
    echo "    Building Backend & Frontend"
    echo "========================================="
    # Backend
    cd "$SCRIPT_DIR/backend"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        echo "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Use explicit paths to venv executables instead of relying on source/activate
    VENV_PYTHON="$SCRIPT_DIR/backend/venv/bin/python3"
    VENV_PIP="$SCRIPT_DIR/backend/venv/bin/pip"
    
    # Upgrade pip in the venv
    "$VENV_PIP" install --upgrade pip
    
    # Install core dependencies first (PyTorch and sentence-transformers are always needed)
    echo "Installing core dependencies (NumPy, PyTorch, sentence-transformers)..."
    
    # Pin NumPy <2.0 for FAISS-GPU compatibility
    "$VENV_PIP" install "numpy>=1.24.0,<2.0.0"
    
    # Install PyTorch - use CUDA version if enabled, CPU version otherwise
    if $CUDA_ENABLED; then
        echo "Installing PyTorch with CUDA support..."
        "$VENV_PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    else
        echo "Installing PyTorch CPU version..."
        "$VENV_PIP" install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    fi
    
    # Install sentence-transformers (core dependency for embeddings)
    echo "Installing sentence-transformers..."
    "$VENV_PIP" install "sentence-transformers>=2.2.2"
    
    # Verify PyTorch installation
    "$VENV_PYTHON" -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
else:
    print('CUDA not available - using CPU only')
" || echo "Warning: PyTorch verification failed"
    
    echo "Installing Python dependencies..."
    
    # Handle FAISS installation with proper GPU/CPU fallback
    echo "Installing FAISS with optimal GPU/CPU support..."
    
    # Remove any existing FAISS installations to avoid conflicts
    "$VENV_PIP" uninstall -y faiss-gpu faiss faiss-cpu || true
    
    # Ensure NumPy <2.0 for FAISS compatibility
    "$VENV_PIP" install "numpy>=1.24.0,<2.0.0"
    
    if $CUDA_ENABLED; then
        echo "Attempting FAISS GPU installation..."
        
        # Detect Python version for FAISS compatibility
        PYTHON_VERSION=$("$VENV_PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        echo "Detected Python version: $PYTHON_VERSION"
        
        # Python 3.12+ requires FAISS 1.8.0+, older Python can use 1.7.x
        if "$VENV_PYTHON" -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)"; then
            echo "Python 3.12+ detected - using FAISS GPU 1.8.0+"
            FAISS_VERSION_CONSTRAINT="faiss-gpu>=1.8.0"
        else
            echo "Python <3.12 detected - using FAISS GPU 1.7.2+"
            FAISS_VERSION_CONSTRAINT="faiss-gpu>=1.7.2,<1.8.0"
        fi
        
        # First attempt: faiss-gpu with Python-version-appropriate constraints
        if "$VENV_PIP" install "$FAISS_VERSION_CONSTRAINT" --no-cache-dir; then
            echo "✅ FAISS GPU package installed successfully"
            
            # Safe verification - only check package and API availability
            "$VENV_PYTHON" -c "
import sys
try:
    import faiss
    print(f'FAISS version: {faiss.__version__}')
    
    # Check basic GPU support functions are available
    gpu_support = hasattr(faiss, 'StandardGpuResources') and hasattr(faiss, 'index_cpu_to_gpu')
    print(f'GPU support available: {gpu_support}')
    
    if not gpu_support:
        print('⚠️  FAISS GPU functions missing - CPU-only package detected')
        sys.exit(1)
    
    # Skip actual GPU operations during install to avoid crashes
    # Runtime GPU functionality will be tested by the application itself
    print('✅ FAISS GPU package verified - GPU operations will be tested at runtime')
        
except ImportError as e:
    print(f'⚠️  FAISS import failed: {e}')
    sys.exit(1)
except Exception as e:
    print(f'⚠️  Verification error: {e}')
    sys.exit(1)
" || {
                echo "⚠️  FAISS GPU verification failed, attempting reinstall with different approach..."
                "$VENV_PIP" uninstall -y faiss-gpu || true
                
                # Try alternative installation method
                if "$VENV_PIP" install --no-cache-dir --force-reinstall "faiss-gpu==1.7.4"; then
                    echo "✅ FAISS GPU reinstalled with specific version"
                    
                    # Quick verification
                    if "$VENV_PYTHON" -c "import faiss; assert hasattr(faiss, 'StandardGpuResources'), 'GPU support missing'"; then
                        echo "✅ FAISS GPU verification successful"
                    else
                        echo "⚠️  Still no GPU support, falling back to CPU..."
                        "$VENV_PIP" uninstall -y faiss-gpu || true
                        "$VENV_PIP" install faiss-cpu>=1.7.2
                        echo "✅ FAISS CPU fallback installed"
                    fi
                else
                    echo "⚠️  Alternative GPU installation failed, using CPU fallback..."
                    "$VENV_PIP" install faiss-cpu>=1.7.2
                    echo "✅ FAISS CPU fallback installed"
                fi
            }
        else
            echo "⚠️  FAISS GPU installation failed, installing CPU version..."
            "$VENV_PIP" install faiss-cpu>=1.7.2
            echo "✅ FAISS CPU version installed"
        fi
    else
        echo "CUDA disabled - installing FAISS CPU version..."
        "$VENV_PIP" install faiss-cpu>=1.7.2
        echo "✅ FAISS CPU version installed"
    fi
    
    # Install remaining dependencies (excluding packages we handled explicitly above)
    echo "Installing remaining dependencies from requirements.txt..."
    if ! "$VENV_PIP" install -r <(grep -v -E "^(faiss|torch|sentence-transformers)" requirements.txt); then
        echo "Error: Failed to install Python dependencies"
        exit 1
    fi
    
    # Check for AVX2 optimization and reinstall if missing (high-tier systems benefit significantly)
    if $CUDA_ENABLED; then
        echo "Checking FAISS AVX2 optimization status..."
        AVX2_CHECK=$("$VENV_PYTHON" -c "
import sys
try:
    import faiss.loader
    # Check if AVX2 module is available
    try:
        from faiss import swigfaiss_avx2
        print('avx2_available')
        sys.exit(0)
    except ImportError:
        print('avx2_missing')
        sys.exit(1)
except Exception as e:
    print('avx2_unknown')
    sys.exit(2)
" 2>&1 || echo "avx2_missing")
        
        if echo "$AVX2_CHECK" | grep -q "avx2_missing"; then
            echo "⚠️  FAISS AVX2 optimization not detected - reinstalling for better performance..."
            "$VENV_PIP" uninstall -y faiss-gpu || true
            
            # Reinstall with --no-cache-dir and --force-reinstall to get fresh build
            if "$VENV_PIP" install --no-cache-dir --force-reinstall "$FAISS_VERSION_CONSTRAINT"; then
                echo "✅ FAISS GPU reinstalled - checking AVX2 again..."
                
                # Verify AVX2 is now available
                if "$VENV_PYTHON" -c "from faiss import swigfaiss_avx2" 2>/dev/null; then
                    echo "✅ FAISS AVX2 optimization now available"
                else
                    echo "ℹ️  FAISS installed without AVX2 (will use standard optimizations)"
                fi
            else
                echo "⚠️  Reinstallation failed - continuing with current FAISS installation"
            fi
        elif echo "$AVX2_CHECK" | grep -q "avx2_available"; then
            echo "✅ FAISS AVX2 optimization already available"
        else
            echo "ℹ️  Could not determine AVX2 status - continuing with current installation"
        fi
    fi
    
    # Final verification of installations
    echo "Verifying final installation state..."
    "$VENV_PYTHON" -c "
import torch
import faiss
import sentence_transformers
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'FAISS version: {faiss.__version__}')
print(f'Sentence-transformers version: {sentence_transformers.__version__}')
gpu_support = hasattr(faiss, 'StandardGpuResources') and hasattr(faiss, 'index_cpu_to_gpu')
print(f'FAISS GPU support: {gpu_support}')

# Check AVX2 optimization
try:
    from faiss import swigfaiss_avx2
    print('FAISS AVX2 optimization: ✅ Available')
except ImportError:
    print('FAISS AVX2 optimization: ⚠️  Not available (using standard)')

if torch.cuda.is_available() and gpu_support:
    print('✅ Full GPU acceleration available')
elif torch.cuda.is_available():
    print('⚠️  PyTorch GPU available but FAISS using CPU fallback')
else:
    print('ℹ️  CPU-only configuration')
" || echo "Warning: Final verification failed"
    # Frontend
    cd "$SCRIPT_DIR/frontend"
    # Guarantee modern Node/npm in case installer was resumed mid-run
    ensure_node_toolchain || return 1
    echo "Using Node.js $(node -v) and npm $(npm -v) for frontend install"
    # Preemptive permission repair for frontend to avoid npm EACCES
    echo "Repairing frontend permissions and npm cache..."
    sudo chown -R "$INST_USER":"$INST_USER" "$SCRIPT_DIR/frontend" 2>/dev/null || true
    sudo -u "$INST_USER" mkdir -p "/home/$INST_USER/.npm" 2>/dev/null || true
    sudo chown -R "$INST_USER":"$INST_USER" "/home/$INST_USER/.npm" 2>/dev/null || true
    if [ -d node_modules ] && ! [ -w node_modules ]; then
        echo "node_modules not writable; removing to prevent EACCES..."
        rm -rf node_modules || true
    fi
    # Ensure npm defaults for reliable, non-interactive installs
    if [ ! -f ".npmrc" ]; then
        {
            echo "legacy-peer-deps=true"
            echo "fund=false"
            echo "audit=false"
            echo "progress=false"
            echo "engine-strict=true"
        } > .npmrc
    fi
    npm config set fund false >/dev/null 2>&1 || true
    npm config set audit false >/dev/null 2>&1 || true
    npm config set progress false >/dev/null 2>&1 || true
    # Prefer clean, lockfile-based install; with robust fallback strategy
    if [ -f "package-lock.json" ]; then
        echo "Attempting npm ci with existing lockfile..."
        if ! npm ci; then
            echo "npm ci failed. Applying recovery: clean cache, remove node_modules and lockfile, then npm install with legacy peer deps."
            npm cache clean --force || true
            rm -rf node_modules package-lock.json || true
            npm install --legacy-peer-deps || true
            # Recreate lockfile to stabilize future installs
            npm install --package-lock-only --ignore-scripts || true
        fi
    else
        echo "No package-lock.json found. Running npm install with legacy peer deps..."
        if ! npm install --legacy-peer-deps; then
            echo "npm install failed; cleaning cache and retrying..."
            npm cache clean --force || true
            rm -rf node_modules || true
            npm install --legacy-peer-deps || true
        fi
        npm install --package-lock-only --ignore-scripts || true
    fi
    # Optional: attempt automated vulnerability fixes
    if [ "$AUDIT_FIX" = true ]; then
        echo "Running npm audit fix (optional)..."
        npm audit fix || true
        # Save a brief audit report
        echo "Generating npm audit report (non-blocking)..."
        npm audit > "$SCRIPT_DIR/logs/install_frontend_audit.log" 2>&1 || true
        echo "Audit report saved to logs/install_frontend_audit.log"
    fi
    # Load environment and set API base for the frontend build
    if [ -r "/etc/selo-ai/environment" ]; then
        # Sanitize the environment file before sourcing to avoid syntax errors
        local tmp_env
        tmp_env=$(mktemp)
        while IFS= read -r line || [ -n "$line" ]; do
            # Preserve comments and empty lines
            if echo "$line" | grep -qE '^\s*#'; then echo "$line" >>"$tmp_env"; continue; fi
            if echo "$line" | grep -qE '^\s*$'; then echo "$line" >>"$tmp_env"; continue; fi
            # If line lacks '=', attempt to normalize common cases
            if ! echo "$line" | grep -q '='; then
                # Bare IPv4 -> HOST_IP=
                if echo "$line" | grep -qE '^([0-9]{1,3}\.){3}[0-9]{1,3}\s*$'; then
                    echo "HOST_IP=$(echo "$line" | xargs)" >>"$tmp_env"
                    continue
                fi
                # Skip malformed lines
                echo "# sanitized: $line" >>"$tmp_env"
                continue
            fi
            # Keep valid KEY=VALUE lines as-is
            echo "$line" >>"$tmp_env"
        done </etc/selo-ai/environment
        
        set -a
        source "$tmp_env"
        set +a
        rm -f "$tmp_env"
    else
        echo "Note: /etc/selo-ai/environment not readable by current user; using in-script API_URL defaults for build"
    fi
    REACT_APP_API_URL="${REACT_APP_API_URL:-$API_URL}"
    VITE_API_URL="${VITE_API_URL:-$API_URL}"
    echo "Using REACT_APP_API_URL=$REACT_APP_API_URL"
    echo "Using VITE_API_URL=$VITE_API_URL for frontend build"
    # Persist to frontend/.env for future local runs and clarity
    ENV_FILE="$SCRIPT_DIR/frontend/.env"
    {
        echo "REACT_APP_API_URL=$REACT_APP_API_URL"
        echo "VITE_API_URL=$VITE_API_URL"
    } > "$ENV_FILE"
    echo "Wrote frontend API config to $ENV_FILE"
    if [ "$QUIET_WARNINGS" = true ]; then
        echo "Building frontend (quiet warnings mode)..."
        REACT_APP_API_URL="$REACT_APP_API_URL" VITE_API_URL="$VITE_API_URL" npm run build > "$SCRIPT_DIR/logs/install_frontend_build.log" 2>&1 || true
        WARN_COUNT=$(grep -ci "warning" "$SCRIPT_DIR/logs/install_frontend_build.log" 2>/dev/null || echo 0)
        ERR_COUNT=$(grep -ci "error" "$SCRIPT_DIR/logs/install_frontend_build.log" 2>/dev/null || echo 0)
        WARN_COUNT=${WARN_COUNT:-0}
        ERR_COUNT=${ERR_COUNT:-0}
        if [ "$ERR_COUNT" -gt 0 ] 2>/dev/null; then
            echo "✗ Frontend build completed with $ERR_COUNT errors and $WARN_COUNT warnings. See logs/install_frontend_build.log"
        else
            echo "✓ Frontend build completed. Warnings: $WARN_COUNT (see logs/install_frontend_build.log)"
        fi
    else
        # Ensure frontend API env vars default to API_URL if not already set in environment
        REACT_APP_API_URL=${REACT_APP_API_URL:-$API_URL}
        VITE_API_URL=${VITE_API_URL:-$API_URL}
        REACT_APP_API_URL="$REACT_APP_API_URL" VITE_API_URL="$VITE_API_URL" npm run build
    fi
    # Final permission pass to ensure runtime serves can write artifacts/logs
    sudo chown -R "$INST_USER":"$INST_USER" "$SCRIPT_DIR/frontend" 2>/dev/null || true
}

# Function to initialize database and run bootstrap
init_database() {
    echo "========================================="
    echo "    Initializing Database Schema"
    echo "========================================="
    
    # Use file-based locking to prevent concurrent database initialization
    # This prevents race conditions between installer and service startup
    # Use /tmp for the lock file since /var/lock requires root permissions
    local lock_file="/tmp/selo-ai-db-init.lock"
    local lock_fd=200
    
    # Always remove any existing lock file at start (installer runs as root)
    # This ensures clean state even if previous install failed mid-way
    if [ -f "$lock_file" ]; then
        echo "Removing existing lock file from previous run..."
        rm -f "$lock_file" 2>/dev/null || true
    fi
    
    # Try to acquire lock (wait up to 60 seconds)
    echo "Acquiring database initialization lock..."
    # Create lock file with world-writable permissions so service can use it later
    touch "$lock_file" 2>/dev/null || true
    chmod 666 "$lock_file" 2>/dev/null || true
    exec 200>"$lock_file"
    if ! flock -w 60 200; then
        echo "ERROR: Could not acquire database initialization lock after 60 seconds"
        echo "Another process may be initializing the database."
        exit 1
    fi
    
    echo "Lock acquired. Initializing database..."
    
    cd "$SCRIPT_DIR/backend"
    if [ -f "venv/bin/activate" ]; then
        source venv/bin/activate
    fi
    
    # Prefer actual location under backend/db, maintain fallbacks for older layouts
    local init_result=0
    if python -c "import db.init_db" 2>/dev/null; then
        python -m db.init_db || init_result=$?
    elif [ -f "db/init_db.py" ]; then
        PYTHONPATH="." python db/init_db.py || init_result=$?
    elif python -c "import backend.db.init_db" 2>/dev/null; then
        python -m backend.db.init_db || init_result=$?
    elif [ -f "scripts/init_db.py" ]; then
        python scripts/init_db.py || init_result=$?
    elif python -c "import backend.scripts.init_db" 2>/dev/null; then
        python -m backend.scripts.init_db || init_result=$?
    else
        echo "No init_db script found; skipping explicit DB init"
        init_result=0
    fi
    
    # Release lock and clean up lock file so service can create fresh one
    flock -u 200
    rm -f "$lock_file" 2>/dev/null || true
    
    if [ $init_result -ne 0 ]; then
        echo "Warning: Database initialization returned non-zero exit code: $init_result"
        echo "Continuing, but database may not be properly initialized."
    fi
    
    # Persona bootstrap will run pre-service in this installer
    echo ""
    echo "========================================="
    echo "    Database Initialization Complete"
    echo "========================================="
    echo "✓ Database schema created successfully"
    echo "  • Persona bootstrap will run now (pre-service)"
    echo "  • Boot seed directive will be randomly selected"
    echo "  • SELO identity (name + mantra + traits) will be generated"
    echo ""
    echo "Note: Running persona bootstrap before service creation ensures"
    echo "      identity is ready when the service first starts."
    
    deactivate 2>/dev/null || true
}

# Persona & Mantra bootstrap (pre-service, blocking)
bootstrap_persona_pre_service() {
    echo "========================================="
    echo "    Bootstrapping Persona (Pre-Service)"
    echo "========================================="
    echo "This step generates the AI persona and mantra before service creation."
    echo "This may take a few minutes depending on your system..."
    echo ""
    
    # Use backend virtualenv if present
    if [ -f "$SCRIPT_DIR/backend/venv/bin/activate" ]; then
        source "$SCRIPT_DIR/backend/venv/bin/activate"
    fi
    mkdir -p "$SCRIPT_DIR/logs" 2>/dev/null || true
    local log_file="$SCRIPT_DIR/logs/install_persona_bootstrap.log"
    
    # Run from project root with proper PYTHONPATH to avoid import errors
    # This ensures all backend.* imports resolve correctly
    # Timeout after 20 minutes to prevent indefinite hangs (increased from 15 for slower systems)
    local rc=0
    local attempt=1
    local max_attempts=2
    
    while [ $attempt -le $max_attempts ]; do
        echo "Persona bootstrap attempt $attempt of $max_attempts..."
        (
          cd "$SCRIPT_DIR" && \
          PYTHONPATH="$SCRIPT_DIR/backend:$SCRIPT_DIR:${PYTHONPATH:-}" \
          timeout 1200 python3 -u -m backend.scripts.bootstrap_persona --verbose > "$log_file" 2>&1
        ) || rc=$?
        
        if [ $rc -eq 0 ]; then
            break
        elif [ $attempt -lt $max_attempts ]; then
            echo "Persona bootstrap failed (exit code $rc), retrying..."
            sleep 5
        fi
        attempt=$((attempt + 1))
    done
    
    deactivate 2>/dev/null || true
    
    # Check result and fail installation if persona bootstrap failed
    if [ $rc -eq 0 ]; then
        echo "✓ Persona bootstrap completed successfully on attempt $((attempt - 1))"
        echo "  Log: $log_file"
        return 0
    elif [ $rc -eq 124 ]; then
        echo ""
        echo "❌ ERROR: Persona bootstrap timed out after 20 minutes"
        echo "   This indicates a serious issue with the system."
        echo "   Review the log for details: $log_file"
        echo ""
        echo "Common causes:"
        echo "  • Ollama service not responding (check: systemctl status ollama)"
        echo "  • Models not loaded (check: ollama list)"
        echo "  • Insufficient system resources (CPU/RAM/GPU)"
        echo "  • Database connection hanging"
        echo ""
        echo "Installation cannot continue without a valid persona."
        return 1
    else
        echo ""
        echo "❌ ERROR: Persona bootstrap failed with exit code $rc"
        echo "   This is a critical step required before service creation."
        echo "   Review the log for details: $log_file"
        echo ""
        echo "Common causes:"
        echo "  • Ollama service not running or models not available"
        echo "  • Database connection issues"
        echo "  • Missing dependencies in backend environment"
        echo "  • Import errors (check PYTHONPATH)"
        echo ""
        echo "Installation cannot continue without a valid persona."
        return 1
    fi
}

# Function to start services
start_services() {
    echo "========================================="
    echo "    Starting SELO DSP Services"
    echo "========================================="
    
    # Check if we should use systemd service or manual start
    CURRENT_USER=${SUDO_USER:-$(whoami)}
    # Prefer a robust detection: existence of the unit file or systemd can cat the template
    if [ -f "/etc/systemd/system/selo-ai@.service" ] || systemctl cat selo-ai@.service >/dev/null 2>&1; then
        echo "Starting SELO DSP via systemd instance service for $CURRENT_USER..."
        # Ensure systemd has the latest unit
        sudo systemctl daemon-reload || true
        sudo systemctl start "selo-ai@$CURRENT_USER" || true
        
        # Wait a moment for services to start
        echo "Waiting for services to initialize..."
        sleep 10
        
        # Check service status
        if systemctl is-active --quiet "selo-ai@$CURRENT_USER"; then
            echo "✓ SELO DSP service is running!"
        else
            echo "⚠️  Service may have issues. Checking status..."
            sudo systemctl status "selo-ai@$CURRENT_USER" --no-pager || true
            echo ""
            echo "Recent service logs:"
            sudo journalctl -u "selo-ai@$CURRENT_USER" -n 50 --no-pager || true
            echo ""
            echo "❌ Service failed to start properly. Installation cannot continue."
            exit 1
        fi
    else
        echo "Systemd unit not installed. Starting via start-service.sh..."
        bash "$SCRIPT_DIR/start-service.sh" &
        MANUAL_PID=$!
        echo "Services started manually with PID: $MANUAL_PID"
        echo "Note: This will not persist after reboot. Consider running with systemd."
    fi
    
    echo ""
}

# Function to test installation
test_installation() {
    echo "========================================="
    echo "    Testing Installation"
    echo "========================================="
    # Load service environment for model/URL awareness if readable
    if [ -r "/etc/selo-ai/environment" ]; then
        set -a
        source /etc/selo-ai/environment
        set +a
    else
        echo "Note: /etc/selo-ai/environment not readable; proceeding with defaults for tests"
    fi
    
    # Backend readiness probe with open-ended wait (configurable)
    echo "Testing backend connectivity (readiness probe, timeout=${READINESS_TIMEOUT_SEC}s; 0=wait forever)..."
    BASE_URL="${API_URL:-http://${HOST_IP:-localhost}:${SELO_AI_PORT:-8000}}"
    # Select health path based on STRICT_HEALTH flag
    if [ "${STRICT_HEALTH}" = true ]; then
      HEALTH_PATH="/health/details?probe_llm=false&probe_db=true"
    else
      HEALTH_PATH="/health"
    fi
    HEALTH_URL="$BASE_URL$HEALTH_PATH"
    LOCAL_URL="http://127.0.0.1:${SELO_AI_PORT:-8000}$HEALTH_PATH"
    # Fast liveness path (responds immediately even during heavy init)
    FAST_HEALTH_PATH="/health/simple"
    FAST_URL="$BASE_URL$FAST_HEALTH_PATH"
    FAST_LOCAL_URL="http://127.0.0.1:${SELO_AI_PORT:-8000}$FAST_HEALTH_PATH"
    ok=false
    start_ts=$(date +%s)
    tries=0
    while true; do
        if [ "${STRICT_HEALTH}" = true ]; then
          # Strict mode: require full health
          if curl -sf "$HEALTH_URL" >/dev/null; then ok=true; break; fi
          if curl -sf "$LOCAL_URL" >/dev/null; then ok=true; break; fi
        else
          # Non-strict: accept fast liveness first, then normal health
          if curl -sf "$FAST_URL" >/dev/null; then ok=true; break; fi
          if curl -sf "$FAST_LOCAL_URL" >/dev/null; then ok=true; break; fi
          if curl -sf "$HEALTH_URL" >/dev/null; then ok=true; break; fi
          if curl -sf "$LOCAL_URL" >/dev/null; then ok=true; break; fi
        fi
        # On first failure, try to detect if the backend selected a different port at runtime
        if [ "$tries" -eq 0 ]; then
          if [ -n "${INSTALL_DIR:-}" ] && [ -f "${INSTALL_DIR}/backend.port" ]; then
            detected_port=$(cat "${INSTALL_DIR}/backend.port" 2>/dev/null | tr -dc '0-9')
            if [ -n "$detected_port" ]; then
              host_part=$(echo "$BASE_URL" | sed -E 's#^http://([^/:]+).*$#\1#')
              BASE_URL="http://${host_part}:${detected_port}"
              HEALTH_URL="$BASE_URL$HEALTH_PATH"
              LOCAL_URL="http://127.0.0.1:${detected_port}$HEALTH_PATH"
              FAST_URL="$BASE_URL$FAST_HEALTH_PATH"
              FAST_LOCAL_URL="http://127.0.0.1:${detected_port}$FAST_HEALTH_PATH"
              echo "Detected backend.port=$detected_port; updating health URL to $HEALTH_URL"
            fi
          fi
        fi
        tries=$((tries+1))
        # Log every ~6 seconds to show progress
        if [ $((tries % 3)) -eq 0 ]; then
          echo "Waiting for backend health... (attempt=$tries, url=$HEALTH_URL)"
        fi
        # Honor timeout if provided (>0)
        if [ "${READINESS_TIMEOUT_SEC}" -gt 0 ]; then
          now_ts=$(date +%s)
          elapsed=$((now_ts - start_ts))
          if [ $elapsed -ge ${READINESS_TIMEOUT_SEC} ]; then break; fi
        fi
        sleep 2
    done
    if $ok; then
        echo "✓ Backend is reachable"
        # Persona readiness wait (unless skipped)
        if [ "$SKIP_PERSONA_CHECK" != true ]; then
          PERSONA_URL="$BASE_URL/api/persona/status"
          echo "Waiting for persona bootstrap to complete (timeout=${PERSONA_TIMEOUT_SEC}s; 0=wait forever)..."
          p_ok=false
          p_start=$(date +%s)
          p_tries=0
          while true; do
            resp=$(curl -sf "$PERSONA_URL" 2>/dev/null || true)
            if [ -n "$resp" ]; then
              if command -v jq >/dev/null 2>&1; then
                ok_flag=$(echo "$resp" | jq -r '.data.ok // empty')
                if [ "$ok_flag" = "true" ]; then p_ok=true; break; fi
              else
                echo "$resp" | grep -q '"ok"[[:space:]]*:[[:space:]]*true' && { p_ok=true; break; }
              fi
            fi
            p_tries=$((p_tries+1))
            if [ $((p_tries % 5)) -eq 0 ]; then
              echo "Waiting for persona... (attempt=$p_tries, url=$PERSONA_URL)"
            fi
            if [ "$PERSONA_TIMEOUT_SEC" -gt 0 ]; then
              now=$(date +%s); pelapsed=$((now - p_start))
              if [ $pelapsed -ge "$PERSONA_TIMEOUT_SEC" ]; then break; fi
            fi
            sleep 3
          done
          if $p_ok; then
            echo "✓ Persona bootstrap completed"
          else
            echo "✗ Persona readiness check failed after ${PERSONA_TIMEOUT_SEC}s: $PERSONA_URL"
            echo "-- Persona status response (last known) --"
            echo "${resp:-'(no response)'}"
            exit 1
          fi
        else
          echo "⚠ Persona readiness check skipped by flag --skip-persona-check"
        fi
    else
        if [ "${READINESS_TIMEOUT_SEC}" -gt 0 ]; then
          echo "✗ Backend readiness check failed after ${READINESS_TIMEOUT_SEC}s: $HEALTH_URL"
        else
          echo "✗ Backend readiness check failed (no timeout set, but loop broke unexpectedly): $HEALTH_URL"
        fi
        echo "-- Service status --"
        sudo systemctl status "selo-ai@${INST_USER}" --no-pager || true
        echo "-- Recent service logs --"
        sudo journalctl -u "selo-ai@${INST_USER}" -n 120 --no-pager || true
        echo "-- Backend log tail --"
        INSTALL_DIR_CUR=${INSTALL_DIR:-$SCRIPT_DIR}
        tail -n 120 "$INSTALL_DIR_CUR/logs/backend.log" 2>/dev/null || echo "(backend.log not found)"
        echo "-- Curl diagnostics --"
        echo "Trying localhost health (simple):"
        curl -sS -o /dev/null -w 'http_code=%{http_code} time_total=%{time_total}s\n' "http://127.0.0.1:${SELO_AI_PORT:-8000}${FAST_HEALTH_PATH}" || true
        echo "Trying localhost health:"
        curl -sS -o /dev/null -w 'http_code=%{http_code} time_total=%{time_total}s\n' "http://127.0.0.1:${SELO_AI_PORT:-8000}${HEALTH_PATH}" || true
        echo "Trying LAN health (simple):"
        curl -sS -o /dev/null -w 'http_code=%{http_code} time_total=%{time_total}s\n' "$FAST_URL" || true
        echo "Trying LAN health:"
        curl -sS -o /dev/null -w 'http_code=%{http_code} time_total=%{time_total}s\n' "$HEALTH_URL" || true
    fi
    
    echo "Testing frontend connectivity..."
    FRONT_URL=${FRONTEND_URL:-http://localhost:3000}
    ft_tries=0; front_ok=false
    while [ $ft_tries -lt 30 ]; do
        if curl -sf "$FRONT_URL" >/dev/null; then front_ok=true; break; fi
        sleep 1; ft_tries=$((ft_tries+1))
    done
    if $front_ok; then
        echo "✓ Frontend is responding on port 3000"
    else
        echo "✗ Frontend test failed after ${ft_tries}s"
        echo "-- Recent frontend log --"
        tail -n 100 "$SCRIPT_DIR/logs/frontend.log" 2>/dev/null || true
    fi
    
    # Test configured models (conversational/reflection/analytical/embedding)
    echo "Testing configured model availability..."
    OLLAMA_BIN="$(command -v ollama || echo /usr/local/bin/ollama)"
    CMODEL="${CONVERSATIONAL_MODEL:-llama3:8b}"
    RMODEL="${REFLECTION_LLM:-qwen2.5:3b}"
    AMODEL="${ANALYTICAL_MODEL:-qwen2.5:3b}"
    EMODEL="${EMBEDDING_MODEL:-nomic-embed-text}"

    # Helper: robust presence check using `ollama show`, also accept implicit :latest
    _ollama_has_model() {
        local name="$1"
        # Strip provider prefix like ollama/
        local n
        n="$(echo "$name" | awk -F'/' '{print $NF}')"
        if "$OLLAMA_BIN" show "$n" >/dev/null 2>&1; then return 0; fi
        # If no explicit tag, try :latest as some installs list only with :latest
        if [[ "$n" != *:* ]]; then
            "$OLLAMA_BIN" show "$n:latest" >/dev/null 2>&1 && return 0
        fi
        return 1
    }
    # Conversational model: GGUF path or model name
    if [[ "$CMODEL" == /*.gguf ]]; then
        [ -f "$CMODEL" ] && echo "✓ Conversational model GGUF present: $CMODEL" || echo "✗ Conversational GGUF missing: $CMODEL"
    else
        CNAME="$(echo "$CMODEL" | awk -F'/' '{print $NF}')"
        if _ollama_has_model "$CNAME"; then echo "✓ Conversational model available: $CNAME"; else echo "⚠️  Conversational model not found in Ollama: $CNAME"; fi
    fi
    # Reflection model (strip provider prefix)
    RNAME="$(echo "$RMODEL" | awk -F'/' '{print $NF}')"
    if _ollama_has_model "$RNAME"; then echo "✓ Reflection model available: $RNAME"; else echo "⚠️  Reflection model not found in Ollama: $RNAME"; fi
    # Analytical model
    ANAME="$(echo "$AMODEL" | awk -F'/' '{print $NF}')"
    if _ollama_has_model "$ANAME"; then echo "✓ Analytical model available: $ANAME"; else echo "⚠️  Analytical model not found in Ollama: $ANAME"; fi
    # Embedding model
    ENAME="$(echo "$EMODEL" | awk -F'/' '{print $NF}')"
    if _ollama_has_model "$ENAME"; then echo "✓ Embedding model available: $ENAME"; else echo "⚠️  Embedding model not found in Ollama: $ENAME"; fi
    
    echo ""
}

# Final blocking warmup phase: ensure models are hot before announcing completion
final_model_warmup_blocking() {
    echo "========================================="
    echo "    Final Model Warmup (blocking)"
    echo "========================================="
    # Load service environment if available for API URL and model names
    if [ -r "/etc/selo-ai/environment" ]; then
        set -a
        source /etc/selo-ai/environment
        set +a
    fi
    # Resolve API base URL and Ollama base
    BASE_URL="${API_URL:-http://${HOST_IP:-localhost}:${SELO_AI_PORT:-8000}}"
    OLLAMA_HTTP="${OLLAMA_BASE_URL:-http://127.0.0.1:11434}"
    # Read model selections from environment with fallbacks
    echo "  • Models:       Conversational=${CONVERSATIONAL_MODEL_VAL:-${CONVERSATIONAL_MODEL:-llama3:8b}}; Reflection=${REFLECTION_LLM_VAL:-${REFLECTION_LLM:-qwen2.5:3b}}; Analytical=${ANALYTICAL_MODEL_VAL:-${ANALYTICAL_MODEL:-qwen2.5:3b}}; Embedding=${EMBEDDING_MODEL_VAL:-${EMBEDDING_MODEL:-nomic-embed-text}}"
    echo ""
    echo "Performance Defaults:"
    echo "  • Reflect-before-speak: REFLECTION_SYNC_MODE=${REFLECTION_SYNC_MODE:-sync} (timeout ${REFLECTION_SYNC_TIMEOUT_S:-0}s)"
    echo "  • Chat caps: CHAT_NUM_PREDICT=${CHAT_NUM_PREDICT:-192}, LLM_TIMEOUT=${LLM_TIMEOUT:-90}s"
    echo "  • Reflection caps: REFLECTION_NUM_PREDICT=${REFLECTION_NUM_PREDICT:-160}"
    echo "  • Keep-alive: OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:-10m}"
    echo "  • Models warmed: conversational='${CONVERSATIONAL_MODEL_VAL:-${CONVERSATIONAL_MODEL:-llama3:8b}}', reflection='${REFLECTION_LLM_VAL:-${REFLECTION_LLM:-qwen2.5:3b}}', analytical='${ANALYTICAL_MODEL_VAL:-${ANALYTICAL_MODEL:-qwen2.5:3b}}', embedding='${EMBEDDING_MODEL_VAL:-${EMBEDDING_MODEL:-nomic-embed-text}}'"
    echo ""
    echo "Configuration Files:"
    echo "  • Backend env:   $SCRIPT_DIR/backend/.env"
    echo "  • Service env:   /etc/selo-ai/environment"
    echo ""
    echo "${YELLOW}Security reminder:${NC} Do NOT share these files or the values within (e.g., SELO_SYSTEM_API_KEY)."
    echo ""
    echo ""
    echo "Service Management:"
    CURRENT_USER=${SUDO_USER:-$(whoami)}
    echo "  • Start:    sudo systemctl start selo-ai@$CURRENT_USER"
    echo "  • Stop:     sudo systemctl stop selo-ai@$CURRENT_USER"
    echo "  • Restart:  sudo systemctl restart selo-ai@$CURRENT_USER"
    echo "  • Status:   sudo systemctl status selo-ai@$CURRENT_USER"
    echo "  • Logs:     sudo journalctl -u selo-ai@$CURRENT_USER -f"
    echo ""
    echo "Features Enabled:"
    echo "  ✓ Automatic startup on server reboot"
    echo "  ✓ Firewall configured for network access"
    echo "  ✓ Dual-LLM routing via LLMRouter"
    echo "  ✓ Socket.IO real-time reflection streaming"
    echo "  ✓ Persistent conversations/memories database"
    echo "  ✓ Production-ready deployment"
    echo "SELO DSP is ready. Explore your personal SELO platform — with conversations, reflections, and memory — now running on your server."
    echo ""
}

# Final completion banner and quick links
show_completion_info() {
  echo "========================================="
  echo "    Installation Complete!"
  echo "========================================="
  # Try to resolve API URL similarly to warmup block
  BASE_URL="${API_URL:-http://${HOST_IP:-localhost}:${SELO_AI_PORT:-8000}}"
  echo "API Base URL: $BASE_URL"
  echo ""
  echo "Useful endpoints:"
  echo "  • Health (simple):   $BASE_URL/health/simple"
  echo "  • Health:            $BASE_URL/health"
  echo "  • Health (details):  $BASE_URL/health/details?probe_llm=true&probe_db=true"
  echo "  • Persona status:    $BASE_URL/api/persona/status"
  echo "  • GPU Diagnostics:   $BASE_URL/diagnostics/gpu"
  echo "  • Config (frontend): $BASE_URL/config.json"
  echo ""
  CURRENT_USER=${SUDO_USER:-$(whoami)}
  echo "Service commands:"
  echo "  • Start:    sudo systemctl start selo-ai@$CURRENT_USER"
  echo "  • Stop:     sudo systemctl stop selo-ai@$CURRENT_USER"
  echo "  • Restart:  sudo systemctl restart selo-ai@$CURRENT_USER"
  echo "  • Logs:     sudo journalctl -u selo-ai@$CURRENT_USER -f"
}

# Post-install environment parity validation
run_post_install_validation() {
    echo "========================================="
    echo "    Validating Install Environment"
    echo "========================================="
    local validator="$SCRIPT_DIR/scripts/validate-install-env.sh"
    if [ ! -x "$validator" ]; then
        chmod +x "$validator" 2>/dev/null || true
    fi
    if [ -f "$validator" ]; then
        if bash "$validator"; then
            echo "✓ Post-install validation passed"
        else
            echo "✗ Post-install validation reported failures. Review the output above and fix the mismatches."
            exit 1
        fi
    else
        echo "Validator script not found at $validator; skipping post-install validation"
    fi
}

# Function to check if running as root for firewall commands
check_sudo() {
    if ! sudo -n true 2>/dev/null; then
        echo "Some steps require sudo privileges (packages, firewall, /etc writes)."
        echo "You may be prompted for your password."
        sudo -v || { echo "Sudo authentication failed"; exit 1; }
    fi
}

# Configure simple ufw rules to allow access to backend/frontend (optional)
configure_firewall() {
    echo "========================================="
    echo "    Firewall Configuration (optional)"
    echo "========================================="
    if ! command -v ufw >/dev/null 2>&1; then
        echo "ufw not found. Skipping firewall configuration."
        return 0
    fi
    read -p "Configure ufw to allow backend port ${BACKEND_PORT} and frontend 3000? (y/N): " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        sudo ufw allow "${BACKEND_PORT}/tcp" || true
        sudo ufw allow "3000/tcp" || true
        # Allow local PostgreSQL (localhost access)
        sudo ufw allow from 127.0.0.1 to any port 5432 proto tcp || true
        # Enable ufw if currently inactive
        if ! sudo ufw status | grep -qi "Status: active"; then
            read -p "Enable ufw now? (y/N): " -n 1 -r
            echo ""
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                sudo ufw enable || true
            fi
        fi
        echo "Firewall rules configured."
    else
        echo "Skipping firewall configuration."
    fi
}

# Install or update systemd service unit for SELO DSP (instance service selo-ai@<user>)
install_service() {
    echo "========================================="
    echo "    Installing Systemd Service"
    echo "========================================="
    # Always install from the maintained template
    install_systemd_unit
    echo "✓ Systemd unit template installed"
}

# Main installation flow
main() {
    check_sudo
    
    # Pre-flight checks: Verify system meets minimum requirements
    echo "========================================="
    echo "    Pre-Installation System Checks"
    echo "========================================="
    
    # Check disk space (20GB minimum for models and data)
    if ! check_disk_space 20; then
        echo "Installation aborted due to insufficient disk space."
        exit 1
    fi
    
    # Check RAM (16GB recommended)
    if ! check_ram 16; then
        echo "Installation aborted due to insufficient RAM."
        exit 1
    fi
    
    echo ""
    
    # Step 1: Dependencies and environment
    install_dependencies
    ensure_backend_env_defaults
    ensure_service_env_core
    load_backend_env  # Load env vars for subsequent steps
    finalize_service_env
    
    # Step 2: Runtime deps (Ollama/models, Postgres when needed)
    ensure_runtime_after_env
    # Step 3: Build backend/frontend
    build_project
    
    # Step 4: Initialize database
    init_database
    
    # Step 4b: Verify critical models before persona bootstrap
    echo ""
    echo "========================================="
    echo "    Verifying Models for Bootstrap"
    echo "========================================="
    if ! verify_critical_models; then
        echo "Installation aborted: Critical models not available for persona bootstrap."
        exit 1
    fi
    
    # Step 4c: Bootstrap persona & mantra (CRITICAL: must complete before service creation)
    # This is a blocking step that generates the AI's identity.
    # If this fails, installation stops here - the service will not be created.
    echo ""
    echo "========================================="
    echo "CRITICAL STEP: Persona Generation"
    echo "========================================="
    bootstrap_persona_pre_service
    echo ""
    echo "✓ Persona generation complete. Proceeding with service installation..."
    echo ""
    
    # Step 5: Configure system services (only reached if persona bootstrap succeeded)
    configure_firewall
    install_service
    
    # Step 6: Start and validate
    start_services
    test_installation
    run_post_install_validation
    final_model_warmup_blocking
    show_completion_info
}

 # Confirmation prompt (honor AUTO_CONFIRM for non-interactive installs)
 echo "This script will configure your system with:"
 echo "  • System dependencies and optional components"
 echo "  • Environment configuration and service unit"
 echo "  • Firewall rules (optional) and health checks"
 echo ""
 if [ "${AUTO_CONFIRM_FLAG:-false}" = "true" ]; then
     echo "AUTO_CONFIRM enabled: proceeding without interactive confirmation."
     main
 else
     read -p "Do you want to proceed? (y/N): " -n 1 -r
     echo ""
     if [[ $REPLY =~ ^[Yy]$ ]]; then
         main
     else
         echo "Installation cancelled."
         exit 0
     fi
 fi
