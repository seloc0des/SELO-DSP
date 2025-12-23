#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Function to prompt for a variable with default and guidance
prompt_var() {
  local var_name="$1"
  local prompt_msg="$2"
  local default_val="$3"
  local secret="$4"
  local help_msg="$5"
  local val
  >&2 echo -e "\n$help_msg"
  if [ "$secret" = "true" ]; then
    read -rsp "$prompt_msg [required]: " val
    echo
  else
    read -rp "$prompt_msg [default: $default_val]: " val
  fi
  if [ -z "$val" ]; then
    val="$default_val"
  fi
  echo "$var_name=$val"
}

# Function to generate a random API key
random_api_key() {
  openssl rand -hex 32
}

BACKEND_ENV=".env.backend.tmp"
FRONTEND_ENV=".env.frontend.tmp"

rm -f "$BACKEND_ENV" "$FRONTEND_ENV"

# Normalize AUTO_CONFIRM for non-interactive runs (accept: 1/true/yes)
AUTO_CONFIRM_FLAG="false"
case "${AUTO_CONFIRM:-}" in
  1|true|TRUE|yes|YES|y|Y) AUTO_CONFIRM_FLAG="true" ;;
esac

# Load centralized tier detection (if not already loaded by parent script)
if [ -z "${PERFORMANCE_TIER:-}" ]; then
  if [ -f "$SCRIPT_DIR/detect-tier.sh" ]; then
    source "$SCRIPT_DIR/detect-tier.sh"
  else
    # Fallback if centralized script not found
    >&2 echo "Warning: detect-tier.sh not found, using defaults"
    export PERFORMANCE_TIER="standard"
    export TIER_REFLECTION_NUM_PREDICT=480
    export TIER_REFLECTION_MAX_TOKENS=480
    export TIER_REFLECTION_WORD_MAX=180
    export TIER_REFLECTION_WORD_MIN=90
    export TIER_ANALYTICAL_NUM_PREDICT=640
    export TIER_CHAT_NUM_PREDICT=1024
    export TIER_CHAT_NUM_CTX=8192
  fi
fi

# Display tier information
if [ "$PERFORMANCE_TIER" = "high" ]; then
  >&2 echo -e "${GREEN}✨ High-Performance Tier Detected (12GB+ GPU)${NC}"
  >&2 echo -e "   Enhanced persona bootstrap depth enabled"
else
  >&2 echo -e "${GREEN}⚡ Standard Tier Detected (8GB GPU)${NC}"
  >&2 echo -e "   Context window: 8192 tokens (full qwen2.5:3b capacity)"
  >&2 echo -e "   Full-quality few-shot examples preserved"
fi

# Backend required vars
{
  # Non-interactive local PostgreSQL configuration with generated password.
  # NOTE: actual user/database creation is performed later by install-complete.sh
  # using this DATABASE_URL as the single source of truth.
  DB_NAME="seloai"
  DB_USER="seloai"
  DB_PASS=$(openssl rand -hex 16)
  # Use postgresql:// form; backend/db/session.py upgrades to asyncpg driver
  DATABASE_URL="postgresql://${DB_USER}:${DB_PASS}@localhost/${DB_NAME}"
  >&2 echo -e "Using automatic DB settings: DB_NAME=$DB_NAME, DB_USER=$DB_USER"
  >&2 echo -e "${YELLOW}Constructed DATABASE_URL:${NC} postgresql://${DB_USER}:********@localhost/${DB_NAME}"

  # Persist DATABASE_URL to env; installer will ensure the matching Postgres user/database.
  echo "DATABASE_URL=$DATABASE_URL"

  # Generate SELO_SYSTEM_API_KEY automatically
  keyval=$(random_api_key)
  echo "SELO_SYSTEM_API_KEY=$keyval"
  >&2 echo "Generated SELO_SYSTEM_API_KEY (ends with ****${keyval: -4})"

  # Brave API key handling: honor preset; otherwise prompt unless AUTO_CONFIRM
  if [ -n "${BRAVE_SEARCH_API_KEY:-}" ]; then
    BRAVE_KEY="$BRAVE_SEARCH_API_KEY"
    echo "BRAVE_SEARCH_API_KEY=$BRAVE_KEY"
    LEN=${#BRAVE_KEY}; TAIL=${BRAVE_KEY: -4}
    >&2 echo "Using preset BRAVE_SEARCH_API_KEY (length=${LEN}, ends with=****${TAIL})"
  else
    if [ "$AUTO_CONFIRM_FLAG" = "true" ]; then
      >&2 echo -e "${YELLOW}AUTO_CONFIRM enabled:${NC} proceeding without BRAVE_SEARCH_API_KEY (web search disabled until set)."
      echo "BRAVE_SEARCH_API_KEY="
    else
      # Interactive prompt with masked input
      >&2 echo -e "The BRAVE_SEARCH_API_KEY is required for web search.\nRegister for an API key at:\n  https://search.brave.com/api\nPaste your API key here."
      read -rsp "Enter Brave Search API key [required]: " BRAVE_KEY
      echo
      if [ -z "$BRAVE_KEY" ]; then
        >&2 echo -e "${YELLOW}Warning:${NC} Brave API key was empty. You can set BRAVE_SEARCH_API_KEY later in backend/.env."
      fi
      echo "BRAVE_SEARCH_API_KEY=$BRAVE_KEY"
      if [ -n "$BRAVE_KEY" ]; then
        LEN=${#BRAVE_KEY}; TAIL=${BRAVE_KEY: -4}
        >&2 echo "Captured BRAVE_SEARCH_API_KEY (length=${LEN}, ends with=****${TAIL})"
      fi
    fi
  fi

  echo "SQL_ECHO=false"
  echo "SQL_POOL_SIZE=5"
  echo "SQL_MAX_OVERFLOW=10"
  echo "SQL_POOL_TIMEOUT=30"
  echo "SQL_POOL_RECYCLE=1800"
  # Scheduler/Resource Monitor (optional, safe defaults)
  echo "SCHEDULER_MIN_INTERVAL=60"
  echo "SCHEDULER_MAX_INTERVAL=3600"
  echo "SCHEDULER_DEFAULT_INTERVAL=300"
  echo "RESOURCE_MONITOR_INTERVAL=60"
  echo "RESOURCE_CPU_THRESHOLD=90"
  echo "RESOURCE_MEMORY_THRESHOLD=90"
  echo "RESOURCE_LOG_USAGE=false"
  
  # Backend network settings & CORS (auto-detected)
  SRV_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
  echo "HOST=0.0.0.0"
  echo "PORT=8000"
  FRONT_VAL="http://${SRV_IP:-localhost}:3000"
  echo "FRONTEND_URL=$FRONT_VAL"
  echo "CORS_ORIGINS=$FRONT_VAL"
  
  # Ollama base URL (where models are served) - default or honor preset, no prompt
  echo "OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-http://localhost:11434}"

  # Feature flags: Always enabled per product requirement
  echo "ENABLE_REFLECTION_SYSTEM=true"
  echo "ENABLE_ENHANCED_SCHEDULER=true"
  echo "SOCKET_IO_ENABLED=true"

  # Map to app_config flags (always true)
  echo "REFLECTION_ENABLED=true"
  echo "REFLECTION_SCHEDULE_ENABLED=true"

  # Session secret for backend (auto-generate)
  sessval=$(random_api_key)
  echo "SESSION_SECRET=$sessval"
  >&2 echo "Generated SESSION_SECRET (ends with ****${sessval: -4})"

  # Model selection (selected trio defaults)
  # Use standard llama3:8b for consistency across all configs
  echo "CONVERSATIONAL_MODEL=llama3:8b"
  echo "ANALYTICAL_MODEL=qwen2.5:3b"
  # Reflection model default (qwen2.5:3b for proper word count generation)
  echo "REFLECTION_LLM=qwen2.5:3b"
  # Embedding configuration
  echo "EMBEDDING_MODEL=nomic-embed-text"
  echo "EMBEDDING_DIM=768"
  # Per-type reflection models (ensure defaults at installation time)
  echo "REFLECTION_MODEL_DEFAULT=qwen2.5:3b"
  echo "REFLECTION_MODEL_MESSAGE=qwen2.5:3b"
  echo "REFLECTION_MODEL_DAILY=qwen2.5:3b"
  echo "REFLECTION_MODEL_WEEKLY=qwen2.5:3b"
  echo "REFLECTION_MODEL_EMOTIONAL=qwen2.5:3b"
  echo "REFLECTION_MODEL_MANIFESTO=qwen2.5:3b"
  echo "REFLECTION_MODEL_PERIODIC=qwen2.5:3b"
  
  # Advanced backend variables: set automatically (no prompts)
  echo "DEBUG=false"
  echo "LOG_LEVEL=INFO"
  # Heavyweight-friendly default: unbounded LLM timeout (0)
  echo "LLM_TIMEOUT=0"
  echo "LLM_MAX_RETRIES=3"
  echo "RATE_LIMIT_REQUESTS=100"
  echo "RATE_LIMIT_WINDOW=60"
  echo "SOCKETIO_PING_TIMEOUT=60"
  echo "SOCKETIO_PING_INTERVAL=25"
  # Performance-oriented sensible defaults for responsiveness
  # Reflect-first synchronously with no backend-imposed timeouts
  echo "REFLECTION_SYNC_MODE=sync"
  echo "REFLECTION_SYNC_TIMEOUT_S=0"
  echo "REFLECTION_LLM_TIMEOUT_S=0"
  echo "REFLECTION_REQUIRED=true"
  echo "CHAT_RESPONSE_MAX_TOKENS=320"
  echo "CHAT_NUM_PREDICT=1024"
  echo "CHAT_TEMPERATURE=0.6"
  echo "CHAT_TOP_K=40"
  echo "CHAT_TOP_P=0.9"
  echo "CHAT_NUM_CTX=8192"
  echo "OLLAMA_KEEP_ALIVE=30m"
  # Note: OLLAMA_GPU_LAYERS is seeded by installer; defaults to 68 when CUDA is enabled
  # Reflection caps (hardware-adaptive based on GPU tier)
  echo "REFLECTION_NUM_PREDICT=$TIER_REFLECTION_NUM_PREDICT"
  echo "REFLECTION_MAX_TOKENS=$TIER_REFLECTION_MAX_TOKENS"
  echo "REFLECTION_WORD_MIN=$TIER_REFLECTION_WORD_MIN"  # Concise inner voice (90 words)
  echo "REFLECTION_WORD_MAX=$TIER_REFLECTION_WORD_MAX"  # Concise inner voice (180 words)
  # Reflection identity validation retries (higher by default for fast 1.5b model)
  echo "REFLECTION_IDENTITY_MAX_RETRIES=4"
  # Analytical token budget for structured outputs (hardware-adaptive)
  echo "ANALYTICAL_NUM_PREDICT=$TIER_ANALYTICAL_NUM_PREDICT"
  echo "ANALYTICAL_TEMPERATURE=0.2"
  # Enable model pre-warm at startup for lower first-turn latency
  echo "PREWARM_MODELS=true"
  # Periodic keepalive pings to keep models hot (minutes)
  echo "KEEPALIVE_ENABLED=true"
  echo "PREWARM_INTERVAL_MIN=5"
} > "$BACKEND_ENV"

# Frontend required vars (auto-set to backend API URL based on detected IP)
{
  SRV_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
  API_URL="http://${SRV_IP:-localhost}:8000"
  echo "REACT_APP_API_URL=$API_URL"
  echo "VITE_API_URL=$API_URL"
} > "$FRONTEND_ENV"

# Simple validation helpers
require_kv() {
  local file="$1"; local key="$2"
  if ! grep -Eq "^${key}=" "$file"; then
    >&2 echo -e "${YELLOW}Missing required key: ${key}${NC}"
    return 1
  fi
  local val
  val=$(grep -E "^${key}=" "$file" | tail -n1 | cut -d '=' -f2-)
  if [ -z "$val" ]; then
    >&2 echo -e "${YELLOW}Required key ${key} has empty value${NC}"
    return 1
  fi
}

# Pre-apply validation
VALID=true
require_kv "$BACKEND_ENV" "DATABASE_URL" || VALID=false
require_kv "$BACKEND_ENV" "SELO_SYSTEM_API_KEY" || VALID=false
require_kv "$BACKEND_ENV" "CONVERSATIONAL_MODEL" || VALID=false
require_kv "$FRONTEND_ENV" "VITE_API_URL" || VALID=false

# Move to correct locations after confirmation (support AUTO_CONFIRM)
echo -e "${YELLOW}Review the following environment variables for backend:${NC}"
cat "$BACKEND_ENV"
echo -e "${YELLOW}Review the following environment variables for frontend:${NC}"
cat "$FRONTEND_ENV"
if [ "$VALID" != true ]; then
  echo -e "${YELLOW}One or more required values are missing. Please re-run and provide required values.${NC}"
  rm -f "$BACKEND_ENV" "$FRONTEND_ENV"
  exit 1
fi
if [ "$AUTO_CONFIRM_FLAG" = "true" ]; then
  confirm="y"
  >&2 echo -e "${YELLOW}AUTO_CONFIRM enabled:${NC} applying environment without prompt."
else
  read -rp "Apply these settings? (y/n): " confirm
fi
if [[ "$confirm" =~ ^[Yy]$ ]]; then
  mv "$BACKEND_ENV" "./backend/.env"
  mv "$FRONTEND_ENV" "./frontend/.env"
  # Create a service environment sample for systemd installation
  SERVICE_ENV_SAMPLE="$SCRIPT_DIR/environment.service.sample"
  # Derive API_URL from selections (prefer HOST/PORT if provided)
  HOST_VAL=$(grep -E '^HOST=' ./backend/.env | tail -n1 | cut -d '=' -f2-)
  PORT_VAL=$(grep -E '^PORT=' ./backend/.env | tail -n1 | cut -d '=' -f2-)
  API_BASE="http://${HOST_VAL:-localhost}:${PORT_VAL:-8000}"
  FRONT_VAL=$(grep -E '^FRONTEND_URL=' ./backend/.env | tail -n1 | cut -d '=' -f2-)
  CORS_VAL=$(grep -E '^CORS_ORIGINS=' ./backend/.env | tail -n1 | cut -d '=' -f2-)
  SOCKET_VAL=$(grep -E '^SOCKET_IO_ENABLED=' ./backend/.env | tail -n1 | cut -d '=' -f2-)
  echo "# Copy to /etc/selo-ai/environment (requires sudo)" > "$SERVICE_ENV_SAMPLE"
  echo "SELO_AI_PORT=${PORT_VAL:-8000}" >> "$SERVICE_ENV_SAMPLE"
  echo "HOST=${HOST_VAL:-0.0.0.0}" >> "$SERVICE_ENV_SAMPLE"
  echo "API_URL=$API_BASE" >> "$SERVICE_ENV_SAMPLE"
  echo "FRONTEND_URL=${FRONT_VAL:-http://localhost:3000}" >> "$SERVICE_ENV_SAMPLE"
  echo "INSTALL_DIR=${SCRIPT_DIR}" >> "$SERVICE_ENV_SAMPLE"
  echo "ENABLE_REFLECTION_SYSTEM=$(grep -E '^ENABLE_REFLECTION_SYSTEM=' ./backend/.env | tail -n1 | cut -d '=' -f2-)" >> "$SERVICE_ENV_SAMPLE"
  echo "ENABLE_ENHANCED_SCHEDULER=$(grep -E '^ENABLE_ENHANCED_SCHEDULER=' ./backend/.env | tail -n1 | cut -d '=' -f2-)" >> "$SERVICE_ENV_SAMPLE"
  echo "REFLECTION_LLM=$(grep -E '^REFLECTION_LLM=' ./backend/.env | tail -n1 | cut -d '=' -f2-)" >> "$SERVICE_ENV_SAMPLE"
  # Include Socket.IO toggle
  if [ -n "$SOCKET_VAL" ]; then echo "SOCKET_IO_ENABLED=$SOCKET_VAL" >> "$SERVICE_ENV_SAMPLE"; else echo "SOCKET_IO_ENABLED=true" >> "$SERVICE_ENV_SAMPLE"; fi
  # Provide CORS origins directly (not commented) for immediate usability
  if [ -n "$CORS_VAL" ]; then
    echo "CORS_ORIGINS=$CORS_VAL" >> "$SERVICE_ENV_SAMPLE"
  else
    # Fallback: allow the chosen frontend and localhost:3000
    echo "CORS_ORIGINS=${FRONT_VAL:-http://localhost:3000},http://localhost:3000" >> "$SERVICE_ENV_SAMPLE"
  fi
  echo -e "${GREEN}Environment variables set successfully.${NC}"
  echo -e "\nLocations:"
  echo "  • Backend env:   $SCRIPT_DIR/backend/.env"
  echo "  • Service env:   /etc/selo-ai/environment (created later by installer)"
  echo "  • Sample file:   $SERVICE_ENV_SAMPLE (review and copy with: sudo mkdir -p /etc/selo-ai && sudo cp '$SERVICE_ENV_SAMPLE' /etc/selo-ai/environment)"
  echo -e "\n${YELLOW}Security reminder:${NC} Do NOT share these files or the values within (e.g., SELO_SYSTEM_API_KEY)."
else
  echo -e "${YELLOW}Environment variable setup cancelled.${NC}"
  rm -f "$BACKEND_ENV" "$FRONTEND_ENV"
  exit 1
fi
