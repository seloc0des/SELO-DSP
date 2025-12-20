#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

# SELO DSP Service Startup Script (for systemd)
# This script is designed to work with systemd service management

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Initialize logs as early as possible so any early failure is captured
mkdir -p "$SCRIPT_DIR/logs" 2>/dev/null || true
chmod 775 "$SCRIPT_DIR/logs" 2>/dev/null || true
BACKEND_LOG="$SCRIPT_DIR/logs/backend.log"
FRONTEND_LOG="$SCRIPT_DIR/logs/frontend.log"
SERVICE_LOG="$SCRIPT_DIR/logs/service.log"
touch "$BACKEND_LOG" "$FRONTEND_LOG" "$SERVICE_LOG" 2>/dev/null || true
chmod 664 "$BACKEND_LOG" "$FRONTEND_LOG" "$SERVICE_LOG" 2>/dev/null || true

# Minimal early logger and error trap (before full log() is defined)
_early_log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $*"
    # Try file first; if it fails, fall back to journal without failing script
    { echo "$msg" >> "$SERVICE_LOG"; } 2>/dev/null || {
        command -v logger >/dev/null 2>&1 && logger -t selo-ai "$msg" || echo "$msg" >&2
    }
}
trap '_early_log "Startup error at line $LINENO; last cmd: ${BASH_COMMAND}"' ERR

# Load system-wide environment early (non-fatal if missing)
if [ -f "/etc/selo-ai/environment" ]; then
    if [ -r "/etc/selo-ai/environment" ]; then
        set -a
        # shellcheck disable=SC1091
        source /etc/selo-ai/environment
        set +a
    else
        # Minimal early warning (SERVICE_LOG not initialized yet)
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Warning: /etc/selo-ai/environment exists but is not readable; continuing without it." >&2
    fi
fi

# Prevent concurrent overlapping starts (e.g., due to rapid restarts)
# Prefer a lock file inside the install dir, but fall back to a user-writable
# runtime directory when the install path is not writable (e.g., owned by root).
LOCK_FILE="$SCRIPT_DIR/.selo-ai-start.lock"
: > "$LOCK_FILE" 2>/dev/null || true
if ! touch "$LOCK_FILE" 2>/dev/null; then
    ALT_LOCK_BASE="${XDG_RUNTIME_DIR:-/run/user/$(id -u)}"
    ALT_LOCK_DIR="${ALT_LOCK_BASE%/}/selo-ai"
    mkdir -p "$ALT_LOCK_DIR" 2>/dev/null || ALT_LOCK_DIR="/tmp/selo-ai"
    mkdir -p "$ALT_LOCK_DIR" 2>/dev/null || true
    LOCK_FILE="$ALT_LOCK_DIR/start-service.lock"
    if ! touch "$LOCK_FILE" 2>/dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Unable to create lock file in writable directory." >&2
        exit 1
    fi
fi
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    # Minimal logger until log() is available later
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Another selo-ai start-service.sh instance is running. Exiting to avoid overlap." >&2
    exit 0
fi

# Load backend .env early so feature flags are available before use
if [ -f "$SCRIPT_DIR/backend/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/backend/.env"
    set +a
fi

# Safe defaults for feature flags and critical vars (avoid set -u crashes)
ENABLE_REFLECTION_SYSTEM=${ENABLE_REFLECTION_SYSTEM:-false}
ENABLE_ENHANCED_SCHEDULER=${ENABLE_ENHANCED_SCHEDULER:-false}
SOCKET_IO_ENABLED=${SOCKET_IO_ENABLED:-true}

# Determine host IP robustly if not provided
deduce_host_ip() {
    # Prefer explicit HOST or SERVER_IP from environment
    if [ -n "${HOST:-}" ]; then echo "$HOST"; return; fi
    if [ -n "${SERVER_IP:-}" ]; then echo "$SERVER_IP"; return; fi
    # Try hostname -I and pick first private IPv4
    if command -v hostname >/dev/null 2>&1; then
        local ips
        ips=$(hostname -I 2>/dev/null || true)
        for ip in $ips; do
            case "$ip" in
                10.*|192.168.*|172.1[6-9].*|172.2[0-9].*|172.3[0-1].*) echo "$ip"; return;;
            esac
        done
        # Fallback to the first token if any
        for ip in $ips; do echo "$ip"; return; done
    fi
    echo "127.0.0.1"
}

# Export HOST_IP if not set so all later URL constructions are consistent
export HOST_IP=${HOST_IP:-$(deduce_host_ip)}

# Ensure fresh logs and remove stale PID files each run
rm -f "$BACKEND_LOG" "$FRONTEND_LOG" "$SERVICE_LOG" \
      "$SCRIPT_DIR/backend.pid" "$SCRIPT_DIR/frontend.pid" 2>/dev/null || true
touch "$BACKEND_LOG" "$FRONTEND_LOG" "$SERVICE_LOG"
# Make log files group-writable to avoid permission issues across service restarts
chmod 664 "$BACKEND_LOG" "$FRONTEND_LOG" "$SERVICE_LOG" 2>/dev/null || true

# Function to log with timestamp
log() {
    local msg="[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    # Write to file if possible; otherwise to journal/STDERR. Never let logging failure abort the script.
    { echo "$msg" >> "$SERVICE_LOG"; } 2>/dev/null || {
        command -v logger >/dev/null 2>&1 && logger -t selo-ai "$msg" || echo "$msg"
    }
}

log "========================================="
log "    SELO DSP Digital Sentience Platform Starting"
log "========================================="
log "Script directory: $SCRIPT_DIR"
 
# Guard: warn if there appears to be another checkout of this repo and we are
# not running from the one under /mnt/Projects. This helps avoid confusion when
# editing code in one path but starting the service from another path.
if [ -d "/mnt/Projects/GitHub/SELOBasiChat/selo-ai" ] && [ "$SCRIPT_DIR" != "/mnt/Projects/GitHub/SELOBasiChat/selo-ai" ]; then
    log "Warning: detected another checkout at /mnt/Projects/GitHub/SELOBasiChat/selo-ai but running from $SCRIPT_DIR."
    log "Edits made in the other path will not affect this run. Consider starting from the edited checkout."
fi

# Ensure we cleanup child processes on stop to prevent lingering node/sh
cleanup() {
    local code=$?
    log "Received stop signal; beginning graceful shutdown (exit code $code)"
    # Gracefully stop known child PIDs if present, then force if needed
    for pidfile in "$SCRIPT_DIR/backend.pid" "$SCRIPT_DIR/frontend.pid"; do
        if [ -f "$pidfile" ]; then
            pid=$(cat "$pidfile" 2>/dev/null || true)
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                log "Sending SIGTERM to PID $pid from $(basename "$pidfile")"
                kill "$pid" 2>/dev/null || true
                # Wait up to 10s for graceful stop
                for i in $(seq 1 10); do
                    if kill -0 "$pid" 2>/dev/null; then sleep 1; else break; fi
                done
                if kill -0 "$pid" 2>/dev/null; then
                    log "PID $pid did not exit in time; sending SIGKILL"
                    kill -9 "$pid" 2>/dev/null || true
                fi
            fi
        fi
    done
    # Terminate any remaining background jobs spawned by this script (watchdog, etc.)
    for j in $(jobs -pr 2>/dev/null || true); do
        kill "$j" 2>/dev/null || true
    done
    # As a last resort, ensure ports are freed
    kill_port "${SELO_AI_PORT:-8000}"
    kill_port 3000
    log "Shutdown sequence complete"
    exit $code
}
# Run cleanup on TERM/INT/EXIT to ensure systemd stops do not leave children
trap cleanup TERM INT EXIT

# Function to check if a port is in use (tries lsof, falls back to ss)
check_port() {
    local port=$1
    if command -v lsof >/dev/null 2>&1; then
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            return 0  # Port is in use
        else
            return 1  # Port is free
        fi
    elif command -v ss >/dev/null 2>&1; then
        if ss -ltn "sport = :$port" 2>/dev/null | tail -n +2 | grep -q ":$port"; then
            return 0
        else
            return 1
        fi
    else
        # Without lsof/ss, assume free and warn once
        log "Warning: neither 'lsof' nor 'ss' found; cannot verify port $port. Assuming free."
        return 1
    fi
}

# Function to kill process on port
kill_port() {
    local port=$1
    log "Killing any existing process on port $port..."
    if command -v lsof >/dev/null 2>&1; then
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
    elif command -v ss >/dev/null 2>&1; then
        # Find PIDs listening on the port and kill them
        PIDS=$(ss -ltnp "sport = :$port" 2>/dev/null | awk -F',' '/pid=/ { for (i=1; i<=NF; i++) { if ($i ~ /^pid=/) { gsub("pid=","",$i); print $i } } }')
        if [ -n "$PIDS" ]; then
            echo "$PIDS" | xargs -r kill -9 2>/dev/null || true
        fi
    else
        log "Warning: cannot determine PIDs to kill on port $port (no lsof/ss)."
    fi
    sleep 2
}

# Check for required LLM models
log "Checking LLM model availability..."

# Determine Ollama binary
OLLAMA_BIN="$(command -v ollama || echo /usr/local/bin/ollama)"
if [ ! -x "$OLLAMA_BIN" ]; then
    log "Ollama binary not found. Please install Ollama before starting the service."
    exit 1
fi

# Keep models hot between requests to avoid cold starts
export OLLAMA_KEEP_ALIVE=${OLLAMA_KEEP_ALIVE:-10m}

# Check conversational model (could be GGUF path or Ollama model)
log "Checking LLM model availability..."
CONVERSATIONAL_MODEL=${CONVERSATIONAL_MODEL:-llama3:8b-instruct}
if [[ "$CONVERSATIONAL_MODEL" == *.gguf ]] && [[ "$CONVERSATIONAL_MODEL" == /* ]]; then
    # GGUF file path - check if file exists
    if [ -f "$CONVERSATIONAL_MODEL" ]; then
        log "Conversational model GGUF file found: $CONVERSATIONAL_MODEL"
    else
        log "Error: Conversational model GGUF file not found: $CONVERSATIONAL_MODEL"
        exit 1
    fi
else
    # Ollama model name - check if available (compare against first column)
    if ! "$OLLAMA_BIN" list | awk '{print $1}' | grep -qx "$CONVERSATIONAL_MODEL" 2>/dev/null; then
      if [ "$CONVERSATIONAL_MODEL" = "humanish-llama3" ]; then
        # Attempt to provision a local alias by pulling the HF GGUF reference, then creating a Modelfile alias
        HF_REF="hf.co/bartowski/Human-Like-LLama3-8B-Instruct-GGUF:Humanish-LLama3-8B-Instruct-Q4_K_L.gguf"
        log "Conversational model 'humanish-llama3' not found. Attempting HF pull and local alias creation..."
        if "$OLLAMA_BIN" pull "$HF_REF" >> "$SERVICE_LOG" 2>&1; then
          TMP_MODELFILE=$(mktemp)
          {
            echo "FROM $HF_REF"
          } > "$TMP_MODELFILE"
          if "$OLLAMA_BIN" create humanish-llama3 -f "$TMP_MODELFILE" >> "$SERVICE_LOG" 2>&1; then
            log "Created local alias 'humanish-llama3' from HF reference."
            SKIP_MODEL_PULL=true
          else
            log "Alias creation failed; will fallback to pulling qwen2.5:3b."
            CONVERSATIONAL_MODEL="qwen2.5:3b"
          fi
          rm -f "$TMP_MODELFILE" 2>/dev/null || true
        else
          log "HF pull for Humanish LLama3 failed; falling back to qwen2.5:3b."
          CONVERSATIONAL_MODEL="qwen2.5:3b"
        fi
      fi
      # Ensure selected conversational model is pulled (either remote or fallback)
      # Skip pulling when we just created a local alias (no remote manifest exists for alias name)
      if [ "${SKIP_MODEL_PULL:-false}" != "true" ]; then
        "$OLLAMA_BIN" pull "$CONVERSATIONAL_MODEL" >> "$SERVICE_LOG" 2>&1 || true
      fi
      # If pull failed, leave model routing to backend fallbacks
    fi
fi

# If a conversational tag variant is requested (e.g., humanish-llama3:8b-q4)
# and it's missing while the base alias exists, create the tag automatically.
if [ -n "${CONVERSATIONAL_MODEL:-}" ]; then
    if ! "$OLLAMA_BIN" list | awk '{print $1}' | grep -qx "$CONVERSATIONAL_MODEL" 2>/dev/null; then
        # Check for humanish-llama3 prefix
        if echo "$CONVERSATIONAL_MODEL" | grep -q '^humanish-llama3:'; then
            if "$OLLAMA_BIN" list | awk '{print $1}' | grep -qx "humanish-llama3" 2>/dev/null; then
                log "Conversational tag $CONVERSATIONAL_MODEL not found; creating tag from base alias 'humanish-llama3'..."
                TMP_MODELFILE=$(mktemp)
                echo "FROM humanish-llama3" > "$TMP_MODELFILE"
                if "$OLLAMA_BIN" create "$CONVERSATIONAL_MODEL" -f "$TMP_MODELFILE" >> "$SERVICE_LOG" 2>&1; then
                    log "Created conversational tag '$CONVERSATIONAL_MODEL' from base alias."
                else
                    log "Failed to create conversational tag '$CONVERSATIONAL_MODEL'. Will continue with base alias if available."
                fi
                rm -f "$TMP_MODELFILE" 2>/dev/null || true
            fi
        fi
    fi
fi

# Reflection model (always ensure availability)
REFLECTION_MODEL=${REFLECTION_LLM:-"qwen2.5:3b"}
REFLECTION_MODEL_NAME=$(echo "$REFLECTION_MODEL" | awk -F'/' '{print $NF}')
if ! "$OLLAMA_BIN" list | awk '{print $1}' | grep -qx "${REFLECTION_MODEL_NAME}" 2>/dev/null; then
    if [ "$REFLECTION_MODEL_NAME" = "phi3:mini-4k-instruct" ]; then
        # Attempt to provision a local alias from Bartowski GGUF
        PHI3_HF_REF="hf.co/bartowski/Phi-3-mini-4k-instruct-GGUF:Phi-3-mini-4k-instruct-Q4_K_M.gguf"
        log "Reflection model 'phi3:mini-4k-instruct' not found. Attempting HF pull and local alias creation..."
        if "$OLLAMA_BIN" pull "$PHI3_HF_REF" >> "$SERVICE_LOG" 2>&1; then
            TMP_MODELFILE=$(mktemp)
            echo "FROM $PHI3_HF_REF" > "$TMP_MODELFILE"
            if "$OLLAMA_BIN" create phi3:mini-4k-instruct -f "$TMP_MODELFILE" >> "$SERVICE_LOG" 2>&1; then
                log "Created local alias 'phi3:mini-4k-instruct' from HF reference."
                SKIP_REF_PULL=true
            else
                log "Alias creation for phi3 failed; will try direct pull name as fallback."
            fi
            rm -f "$TMP_MODELFILE" 2>/dev/null || true
        else
            log "HF pull for Phi-3 Mini 4K instruct failed; will try direct pull name as fallback."
        fi
    fi
    # Pull by direct name if alias path not taken or failed
    if [ "${SKIP_REF_PULL:-false}" != "true" ]; then
        log "Reflection model not found. Pulling ${REFLECTION_MODEL_NAME}..."
        "$OLLAMA_BIN" pull "${REFLECTION_MODEL_NAME}" >> "$SERVICE_LOG" 2>&1 || true
    fi
    # Final check
    if "$OLLAMA_BIN" list | awk '{print $1}' | grep -qx "${REFLECTION_MODEL_NAME}" 2>/dev/null; then
        log "Reflection model successfully installed!"
    else
        log "Failed to make reflection model available. Will use Mistral as fallback."
        export REFLECTION_LLM="ollama/mistral:latest"
    fi
else
    log "Reflection model is already available."
fi

# Clean up any existing processes
log "Cleaning up existing processes..."
# Use SELO_AI_PORT exclusively (no PORT fallback) to avoid ambiguity
BACKEND_PORT=${SELO_AI_PORT:-8000}
kill_port "$BACKEND_PORT"  # Backend
kill_port 3000  # Frontend

# If backend port is still in use after kill attempt, abort instead of shifting
if check_port "$BACKEND_PORT"; then
    log "Port $BACKEND_PORT still in use after kill attempt. Auto-selecting a free port to allow installer to complete..."
    # Find a free port between 8000-8100
    FREE_PORT=$BACKEND_PORT
    for p in $(seq 8000 8100); do
        if ! check_port "$p"; then FREE_PORT=$p; break; fi
    done
    if [ "$FREE_PORT" != "$BACKEND_PORT" ]; then
        BACKEND_PORT=$FREE_PORT
        export SELO_AI_PORT="$BACKEND_PORT"
        log "Using alternate backend port: $BACKEND_PORT"
    else
        log "No free port found in 8000-8100 range; continuing with $BACKEND_PORT (may fail)."
    fi
fi

# Export selected port so backend binds correctly
export SELO_AI_PORT="$BACKEND_PORT"
echo -n "$BACKEND_PORT" > "$SCRIPT_DIR/backend.port" 2>/dev/null || true

# Persist effective URLs to /etc/selo-ai/environment for diagnostics
# NOTE: Runtime writes are intentionally disabled to honor systemd ProtectSystem=full.
#       The installer is responsible for creating/updating /etc/selo-ai/environment.
#       This service will rely on backend/.env and exported env vars only.
persist_service_env() {
    log "Skipping runtime updates to /etc/selo-ai/environment (read-only under systemd)."
    log "Env sources: backend/.env and installer-provisioned /etc/selo-ai/environment (read-only)."
    return 0
}

persist_service_env

# Verify core runtimes before proceeding
if ! command -v python3 >/dev/null 2>&1; then
    log "Error: python3 is not installed. Install Python 3.10+ and retry."
    exit 1
fi
if ! python3 -m venv --help >/dev/null 2>&1; then
    log "Error: Python venv module is unavailable. Install python3-venv."
    exit 1
fi
if ! command -v node >/dev/null 2>&1; then
    log "Error: node is not installed. Install Node.js 18+ and retry."
    exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
    log "Error: npm is not installed. Install npm and retry."
    exit 1
fi

# Start Backend
log "Starting SELO DSP Backend..."
cd "$SCRIPT_DIR/backend"

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    log "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
log "Activating virtual environment and installing dependencies..."
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip >> "$SERVICE_LOG" 2>&1

# Install dependencies with proper lxml support
log "Installing Python dependencies..."
pip install -r requirements.txt >> "$SERVICE_LOG" 2>&1

# Load environment variables for the backend early so DATABASE_URL is available for DB init
if [ -f "$SCRIPT_DIR/backend/.env" ]; then
    set -a  # automatically export all variables
    source "$SCRIPT_DIR/backend/.env"
    set +a
    log "Environment variables loaded from backend/.env (pre-init)"
fi

# Verify critical imports work
log "Verifying dependencies..."
python -c "import fastapi, uvicorn, trafilatura, requests, socketio, pydantic, asyncio; print('✓ Core dependencies verified')" >> "$SERVICE_LOG" 2>&1

# Verify reflection system dependencies if enabled
if [ "${ENABLE_REFLECTION_SYSTEM:-false}" = "true" ]; then
    log "Verifying reflection system dependencies..."
    python -c "import faiss, numpy, jinja2, sentence_transformers; print('✓ Reflection system dependencies verified')" >> "$SERVICE_LOG" 2>&1 || log "Warning: Some reflection dependencies missing. Will use fallbacks where possible."
fi

# Verify enhanced scheduler dependencies if enabled
if [ "${ENABLE_ENHANCED_SCHEDULER:-false}" = "true" ]; then
    log "Verifying enhanced scheduler dependencies..."
    python -c "import apscheduler, psutil, statistics, pytz; print('✓ Enhanced scheduler dependencies verified')" >> "$SERVICE_LOG" 2>&1 || log "Warning: Some scheduler dependencies missing. Will use fallbacks where possible."
fi

# Start backend in background with logging
log "Starting SELO DSP backend with reflection system on port ${SELO_AI_PORT:-8000}..."

# Set up PostgreSQL schema if a database is configured
if [ -n "${DATABASE_URL:-}" ]; then
    log "Initializing database schema (users, reflections, schedules)..."
    
    # Use file-based locking to prevent concurrent database initialization
    # This prevents race conditions with the installer
    # Use /tmp for the lock file since /var/lock requires root permissions
    lock_file="/tmp/selo-ai-db-init.lock"
    
    # Remove stale lock file if it exists with wrong permissions
    if [ -f "$lock_file" ] && [ ! -w "$lock_file" ]; then
        log "Removing stale lock file with wrong permissions..."
        rm -f "$lock_file" 2>/dev/null || sudo rm -f "$lock_file" 2>/dev/null || true
    fi
    
    # Try to acquire lock (wait up to 30 seconds, installer has priority)
    log "Acquiring database initialization lock..."
    # Ensure lock file exists with proper permissions
    touch "$lock_file" 2>/dev/null || true
    chmod 666 "$lock_file" 2>/dev/null || true
    exec 201>"$lock_file"
    if flock -w 30 201; then
        log "Lock acquired. Initializing database..."
        
        # Initialize database tables inline (no temp file to avoid race conditions)
        python - <<'PYCODE' >> "$SERVICE_LOG" 2>&1
import asyncio, logging
logging.basicConfig(level=logging.INFO)
logging.info("Initializing SELO DSP database tables (users, reflections, schedules)...")
try:
    from db.init_db import create_tables
    asyncio.run(create_tables())
    logging.info("\u2713 Database tables created successfully!")
except Exception as e:
    logging.error(f"\u2717 Database initialization error: {e}")
    raise
PYCODE
        INIT_RESULT=$?
        
        # Release lock
        flock -u 201
        
        if [ $INIT_RESULT -ne 0 ]; then
            log "Warning: Database initialization failed. Backend may experience errors until resolved."
        else
            log "✓ Database schema initialized successfully!"
        fi
    else
        log "Could not acquire database lock (installer may be running). Skipping DB init."
        log "Database should already be initialized by installer."
    fi
fi

# (env already loaded above before DB init if present)

# Helper to start the backend with uvicorn factory app (avoids ASGI factory warning)
cd "$SCRIPT_DIR"
UVICORN_BIN="$(command -v uvicorn || true)"
if [ -z "$UVICORN_BIN" ]; then
    UVICORN_BIN="$SCRIPT_DIR/backend/venv/bin/uvicorn"
fi

start_backend() {
    # Use virtual environment Python directly to ensure proper package resolution
    cd "$SCRIPT_DIR"
    VENV_PYTHON="$SCRIPT_DIR/backend/venv/bin/python"
    nohup "$VENV_PYTHON" -m uvicorn "backend.main:get_socketio_app" \
        --host 0.0.0.0 \
        --port "$BACKEND_PORT" \
        --factory \
        --timeout-graceful-shutdown 10 \
        > "$BACKEND_LOG" 2>&1 &
    BACKEND_PID=$!
    echo $BACKEND_PID > "$SCRIPT_DIR/backend.pid"
    log "Backend started with PID: $BACKEND_PID"
}

start_backend

# Wait for backend to start (up to ~30s)
ATTEMPTS=0
MAX_ATTEMPTS=30
until check_port "$BACKEND_PORT"; do
    ATTEMPTS=$((ATTEMPTS+1))
    if [ "$ATTEMPTS" -ge "$MAX_ATTEMPTS" ]; then
        log "✗ Backend failed to start on port ${SELO_AI_PORT:-8000} after ${MAX_ATTEMPTS}s"
        log "Last 100 lines of backend log:" 
        tail -n 100 "$BACKEND_LOG" | tee -a "$SERVICE_LOG"
        exit 1
    fi
    sleep 1
done
log "✓ Backend is running on http://${HOST_IP:-127.0.0.1}:${SELO_AI_PORT:-8000}"

# Warm conversational, reflection, and analytical models to reduce first-turn latency
(
  log "Warming LLM models to reduce cold-start latency..."
  # Conversational model (alias or concrete model name)
  if [ -n "${CONVERSATIONAL_MODEL:-}" ]; then
    "${OLLAMA_BIN}" run "${CONVERSATIONAL_MODEL}" <<< "ok" >/dev/null 2>&1 || true
  fi
  # Reflection model: prefer REFLECTION_LLM, else fall back to resolved name
  WARM_REFLECTION="${REFLECTION_LLM:-${REFLECTION_MODEL_NAME:-qwen2.5:3b}}"
  # Normalize potential "ollama/" prefix to the model name that ollama understands
  WARM_REFLECTION="$(echo "$WARM_REFLECTION" | sed 's#^ollama/##')"
  if [ -n "$WARM_REFLECTION" ]; then
    "${OLLAMA_BIN}" run "$WARM_REFLECTION" <<< "ok" >/dev/null 2>&1 || true
  fi
  # Analytical model warmup if configured
  if [ -n "${ANALYTICAL_MODEL:-}" ]; then
    "${OLLAMA_BIN}" run "${ANALYTICAL_MODEL}" <<< "ok" >/dev/null 2>&1 || true
  fi
) &

# After backend port selection and startup, export API_URL for consistency (used by /config.json and frontend)
export API_URL="http://${HOST_IP}:${BACKEND_PORT}"

# Emit an environment snapshot (redact secrets) for diagnostics
log_env_snapshot() {
    local redacted_db="${DATABASE_URL:-}"
    if [ -n "$redacted_db" ]; then
        redacted_db="$(echo "$redacted_db" | sed -E 's#(postgres(ql)?(\+[^:]*)?://[^:]+:)[^@]+#\1*****#')"
    fi
    log "Env Snapshot: HOST_IP=${HOST_IP} SELO_AI_PORT=${SELO_AI_PORT} API_URL=${API_URL} FRONTEND_URL=http://${HOST_IP}:3000 DATABASE_URL=${redacted_db}"
}
log_env_snapshot

# Lightweight backend watchdog: restart uvicorn if it dies unexpectedly
(
    while true; do
        sleep 10
        if ! check_port "$BACKEND_PORT"; then
            log "Backend not listening on $BACKEND_PORT; attempting restart..."
            start_backend
            # give it a moment and print last lines if still down
            sleep 3
            if ! check_port "$BACKEND_PORT"; then
                log "Warning: backend restart did not bind to $BACKEND_PORT yet. Last 50 lines of backend log:"
                tail -n 50 "$BACKEND_LOG" | tee -a "$SERVICE_LOG"
            fi
        fi
    done
) &

# Start Frontend
log "Starting SELO DSP Frontend..."
cd "$SCRIPT_DIR/frontend"

# Derive API base URL for frontend STRICTLY from selected backend port to avoid mismatch
API_HOST="${HOST_IP:-127.0.0.1}"
API_BASE="http://${API_HOST}:${BACKEND_PORT}"
export REACT_APP_API_URL="$API_BASE"
export VITE_API_URL="$API_BASE"

# Preemptive permission repair: remove root-owned node_modules and fix ownership
if [ -d node_modules ] && ! [ -w node_modules ]; then
    log "Frontend node_modules not writable; removing and repairing permissions..."
    rm -rf node_modules || true
fi
# Ensure entire frontend subtree is owned by the current service user
chown -R "$(id -un):$(id -gn)" . 2>/dev/null || true
# Ensure user npm cache dir exists and is writable
mkdir -p "$HOME/.npm" 2>/dev/null || true
chown -R "$(id -un):$(id -gn)" "$HOME/.npm" 2>/dev/null || true
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

# Install frontend dependencies (prefer clean lockfile install)
log "Installing frontend dependencies..."
if [ -f "package-lock.json" ]; then
    if ! npm ci >> "$SERVICE_LOG" 2>&1; then
        log "npm ci failed; cleaning cache and retrying with npm install --legacy-peer-deps"
        npm cache clean --force >> "$SERVICE_LOG" 2>&1 || true
        rm -rf node_modules package-lock.json >> "$SERVICE_LOG" 2>&1 || true
        npm install --legacy-peer-deps >> "$SERVICE_LOG" 2>&1 || true
    fi
else
    if ! npm install --legacy-peer-deps >> "$SERVICE_LOG" 2>&1; then
        log "npm install failed; cleaning cache and retrying..."
        npm cache clean --force >> "$SERVICE_LOG" 2>&1 || true
        rm -rf node_modules >> "$SERVICE_LOG" 2>&1 || true
        npm install --legacy-peer-deps >> "$SERVICE_LOG" 2>&1 || true
    fi
fi

# Build the React app (do not let non-zero exit kill the service)
log "Building React application (API=${VITE_API_URL})..."
set +e
npm run build >> "$SERVICE_LOG" 2>&1
BUILD_STATUS=$?
set -e

FRONTEND_STARTED=false
if [ $BUILD_STATUS -ne 0 ]; then
    log "✗ Frontend build failed (npm exit=$BUILD_STATUS). Skipping frontend startup; backend remains available at $API_BASE."
elif [ ! -d "build" ]; then
    log "✗ Frontend build directory missing. Skipping frontend startup; backend remains available at $API_BASE."
else
    # Start frontend in background using 'serve' if build output exists
    log "Starting React frontend on port 3000..."
    if command -v npx >/dev/null 2>&1; then
        nohup npx --yes serve -s build -l tcp://0.0.0.0:3000 > "$FRONTEND_LOG" 2>&1 &
        FRONTEND_PID=$!
        echo $FRONTEND_PID > "$SCRIPT_DIR/frontend.pid"
        FRONTEND_STARTED=true
    else
        # Ensure a local 'serve' binary is available
        if [ ! -x "node_modules/.bin/serve" ]; then
            log "Installing local 'serve' static server..."
            npm install --no-save serve >> "$SERVICE_LOG" 2>&1 || true
        fi
        if [ -x "node_modules/.bin/serve" ]; then
            nohup node_modules/.bin/serve -s build -l tcp://0.0.0.0:3000 > "$FRONTEND_LOG" 2>&1 &
            FRONTEND_PID=$!
            echo $FRONTEND_PID > "$SCRIPT_DIR/frontend.pid"
            FRONTEND_STARTED=true
        elif command -v serve >/dev/null 2>&1; then
            nohup serve -s build -l tcp://0.0.0.0:3000 > "$FRONTEND_LOG" 2>&1 &
            FRONTEND_PID=$!
            echo $FRONTEND_PID > "$SCRIPT_DIR/frontend.pid"
            FRONTEND_STARTED=true
        else
            log "Warning: no 'serve' available. Skipping frontend startup."
        fi
    fi
fi

# Wait for frontend to start (up to ~30s)
ATTEMPTS=0
MAX_ATTEMPTS=30
until check_port 3000; do
    ATTEMPTS=$((ATTEMPTS+1))
    if [ "$ATTEMPTS" -ge "$MAX_ATTEMPTS" ]; then
        log "✗ Frontend failed to become ready on port 3000 after ${MAX_ATTEMPTS}s"
        log "Last 100 lines of frontend log:"
        tail -n 100 "$FRONTEND_LOG" | tee -a "$SERVICE_LOG"
        log "Continuing with Backend only. You can still use the API at $API_BASE."
        break
    fi
    sleep 1
done
if check_port 3000; then
    log "✓ Frontend is running on http://${HOST_IP:-127.0.0.1}:3000"
fi

log ""
log "========================================="
log "    SELO DSP Digital Sentience Platform Successfully Started!"
log "========================================="
log "Frontend URL:       http://${HOST_IP:-127.0.0.1}:3000"
log "Backend API:        http://${HOST_IP:-127.0.0.1}:${SELO_AI_PORT:-8000}"
log "Chat Model:         ${CONVERSATIONAL_MODEL:-llama3:8b-instruct}"
log "Reflection System:  ${ENABLE_REFLECTION_SYSTEM:-false}"
log "Enhanced Scheduler: ${ENABLE_ENHANCED_SCHEDULER:-false}"
if [ "${ENABLE_REFLECTION_SYSTEM:-false}" = "true" ]; then
    log "Reflection Model:   ${REFLECTION_LLM:-qwen2.5:3b}"
    log "Socket.IO Enabled:  ${SOCKET_IO_ENABLED:-false}"
    log "Vector Embeddings:  ${EMBEDDING_MODEL:-all-MiniLM-L6-v2}"
fi
if [ "${ENABLE_ENHANCED_SCHEDULER:-false}" = "true" ]; then
    log "Scheduler Min Interval: ${SCHEDULER_MIN_INTERVAL:-120} seconds"
    log "Resource Monitoring:    ${RESOURCE_UPDATE_INTERVAL:-60} seconds"
fi
log ""
log "Service is running in background. Check logs:"
log "  Service: $SERVICE_LOG"
log "  Backend: $BACKEND_LOG"
log "  Frontend: $FRONTEND_LOG"
log ""

# Keep the script running (systemd will manage it)
wait
