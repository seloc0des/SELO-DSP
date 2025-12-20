#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

# SELO DSP Uninstall Script (safe and idempotent)
# Removes services, env files, local artifacts, and lingering processes so a fresh
# install can succeed cleanly.
#
# Usage:
#   bash uninstall-selo-ai.sh                # interactive confirmation
#   bash uninstall-selo-ai.sh -y             # non-interactive, assume yes
#   bash uninstall-selo-ai.sh -y --purge-models        # also remove /opt/selo-ai/models
#   bash uninstall-selo-ai.sh -y --revert-firewall     # attempt to remove ufw rules for 8000/3000
#   bash uninstall-selo-ai.sh -y --purge-ollama-models # remove created Ollama models/tags (best-effort)
#
# Notes:
# - System-wide paths touched:
#   /etc/selo-ai/, /etc/systemd/system/selo-ai@.service, /opt/selo-ai (optional)
# - Project-local cleanup in this repo's `selo-ai/` directory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
ASSUME_YES=false
PURGE_MODELS=false
REVERT_FIREWALL=false
PURGE_OLLAMA=false
# New: optionally revert Ollama systemd override created by installer
REVERT_OLLAMA_OVERRIDE=false
# New: optionally purge application database (local PostgreSQL or SQLite files)
PURGE_DB=false
# New: optionally purge per-user beta license files (~/.selo_ai/license)
PURGE_LICENSE=false

print_header() {
  echo "========================================="
  echo "        SELO DSP Uninstall Utility"
  echo "========================================="
}

# Optionally remove beta license files from common locations so fresh installs don't inherit old state
purge_license_files() {
  if [ "$PURGE_LICENSE" != true ]; then return 0; fi
  echo "-- Purging beta license files (~/.selo_ai/license)"
  # Current user
  rm -rf "$HOME/.selo_ai/license" 2>/dev/null || true
  # Root user
  if [ -d "/root" ]; then rm -rf "/root/.selo_ai/license" 2>/dev/null || true; fi
  # All home directories (best-effort)
  for d in /home/*; do
    [ -d "$d" ] || continue
    rm -rf "$d/.selo_ai/license" 2>/dev/null || true
  done
}

confirm() {
  if [ "$ASSUME_YES" = true ]; then return 0; fi
  read -rp "This will remove SELO DSP services, env, and local artifacts. Continue? (y/N): " ans
  [[ $ans =~ ^[Yy]$ ]]
}

parse_args() {
  while (( "$#" )); do
    case "$1" in
      -y|--yes) ASSUME_YES=true; shift ;;
      --purge-models) PURGE_MODELS=true; shift ;;
      --revert-firewall) REVERT_FIREWALL=true; shift ;;
      --purge-ollama-models) PURGE_OLLAMA=true; shift ;;
      --revert-ollama-override) REVERT_OLLAMA_OVERRIDE=true; shift ;;
      --purge-db) PURGE_DB=true; shift ;;
      --purge-license) PURGE_LICENSE=true; shift ;;
      *) echo "Unknown argument: $1"; exit 2 ;;
    esac
  done
}

stop_services() {
  echo "-- Stopping services (if any)"
  # Stop all running selo-ai@*.service instances, not only current user
  if systemctl list-unit-files | grep -q '^selo-ai@.service'; then
    mapfile -t INSTANCES < <(systemctl list-units 'selo-ai@*.service' --all --no-legend | awk '{print $1}')
    for unit in "${INSTANCES[@]}"; do
      [ -n "$unit" ] || continue
      sudo systemctl stop "$unit" 2>/dev/null || true
      sudo systemctl disable "$unit" 2>/dev/null || true
    done
  fi
  # Kill repo-local processes by PID files
  if [ -f "$SCRIPT_DIR/backend.pid" ]; then
    kill "$(cat "$SCRIPT_DIR/backend.pid" 2>/dev/null)" 2>/dev/null || true
    rm -f "$SCRIPT_DIR/backend.pid" || true
  fi
  if [ -f "$SCRIPT_DIR/frontend.pid" ]; then
    kill "$(cat "$SCRIPT_DIR/frontend.pid" 2>/dev/null)" 2>/dev/null || true
    rm -f "$SCRIPT_DIR/frontend.pid" || true
  fi
  # Defensive: kill any child processes of prior starts in this working dir
  pkill -f "python.*selo-ai/backend" 2>/dev/null || true
  pkill -f "node.*selo-ai/frontend" 2>/dev/null || true
}

remove_systemd_unit() {
  echo "-- Removing systemd unit (if installed)"
  if [ -f "/etc/systemd/system/selo-ai@.service" ]; then
    # Stop and disable all running instances
    mapfile -t INSTANCES < <(systemctl list-units 'selo-ai@*.service' --all --no-legend | awk '{print $1}')
    for unit in "${INSTANCES[@]}"; do
      [ -n "$unit" ] || continue
      sudo systemctl stop "$unit" 2>/dev/null || true
      sudo systemctl disable "$unit" 2>/dev/null || true
    done
    sudo rm -f "/etc/systemd/system/selo-ai@.service"
    sudo systemctl daemon-reload || true
  fi
}

free_ports() {
  echo "-- Freeing common SELO DSP ports (best-effort)"
  local TARGET_PORTS=(8000 3000)
  if [ -r "/etc/selo-ai/environment" ]; then
    local cfg_port
    cfg_port=$(awk -F'=' '/^(SELO_AI_PORT|PORT)=/{print $2; exit}' /etc/selo-ai/environment 2>/dev/null || echo "")
    if [ -n "$cfg_port" ]; then TARGET_PORTS+=("$cfg_port"); fi
  fi
  # De-duplicate ports
  local unique_ports=()
  for p in "${TARGET_PORTS[@]}"; do
    local seen=false
    for up in "${unique_ports[@]}"; do [ "$up" = "$p" ] && seen=true && break; done
    [ "$seen" = true ] || unique_ports+=("$p")
  done
  for port in "${unique_ports[@]}"; do
    # List and kill listeners on this port
    if command -v lsof >/dev/null 2>&1; then
      if lsof -iTCP -sTCP:LISTEN -n -P | grep -q ":$port"; then
        echo "   Killing listeners on :$port"
        lsof -ti:$port | xargs -r kill -9 2>/dev/null || true
      fi
    elif command -v ss >/dev/null 2>&1; then
      if ss -ltnp "sport = :$port" 2>/dev/null | tail -n +2 | grep -q ":$port"; then
        echo "   Killing listeners on :$port"
        ss -ltnp "sport = :$port" 2>/dev/null | awk -F',' '/pid=/{for(i=1;i<=NF;i++){if($i ~ /^pid=/){gsub("pid=","",$i); print $i}}}' | xargs -r kill -9 2>/dev/null || true
      fi
    else
      # Fallback
      sudo fuser -k "$port"/tcp 2>/dev/null || true
    fi
  done
}

remove_env_and_state() {
  echo "-- Removing service environment and state"
  # Main env for service
  if [ -d "/etc/selo-ai" ]; then
    sudo rm -rf "/etc/selo-ai"
  fi
}

# Optionally drop application database and user (local PostgreSQL) and delete SQLite files
purge_database() {
  if [ "$PURGE_DB" != true ]; then return 0; fi
  echo "-- Purging application database (optional)"
  local BACKEND_ENV="$PROJECT_ROOT/selo-ai/backend/.env"
  local DB_URL=""
  local DB_DROPPED=false
  if [ -f "$BACKEND_ENV" ]; then
    DB_URL=$(grep -E '^DATABASE_URL=' "$BACKEND_ENV" 2>/dev/null | head -n1)
    DB_URL=${DB_URL#DATABASE_URL=}
  fi
  # Also check service env if backend/.env is missing
  if [ -z "$DB_URL" ] && [ -r "/etc/selo-ai/environment" ]; then
    DB_URL=$(awk -F'=' '/^DATABASE_URL=/{print $2; exit}' /etc/selo-ai/environment 2>/dev/null || echo "")
  fi

  # Remove SQLite local files if present regardless of URL
  find "$PROJECT_ROOT/selo-ai/backend" -maxdepth 2 -type f \( -iname "*.sqlite" -o -iname "*.sqlite3" -o -iname "*.db" \) -print -exec rm -f {} \; 2>/dev/null || true

  if [ -z "$DB_URL" ]; then
    echo "   No DATABASE_URL found; skipped dropping PostgreSQL database."
    drop_application_tables ""
    return 0
  fi

  # Only handle local PostgreSQL URLs
  if echo "$DB_URL" | grep -Eiq '^postgres|^postgresql'; then
    # Extract components safely
    local DB_HOST DB_USER DB_PASS DB_NAME DRIVER
    DRIVER=$(echo "$DB_URL" | sed -E 's|^([^:]+):.*$|\1|')
    DB_HOST=$(echo "$DB_URL" | sed -E 's|^postgres(ql)?(\+[^:]*)?://[^@]+@([^/:?#]+).*$|\3|')
    DB_USER=$(echo "$DB_URL" | sed -E 's|^postgres(ql)?(\+[^:]*)?://([^/:]+):([^@]+)@.*$|\3|')
    DB_PASS=$(echo "$DB_URL" | sed -E 's|^postgres(ql)?(\+[^:]*)?://([^/:]+):([^@]+)@.*$|\4|')
    DB_NAME=$(echo "$DB_URL" | sed -E 's|^.*/([^/?#]+)(\?.*)?$|\1|')
    # Only drop if host is local
    if echo "$DB_HOST" | grep -Eq '^(localhost|127\.0\.0\.1|::1)$'; then
      echo "   Targeting local PostgreSQL database '$DB_NAME' (user '$DB_USER')."
      if [ "$ASSUME_YES" != true ]; then
        read -rp "   Drop database '$DB_NAME' and user '$DB_USER'? This cannot be undone. (y/N): " ans
        [[ $ans =~ ^[Yy]$ ]] || { echo "   Skipping database drop."; return 0; }
      fi
      # Perform drop commands
      # Create a temporary PSQL command to avoid exposing password in process table
      if command -v psql >/dev/null 2>&1; then
        # Drop active connections and then drop DB and role
        echo "   Terminating active connections to database '$DB_NAME'..."
        sudo -u postgres psql -d postgres -v ON_ERROR_STOP=1 -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE datname='${DB_NAME}'" 2>/dev/null || true
        
        echo "   Dropping database '$DB_NAME'..."
        if sudo -u postgres psql -d postgres -v ON_ERROR_STOP=1 -c "DROP DATABASE IF EXISTS \"${DB_NAME}\";"; then
          DB_DROPPED=true
        else
          echo "   Warning: DROP DATABASE command reported an error (see above)."
        fi
        
        echo "   Dropping user '$DB_USER'..."
        sudo -u postgres psql -d postgres -v ON_ERROR_STOP=1 -c "DROP ROLE IF EXISTS \"${DB_USER}\";" || true
        
        # Also clear any potential cached connections or schemas
        echo "   Clearing PostgreSQL connection cache..."
        sudo systemctl reload postgresql 2>/dev/null || true
        
        local EXISTS_FLAG
        EXISTS_FLAG=$(sudo -u postgres psql -d postgres -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" 2>/dev/null || true)
        if [ -z "$EXISTS_FLAG" ]; then
          DB_DROPPED=true
          echo "   ✓ Dropped local PostgreSQL database '$DB_NAME' and user '$DB_USER' (if existed)."
          echo "   ✓ All conversation history, persona data, and memories have been completely purged."
          echo "   ✓ Database will be recreated fresh on next installation."
        else
          echo "   ⚠️  Database '$DB_NAME' still exists after drop attempt; will fallback to table-level cleanup."
        fi
      else
        echo "   psql not found; cannot drop PostgreSQL database automatically."
      fi
    else
      echo "   DATABASE_URL host is non-local ('$DB_HOST'); will not drop remote databases."
    fi
  fi

  if [ "$DB_DROPPED" != true ]; then
    drop_application_tables "$DB_URL"
  fi
}

drop_application_tables() {
  local URL="$1"
  echo "-- Dropping SELO DSP tables (best-effort)"
  if ! command -v python3 >/dev/null 2>&1; then
    echo "   python3 not found; cannot drop tables automatically."
    return 0
  fi

  # Prefer URL argument; fallback to environment/service files if missing
  if [ -z "$URL" ]; then
    local BACKEND_ENV="$PROJECT_ROOT/selo-ai/backend/.env"
    if [ -f "$BACKEND_ENV" ]; then
      URL=$(grep -E '^DATABASE_URL=' "$BACKEND_ENV" 2>/dev/null | head -n1)
      URL=${URL#DATABASE_URL=}
    fi
    if [ -z "$URL" ] && [ -r "/etc/selo-ai/environment" ]; then
      URL=$(awk -F'=' '/^DATABASE_URL=/{print $2; exit}' /etc/selo-ai/environment 2>/dev/null || echo "")
    fi
  fi

  if [ -z "$URL" ]; then
    echo "   No DATABASE_URL available; skipping table drop."
    return 0
  fi

  local PYTHONPATH_BAK="${PYTHONPATH:-}"
  export PYTHONPATH="$SCRIPT_DIR/backend:$SCRIPT_DIR:$PYTHONPATH"
  DATABASE_URL="$URL" python3 <<'PY'
import asyncio
import os
from sqlalchemy import text

try:
    from backend.db.session import engine
except Exception as exc:
    print(f"   Unable to import backend DB session for cleanup: {exc}")
    raise SystemExit(0)

TABLES_IN_DROP_ORDER = [
    "relationship_question_queue",
    "reflection_schedule",
    "reflection_memories",
    "reflections",
    "concept_connections",
    "learning_concept",
    "concepts",
    "learnings",
    "persona_concept_association",
    "persona_traits",
    "persona_evolutions",
    "personas",
    "memories",
    "conversation_messages",
    "conversations",
    "users",
]

async def drop_tables():
    async with engine.begin() as conn:
        for table in TABLES_IN_DROP_ORDER:
            try:
                await conn.execute(text(f'DROP TABLE IF EXISTS {table} CASCADE;'))
                print(f"   Dropped table {table} (if existed)")
            except Exception as exc:
                print(f"   Warning: failed to drop table {table}: {exc}")
    await engine.dispose()

asyncio.run(drop_tables())
PY
  export PYTHONPATH="$PYTHONPATH_BAK"
}

# Optionally remove Ollama systemd override created by installer and restart daemon
revert_ollama_override() {
  if [ "$REVERT_OLLAMA_OVERRIDE" != true ]; then return 0; fi
  echo "-- Reverting Ollama systemd override"
  local dropin_dir="/etc/systemd/system/ollama.service.d"
  local override_file="$dropin_dir/override.conf"
  if [ -f "$override_file" ]; then
    sudo rm -f "$override_file" || true
    # Remove directory if empty
    if [ -d "$dropin_dir" ] && [ -z "$(ls -A "$dropin_dir" 2>/dev/null)" ]; then
      sudo rmdir "$dropin_dir" 2>/dev/null || true
    fi
    sudo systemctl daemon-reload || true
    sudo systemctl restart ollama 2>/dev/null || true
  else
    echo "   No Ollama override file found (nothing to revert)"
  fi
}

revert_firewall_rules() {
  if [ "$REVERT_FIREWALL" != true ]; then return 0; fi
  echo "-- Reverting firewall rules (ufw)"
  if command -v ufw >/dev/null 2>&1; then
    # Best-effort removal for typical ports and detected configured port
    TARGET_PORT=""
    if [ -r "/etc/selo-ai/environment" ]; then
      TARGET_PORT=$(awk -F'=' '/^(SELO_AI_PORT|PORT)=/{print $2; exit}' /etc/selo-ai/environment 2>/dev/null || echo "")
    fi
    # Delete generic rules
    sudo ufw delete allow 8000/tcp 2>/dev/null || true
    sudo ufw delete allow 3000/tcp 2>/dev/null || true
    # Delete detected backend port rule if different
    if [ -n "$TARGET_PORT" ] && [ "$TARGET_PORT" != "8000" ]; then
      sudo ufw delete allow "$TARGET_PORT"/tcp 2>/dev/null || true
    fi
  fi
}

purge_models_dir() {
  if [ "$PURGE_MODELS" != true ]; then return 0; fi
  echo "-- Purging models directory under /opt/selo-ai (optional)"
  if [ -d "/opt/selo-ai" ]; then
    sudo rm -rf "/opt/selo-ai"
  fi
}

purge_ollama_models() {
  if [ "$PURGE_OLLAMA" != true ]; then return 0; fi
  echo "-- Purging Ollama models/tags (best-effort)"
  if command -v ollama >/dev/null 2>&1; then
    # Build removal list from environment if present
    ENV_FILE="$PROJECT_ROOT/selo-ai/backend/.env"
    SVC_ENV="/etc/selo-ai/environment"
    MODELS_TO_REMOVE=()
    add_model() { local m="$1"; [ -n "$m" ] && MODELS_TO_REMOVE+=("$m"); }
    if [ -f "$ENV_FILE" ]; then
      add_model "$(grep -E '^CONVERSATIONAL_MODEL=' "$ENV_FILE" | tail -n1 | cut -d'=' -f2-)"
      add_model "$(grep -E '^REFLECTION_LLM=' "$ENV_FILE" | tail -n1 | cut -d'=' -f2-)"
      add_model "$(grep -E '^ANALYTICAL_MODEL=' "$ENV_FILE" | tail -n1 | cut -d'=' -f2-)"
      add_model "$(grep -E '^EMBEDDING_MODEL=' "$ENV_FILE" | tail -n1 | cut -d'=' -f2-)"
    fi
    if [ -r "$SVC_ENV" ]; then
      add_model "$(awk -F'=' '/^CONVERSATIONAL_MODEL=/{print $2; exit}' "$SVC_ENV" 2>/dev/null)"
      add_model "$(awk -F'=' '/^REFLECTION_LLM=/{print $2; exit}' "$SVC_ENV" 2>/dev/null)"
      add_model "$(awk -F'=' '/^ANALYTICAL_MODEL=/{print $2; exit}' "$SVC_ENV" 2>/dev/null)"
      add_model "$(awk -F'=' '/^EMBEDDING_MODEL=/{print $2; exit}' "$SVC_ENV" 2>/dev/null)"
    fi
    # Include common defaults used by installer as a safety net
    MODELS_TO_REMOVE+=(
      "humanish-llama3"
      "humanish-llama3:8b-q4"
      "phi3:mini-4k-instruct"
      "qwen2.5-coder:3b"
      "nomic-embed-text"
    )
    # De-duplicate
    UNIQUE_MODELS=()
    for m in "${MODELS_TO_REMOVE[@]}"; do
      [ -z "$m" ] && continue
      seen=false
      for um in "${UNIQUE_MODELS[@]}"; do [ "$um" = "$m" ] && seen=true && break; done
      [ "$seen" = true ] || UNIQUE_MODELS+=("$m")
    done
    # Attempt removal
    for name in "${UNIQUE_MODELS[@]}"; do
      # Strip any provider prefix like ollama/
      short_name="$(echo "$name" | awk -F'/' '{print $NF}')"
      ollama rm "$short_name" 2>/dev/null || true
      # Also try :latest to catch tagged bases
      ollama rm "$short_name:latest" 2>/dev/null || true
    done
  fi
}

clean_project_local() {
  echo "-- Cleaning project-local artifacts"
  # Backend venv and caches
  if [ -d "$SCRIPT_DIR/backend/venv" ]; then rm -rf "$SCRIPT_DIR/backend/venv"; fi
  rm -rf "$SCRIPT_DIR/backend/__pycache__" 2>/dev/null || true
  # Persistent vector store (FAISS index and metadata)
  rm -rf "$SCRIPT_DIR/backend/data/vector_store" 2>/dev/null || true
  # If data directory becomes empty, remove it as well
  if [ -d "$SCRIPT_DIR/backend/data" ] && [ -z "$(ls -A "$SCRIPT_DIR/backend/data" 2>/dev/null)" ]; then
    rmdir "$SCRIPT_DIR/backend/data" 2>/dev/null || true
  fi
  # Frontend node_modules and build outputs (Vite/React)
  rm -rf "$SCRIPT_DIR/frontend/node_modules" 2>/dev/null || true
  rm -rf "$SCRIPT_DIR/frontend/dist" 2>/dev/null || true
  rm -rf "$SCRIPT_DIR/frontend/build" 2>/dev/null || true
  rm -f "$SCRIPT_DIR/frontend/.npmrc" 2>/dev/null || true
  # Logs and pid files
  rm -rf "$SCRIPT_DIR/logs" 2>/dev/null || true
  rm -f "$SCRIPT_DIR"/*.pid 2>/dev/null || true
  # Remove env files and port marker for a true fresh install
  rm -f "$SCRIPT_DIR/backend/.env" 2>/dev/null || true
  rm -f "$SCRIPT_DIR/frontend/.env" 2>/dev/null || true
  rm -f "$SCRIPT_DIR/backend.port" 2>/dev/null || true
  # Remove first introduction marker so fresh installs can introduce themselves
  rm -f "$SCRIPT_DIR/.first_intro_done" 2>/dev/null || true
  # Remove boot reflection adoption marker so fresh installs can adopt/emit once
  rm -f "$SCRIPT_DIR/.boot_reflection_adopted" 2>/dev/null || true
}

summary() {
  echo ""
  echo "✓ Uninstall complete. You can now run a fresh install."
  echo "Optional cleanup flags you can use:"
  echo "  --revert-ollama-override    # remove Ollama daemon override and restart it"
  echo "  --purge-ollama-models       # remove Ollama models/tags created by installer"
  echo "  --revert-firewall           # remove ufw rules for backend/frontend ports"
  echo "  --purge-models              # remove /opt/selo-ai directory"
  echo "  --purge-db                  # drop local PostgreSQL database/user and delete local SQLite files"
  echo ""
}

main() {
  print_header
  parse_args "$@"
  if ! confirm; then echo "Cancelled."; exit 0; fi
  stop_services
  remove_systemd_unit
  remove_env_and_state
  purge_database
  revert_ollama_override
  revert_firewall_rules
  purge_models_dir
  purge_ollama_models
  purge_license_files
  clean_project_local
  free_ports
  summary
}

main "$@"
