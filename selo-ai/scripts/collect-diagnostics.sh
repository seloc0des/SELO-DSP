#!/usr/bin/env bash
# Collect fresh diagnostics for SELO DSP.
# Simplified: by default writes under a temp directory (/tmp) to avoid any repo permission issues.
# Use --out <dir> if you want to choose a specific output folder.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

# Lightweight CLI parsing (only --out supported)
OUT_OVERRIDE=""
while [[ ${1:-} == --* ]]; do
  case "$1" in
    --out)
      OUT_OVERRIDE="${2:-}"
      shift 2 || true
      ;;
    --help|-h)
      echo "Usage: $0 [--out <dir>]"; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2; shift || true ;;
  esac
done

TIMESTAMP="$(date '+%Y-%m-%d_%H-%M-%S')"
# Default output parent to a temp area; can be overridden via --out or SELO_DIAG_OUT
DEFAULT_PARENT="${TMPDIR:-/tmp}/seloai-diag"
OUT_PARENT="${OUT_OVERRIDE:-${SELO_DIAG_OUT:-$DEFAULT_PARENT}}"
OUT_DIR="$OUT_PARENT/$TIMESTAMP"

# Ensure output parent and timestamped dir exist
mkdir -p "$OUT_PARENT" "$OUT_DIR"

note() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

note "Writing diagnostics to: $OUT_DIR"

# Systemd status and recent logs (robust instance detection)
detect_instance_user() {
  # Prefer the non-root user when running under sudo
  if [ -n "${SUDO_USER:-}" ] && [ "$SUDO_USER" != "root" ]; then
    echo "$SUDO_USER"; return
  fi
  # Try loginctl for the active seat
  if command -v loginctl >/dev/null 2>&1; then
    u=$(loginctl list-sessions 2>/dev/null | awk 'NR>1 {print $3}' | head -n1)
    if [ -n "$u" ] && [ "$u" != "root" ]; then echo "$u"; return; fi
  fi
  # Attempt to infer from a running uvicorn process
  u=$(ps aux | awk '/uvicorn .*backend\.main|get_socketio_app/ && $1!~/_apt|root/ {print $1; exit}')
  if [ -n "$u" ]; then echo "$u"; return; fi
  # Fallback to current user
  echo "$(whoami)"
}

INST_USER="$(detect_instance_user)"
SYSTEMD_UNIT="selo-ai@${INST_USER}"
systemctl status "$SYSTEMD_UNIT" --no-pager > "$OUT_DIR/systemd.txt" 2>&1 || true
# Capture live journal for the active instance (separate file)
journalctl -u "$SYSTEMD_UNIT" -b -n 500 --no-pager > "$OUT_DIR/instance_journal.txt" 2>&1 || true
# Capture ExecStart/Environment for clarity
systemctl show "$SYSTEMD_UNIT" -p ExecStart,Environment,FragmentPath > "$OUT_DIR/systemd_show.txt" 2>&1 || true

# Network listeners and processes
{
  ss -ltnp || true
  echo "\n---\n"
  ps aux | egrep 'uvicorn|python -m backend\.main|serve -s|node|npm' | grep -v egrep || true
} > "$OUT_DIR/backend_http.txt" 2>&1

# Backend/Frontend logs (if present)
BACKEND_LOG="$ROOT_DIR/logs/backend.log"
FRONTEND_LOG="$ROOT_DIR/logs/frontend.log"
{
  echo "== backend.log (last 300) =="
  [ -f "$BACKEND_LOG" ] && tail -n 300 "$BACKEND_LOG" || echo "(missing)"
  echo "\n== frontend.log (last 300) =="
  [ -f "$FRONTEND_LOG" ] && tail -n 300 "$FRONTEND_LOG" || echo "(missing)"
} > "$OUT_DIR/logs.txt" 2>&1

# Resolve configured backend port and base URLs
resolve_backend_port() {
  # Priority: SELO_AI_PORT in /etc/selo-ai/environment, then API_URL port, then 8000
  if [ -f "/etc/selo-ai/environment" ]; then
    p=$(grep -E '^SELO_AI_PORT=' /etc/selo-ai/environment | tail -n1 | sed -E 's/^SELO_AI_PORT=//')
    if [ -n "$p" ]; then echo "$p"; return; fi
    api=$(grep -E '^API_URL=' /etc/selo-ai/environment | sed -E 's/^API_URL=//; s#^https?://##; s#/.*$##' | head -n1)
    if echo "$api" | grep -q ':'; then echo "$api" | cut -d: -f2; return; fi
  fi
  echo "8000"
}

BE_PORT="$(resolve_backend_port)"
LOCAL_BASE="http://127.0.0.1:${BE_PORT}"

# Health checks (best-effort)
{
  # Try to resolve LAN host IP from environment file; fallback to first non-loopback IP
  HOST_IP=""
  if [ -f "/etc/selo-ai/environment" ]; then
    HOST_IP=$(grep -E '^HOST_IP=' /etc/selo-ai/environment | tail -n1 | sed -E 's/^HOST_IP=//')
    [ -z "$HOST_IP" ] && HOST_IP=$(grep -E '^API_URL=' /etc/selo-ai/environment | sed -E 's/^API_URL=http:\/\///; s/:.*$//' | head -n1)
  fi
  if [ -z "$HOST_IP" ]; then
    HOST_IP=$(hostname -I 2>/dev/null | awk '{for(i=1;i<=NF;i++){if($i!~/^127\./){print $i; exit}}}')
  fi
  HOST_IP=${HOST_IP:-127.0.0.1}

  echo "== /health =="
  curl -sS "$LOCAL_BASE/health" || true
  echo "\n---\n== /api/health =="
  curl -sS "$LOCAL_BASE/api/health" || true
  echo "\n---\n== / (root) =="
  curl -sS "$LOCAL_BASE/" || true
  echo "\n---\n== /health (LAN) =="
  curl -sS "http://${HOST_IP}:${BE_PORT}/health" || true
} > "$OUT_DIR/health.txt" 2>&1

# Backend extended diagnostics and API schema
{
  echo "== /diagnostics/env =="
  curl -sS "$LOCAL_BASE/diagnostics/env" || true
  echo "\n---\n== /diagnostics/gpu?test_llm=true&model_role=reflection =="
  curl -sS "$LOCAL_BASE/diagnostics/gpu?test_llm=true&model_role=reflection" || true
  echo "\n---\n== /health/details (no probes) =="
  curl -sS "$LOCAL_BASE/health/details" || true
  echo "\n---\n== /health/details?probe_db=true =="
  curl -sS "$LOCAL_BASE/health/details?probe_db=true" || true
  echo "\n---\n== /openapi.json =="
  curl -sS "$LOCAL_BASE/openapi.json" | head -n 400 || true
} > "$OUT_DIR/backend_api.txt" 2>&1

# Frontend runtime config served by backend (useful when frontend is built without .env)
{
  echo "== /config.json =="
  curl -sS "$LOCAL_BASE/config.json" || true
} > "$OUT_DIR/frontend_config.txt" 2>&1

# Socket.IO basic probe (Engine.IO polling)
{
  echo "== /socket.io (polling) =="
  curl -sS "$LOCAL_BASE/socket.io/?EIO=4&transport=polling" || true
} > "$OUT_DIR/socketio.txt" 2>&1

# End-to-end functional diagnostics
{
  echo "== E2E functional diagnostics =="
  BASE_URL="http://127.0.0.1:8000"
  # If API_URL is configured in system environment, prefer it
  if [ -f "/etc/selo-ai/environment" ]; then
    CONF_API_URL=$(grep -E '^API_URL=' /etc/selo-ai/environment | sed -E 's/^API_URL=//') || true
    [ -n "$CONF_API_URL" ] && BASE_URL="$CONF_API_URL"
  elif [ -n "${API_URL:-}" ]; then
    BASE_URL="$API_URL"
  fi

  # System key for protected endpoints (optional; tests will be skipped if not present)
  SYS_KEY="${SELO_SYSTEM_API_KEY:-}"
  if [ -z "$SYS_KEY" ] && [ -f "/etc/selo-ai/environment" ]; then
    SYS_KEY=$(grep -E '^SELO_SYSTEM_API_KEY=' /etc/selo-ai/environment | sed -E 's/^SELO_SYSTEM_API_KEY=//') || true
  fi

  # Brave key (optional for search test)
  BRAVE_KEY="${BRAVE_SEARCH_API_KEY:-}"
  if [ -z "$BRAVE_KEY" ] && [ -f "/etc/selo-ai/environment" ]; then
    BRAVE_KEY=$(grep -E '^BRAVE_SEARCH_API_KEY=' /etc/selo-ai/environment | sed -E 's/^BRAVE_SEARCH_API_KEY=//') || true
  fi

  # Write E2E artifacts directly into the timestamped root to avoid subdir quirks
  E2E_DIR="$OUT_DIR"
  SUMMARY="$OUT_DIR/e2e_summary.txt"
  DETAILS="$OUT_DIR/e2e_details.txt"
  E2E_STDERR="$OUT_DIR/e2e_stdout.txt"
  : > "$SUMMARY"
  : > "$DETAILS"
  : > "$E2E_STDERR"

  pass() { echo "PASS - $1" | tee -a "$SUMMARY" >/dev/null; }
  fail() { echo "FAIL - $1" | tee -a "$SUMMARY" >/dev/null; }
  skip() { echo "SKIP - $1" | tee -a "$SUMMARY" >/dev/null; }

  # Helper: curl JSON, capture status code and body, writing via temp file to avoid EACCES noise
  curl_json() {
    local url="$1"; shift
    local outfile="$1"; shift
    local outdir
    outdir=$(dirname "$outfile")
    # Ensure target directory exists
    mkdir -p "$outdir" 2>/dev/null || true
    # Try to ensure directory is writable by current user
    chmod u+rwX "$outdir" 2>/dev/null || true
    # Create a temp file for the download in the preferred directory; fallback to OUT_DIR
    local tmpfile=""
    if tmpfile=$(mktemp -p "$outdir" 2>/dev/null); then
      :
    else
      tmpfile=$(mktemp -p "$OUT_DIR" 2>/dev/null || echo "$OUT_DIR/tmp.$$.")
    fi
    # Timeouts: allow override per-call via CURL_TIMEOUT_OVERRIDE_S; defaults otherwise
    local CONNECT_TIMEOUT_S
    local CURL_TIMEOUT_S
    CONNECT_TIMEOUT_S=${CONNECT_TIMEOUT_S:-5}
    CURL_TIMEOUT_S=${CURL_TIMEOUT_OVERRIDE_S:-${CURL_TIMEOUT_S:-30}}
    local code
    code=$(curl --connect-timeout "$CONNECT_TIMEOUT_S" --max-time "$CURL_TIMEOUT_S" -sS -o "$tmpfile" -w '%{http_code}' "$@" "$url" 2>>"$E2E_STDERR" || true)
    # Move into place; suppress errors to avoid noisy diagnostics
    if mv -f "$tmpfile" "$outfile" 2>>"$E2E_STDERR" || cp -f "$tmpfile" "$outfile" 2>>"$E2E_STDERR" || { mkdir -p "$outdir" 2>/dev/null; chmod ugo+rwX "$outdir" 2>/dev/null; cat "$tmpfile" > "$outfile" 2>>"$E2E_STDERR"; }; then
      :
    else
      # Final fallback: write to timestamped root with the same basename
      local base
      base=$(basename "$outfile")
      local alt_out="$OUT_DIR/$base"
      if mv -f "$tmpfile" "$alt_out" 2>>"$E2E_STDERR" || cp -f "$tmpfile" "$alt_out" 2>>"$E2E_STDERR" || cat "$tmpfile" > "$alt_out" 2>>"$E2E_STDERR"; then
        # Record that we fell back so users know where to find the file
        if [ -n "${DETAILS:-}" ]; then
          echo "fallback_write:$base -> $alt_out" >> "$DETAILS" 2>/dev/null || true
        fi
      fi
    fi
    # Best-effort cleanup if tmp remained
    rm -f "$tmpfile" 2>/dev/null || true
    echo "$code"
  }

  # Disable exit-on-error during E2E to ensure we gather as much as possible
  set +e

  echo "BASE_URL=$BASE_URL" | tee -a "$DETAILS" >/dev/null

  # 1) Health
  CODE=$(curl_json "$BASE_URL/health" "$E2E_DIR/health.json")
  if [[ "$CODE" =~ ^2 ]]; then pass "GET /health ($CODE)"; else fail "GET /health ($CODE)"; fi

  # 1b) Diagnostics env + assert reflection timeout cap
  CODE=$(curl_json "$BASE_URL/diagnostics/env" "$E2E_DIR/diag_env.json")
  if [[ "$CODE" =~ ^2 ]]; then
    # Assert REFLECTION_SYNC_TIMEOUT_S == "60"
    RT=$(python3 - <<'PY'
import json,sys
try:
  d=json.load(open(sys.argv[1]));
  print(str(d.get('REFLECTION_SYNC_TIMEOUT_S','')))
except Exception:
  print('')
PY
"$E2E_DIR/diag_env.json")
    echo "REFLECTION_SYNC_TIMEOUT_S=$RT" >> "$DETAILS"
    if [ "$RT" = "60" ]; then pass "diagnostics/env: REFLECTION_SYNC_TIMEOUT_S=60"; else fail "diagnostics/env: REFLECTION_SYNC_TIMEOUT_S expected 60 got '$RT'"; fi
  else
    fail "GET /diagnostics/env ($CODE)"
  fi

  # 1c) GPU diagnostics with test LLM
  CURL_TIMEOUT_OVERRIDE_S=180 CODE=$(curl_json "$BASE_URL/diagnostics/gpu?test_llm=true&model_role=reflection" "$E2E_DIR/gpu.json"); unset CURL_TIMEOUT_OVERRIDE_S
  if [[ "$CODE" =~ ^2 ]]; then pass "GET /diagnostics/gpu?test_llm=true ($CODE)";
    # Extract a few fields for details
    python3 - <<'PY' "$E2E_DIR/gpu.json" >>"$DETAILS" 2>/dev/null || true
import json,sys
try:
  d=json.load(open(sys.argv[1]));
  oll=d.get('ollama',{});
  ref=d.get('reflection',{});
  print(f"ollama.num_thread={oll.get('num_thread')} gpu_layers={oll.get('gpu_layers')} keep_alive={oll.get('keep_alive')}")
  print(f"reflection.sync_timeout_s={ref.get('sync_timeout_s')} status={d.get('status')} tested_llm={d.get('tested_llm')} test_duration_s={d.get('test_duration_s')}")
except Exception as e:
  print(f"gpu_diagnostics_parse_error={e}")
PY
  else
    fail "GET /diagnostics/gpu?test_llm=true ($CODE)"
  fi

  # 2) Health details (DB)
  CODE=$(curl_json "$BASE_URL/health/details?probe_db=true" "$E2E_DIR/health_db.json")
  if [[ "$CODE" =~ ^2 ]]; then pass "GET /health/details?probe_db=true ($CODE)"; else fail "GET /health/details?probe_db=true ($CODE)"; fi

  # 3) Health details (LLM)
  CODE=$(curl_json "$BASE_URL/health/details?probe_llm=true" "$E2E_DIR/health_llm.json")
  if [[ "$CODE" =~ ^2 ]]; then pass "GET /health/details?probe_llm=true ($CODE)"; else fail "GET /health/details?probe_llm=true ($CODE)"; fi

  # 4) Runtime config
  CODE=$(curl_json "$BASE_URL/config.json" "$E2E_DIR/config.json")
  if [[ "$CODE" =~ ^2 ]]; then pass "GET /config.json ($CODE)"; else fail "GET /config.json ($CODE)"; fi

  # 5) Chat flow
  CHAT_PAYLOAD='{"session_id":"e2e-session","prompt":"Say hello in one short line.","model":"mistral:latest"}'
  CODE=$(curl -sS -o "$E2E_DIR/chat.json" -w '%{http_code}' -H 'Content-Type: application/json' -d "$CHAT_PAYLOAD" "$BASE_URL/chat" || true)
  if [[ "$CODE" =~ ^2 ]]; then pass "POST /chat ($CODE)"; else fail "POST /chat ($CODE)"; fi

  # 6) Persona default
  CODE=$(curl_json "$BASE_URL/persona/default" "$E2E_DIR/persona_default.json")
  if [[ "$CODE" =~ ^2 ]]; then pass "GET /persona/default ($CODE)"; else fail "GET /persona/default ($CODE)"; fi

  # Extract persona_id and user_id (best-effort) for subsequent tests
  PER_ID=$(python3 - <<'PY'
import json,sys
try:
  data=json.load(open(sys.argv[1]))
  print(((data.get('data') or {}).get('persona') or {}).get('id',''))
except Exception:
  print('')
PY
"$E2E_DIR/persona_default.json")

  USER_ID=$(python3 - <<'PY'
import json,sys
try:
  data=json.load(open(sys.argv[1]))
  print(((data.get('data') or {}).get('persona') or {}).get('user_id',''))
except Exception:
  print('')
PY
"$E2E_DIR/persona_default.json")

  echo "persona_id=$PER_ID user_id=$USER_ID" >> "$DETAILS"

  # 7) Persona traits (default)
  CODE=$(curl_json "$BASE_URL/persona/default/traits" "$E2E_DIR/persona_traits.json")
  if [[ "$CODE" =~ ^2 ]]; then pass "GET /persona/default/traits ($CODE)"; else fail "GET /persona/default/traits ($CODE)"; fi

  # 8) Reflection generate (requires system key)
  if [ -n "$SYS_KEY" ] && [ -n "$USER_ID" ]; then
    REQ='{"reflection_type":"message","user_profile_id":"'"$USER_ID"'","memory_ids":null,"trigger_source":"system"}'
    CODE=$(curl -sS -o "$E2E_DIR/reflection_generate.json" -w '%{http_code}' -H 'Content-Type: application/json' -H "X-API-KEY: $SYS_KEY" -d "$REQ" "$BASE_URL/reflection/generate" || true)
    if [[ "$CODE" =~ ^2 ]]; then pass "POST /reflection/generate ($CODE)"; else fail "POST /reflection/generate ($CODE)"; fi
  else
    skip "POST /reflection/generate (missing system key or user_id)"
  fi

  # 9) Persona reassert (requires system key)
  if [ -n "$SYS_KEY" ]; then
    CODE=$(curl -sS -o "$E2E_DIR/persona_reassert.json" -w '%{http_code}' -H "X-API-KEY: $SYS_KEY" -X POST "$BASE_URL/persona/default/reassert" || true)
    if [[ "$CODE" =~ ^2 ]]; then pass "POST /persona/default/reassert ($CODE)"; else fail "POST /persona/default/reassert ($CODE)"; fi
  else
    skip "POST /persona/default/reassert (missing system key)"
  fi

  # 10) Search (optional; requires Brave key configured server-side; backend enforces this)
  if [ -n "$BRAVE_KEY" ]; then
    CODE=$(curl -sS -o "$E2E_DIR/search.json" -w '%{http_code}' "$BASE_URL/search?query=latest%20news" || true)
    if [[ "$CODE" =~ ^2 ]]; then pass "GET /search ($CODE)"; else fail "GET /search ($CODE)"; fi
  else
    skip "GET /search (no Brave key)"
  fi

  # 11) Socket.IO handshake (polling)
  CODE=$(curl -sS -o "$E2E_DIR/socketio_poll.txt" -w '%{http_code}' "$BASE_URL/socket.io/?EIO=4&transport=polling" || true)
  if [[ "$CODE" =~ ^2 ]]; then pass "GET /socket.io polling ($CODE)"; else fail "GET /socket.io polling ($CODE)"; fi

  # 12) Persona system-prompt (requires persona_id)
  if [ -n "$PER_ID" ]; then
    CODE=$(curl_json "$BASE_URL/persona/system-prompt/$PER_ID" "$E2E_DIR/persona_system_prompt.json")
    if [[ "$CODE" =~ ^2 ]]; then pass "GET /persona/system-prompt/{persona_id} ($CODE)"; else fail "GET /persona/system-prompt/{persona_id} ($CODE)"; fi
  else
    skip "GET /persona/system-prompt/{persona_id} (no persona_id)"
  fi

  # 13) Persona history (requires persona_id)
  if [ -n "$PER_ID" ]; then
    CODE=$(curl_json "$BASE_URL/persona/history/$PER_ID" "$E2E_DIR/persona_history.json")
    if [[ "$CODE" =~ ^2 ]]; then pass "GET /persona/history/{persona_id} ($CODE)"; else fail "GET /persona/history/{persona_id} ($CODE)"; fi
  else
    skip "GET /persona/history/{persona_id} (no persona_id)"
  fi

  # 14) Personas for user (requires user_id)
  if [ -n "$USER_ID" ]; then
    CODE=$(curl_json "$BASE_URL/persona/user/$USER_ID" "$E2E_DIR/persona_for_user.json")
    if [[ "$CODE" =~ ^2 ]]; then pass "GET /persona/user/{user_id} ($CODE)"; else fail "GET /persona/user/{user_id} ($CODE)"; fi
  else
    skip "GET /persona/user/{user_id} (no user_id)"
  fi

  # 15) Reflections list (requires user_id)
  if [ -n "$USER_ID" ]; then
    CODE=$(curl_json "$BASE_URL/reflection/list?user_profile_id=$USER_ID&limit=5" "$E2E_DIR/reflection_list.json")
    if [[ "$CODE" =~ ^2 ]]; then pass "GET /reflection/list ($CODE)"; else fail "GET /reflection/list ($CODE)"; fi
  else
    skip "GET /reflection/list (no user_id)"
  fi

  # 16) Events publish (requires system key and scheduler availability)
  if [ -n "$SYS_KEY" ]; then
    # Include timestamp and optional user_id
    UID_FIELD=""
    if [ -n "$USER_ID" ]; then
      UID_FIELD=",\"user_id\":\"$USER_ID\""
    fi
    EVENT='{'\
"event_type":"e2e_test",'\
,"event_data":{"source":"collect-diagnostics","ts":"'$TIMESTAMP'"}'\
$UID_FIELD'\
}'
    CODE=$(curl -sS -o "$E2E_DIR/event_publish.json" -w '%{http_code}' -H "X-API-KEY: $SYS_KEY" -H 'Content-Type: application/json' -d "$EVENT" "$BASE_URL/events/publish" || true)
    if [[ "$CODE" =~ ^2 ]]; then pass "POST /events/publish ($CODE)"; else fail "POST /events/publish ($CODE)"; fi
  else
    skip "POST /events/publish (missing system key)"
  fi

} > "$OUT_DIR/e2e_stdout.txt" 2>&1

# Backend layout snapshot
{
  echo "== backend tree =="
  (cd "$ROOT_DIR/backend" && find . -maxdepth 3 -type d -name venv -prune -o -print) 2>/dev/null || true
} > "$OUT_DIR/backend_layout.txt"

# Python env + imports
{
  if [ -d "$ROOT_DIR/backend/venv" ]; then
    source "$ROOT_DIR/backend/venv/bin/activate"
  fi
  # Ensure project root is on PYTHONPATH so `import backend.main` resolves
  export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
  python3 -V 2>&1 || true
  echo "\n== pip freeze (top 100) =="
  pip freeze 2>/dev/null | head -n 100 || true
  echo "\n== import check =="
  python3 - <<'PY'
try:
    import fastapi, uvicorn, socketio, trafilatura
    print('core imports: OK')
except Exception as e:
    print('core imports: FAIL', e)
try:
    import backend.main as m
    print('backend.main import: OK')
except Exception as e:
    print('backend.main import: FAIL', e)
PY
} > "$OUT_DIR/backend_import_check.txt" 2>&1

# Frontend env snapshot (prefer reading frontend/.env to avoid empty shell vars)
{
  VITE=""
  REACT_APP=""
  if [ -f "$ROOT_DIR/frontend/.env" ]; then
    VITE=$(grep -E '^VITE_API_URL=' "$ROOT_DIR/frontend/.env" | tail -n1 | sed -E 's/^VITE_API_URL=//')
    REACT_APP=$(grep -E '^REACT_APP_API_URL=' "$ROOT_DIR/frontend/.env" | tail -n1 | sed -E 's/^REACT_APP_API_URL=//')
  fi
  VITE=${VITE:-${VITE_API_URL:-}}
  REACT_APP=${REACT_APP:-${REACT_APP_API_URL:-}}
  echo "VITE_API_URL=${VITE}"
  echo "REACT_APP_API_URL=${REACT_APP}"
} > "$OUT_DIR/frontend.txt" 2>&1

# Frontend .env file contents (sanitized dump for debugging)
if [ -f "$ROOT_DIR/frontend/.env" ]; then
  {
    echo "== frontend/.env =="
    sed -n '1,200p' "$ROOT_DIR/frontend/.env"
  } > "$OUT_DIR/frontend_env.txt" 2>&1 || true
else
  echo "(no frontend/.env found)" > "$OUT_DIR/frontend_env.txt" 2>&1 || true
fi

# PostgreSQL snapshot (best-effort)
{
  systemctl is-active --quiet postgresql && echo "postgresql: active" || echo "postgresql: inactive"
  command -v psql >/dev/null 2>&1 && sudo -u postgres psql -XtAc "\l" | head -n 50 || true
} > "$OUT_DIR/postgres.txt" 2>&1

# Ollama snapshot (best-effort)
{
  command -v ollama >/dev/null 2>&1 && ollama list || echo "ollama not installed"
} > "$OUT_DIR/ollama.txt" 2>&1

# Database-related files quick view
{
  grep -R "DATABASE_URL" -n "$ROOT_DIR/backend" || true
} > "$OUT_DIR/db_files.txt" 2>&1

# Backend environment snapshot (sanitized). Do not dump secrets; only known non-secret keys.
{
  env | egrep '^(HOST|PORT|SELO_AI_PORT|API_URL|ENABLE_REFLECTION_SYSTEM|ENABLE_ENHANCED_SCHEDULER|CONVERSATIONAL_MODEL|REFLECTION_LLM|EMBEDDING_MODEL)=' | sort || true
} > "$OUT_DIR/backend_env.txt" 2>&1

# System service environment snapshot (sanitized)
if [ -f "/etc/selo-ai/environment" ]; then
  {
    echo "== /etc/selo-ai/environment (sanitized) =="
    sed -E 's/(PASSWORD|SECRET|API_KEY|TOKEN)=.*/\1=REDACTED/g' \
      "/etc/selo-ai/environment" | sed -n '1,200p'
  } > "$OUT_DIR/system_env.txt" 2>&1 || true
fi

note "Diagnostics collection complete: $OUT_DIR"
