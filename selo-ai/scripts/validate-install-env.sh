#!/bin/bash
# SELO DSP post-install environment validation
# Checks parity between backend/.env and /etc/selo-ai/environment and performs basic health probes.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/.."
BE_ENV="$SCRIPT_DIR/backend/.env"
SVC_ENV="/etc/selo-ai/environment"

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

pass_count=0
warn_count=0
fail_count=0

note() { echo -e "${YELLOW}•${NC} $*"; }
ok() { echo -e "${GREEN}✓${NC} $*"; pass_count=$((pass_count+1)); }
warn() { echo -e "${YELLOW}!${NC} $*"; warn_count=$((warn_count+1)); }
fail() { echo -e "${RED}✗${NC} $*"; fail_count=$((fail_count+1)); }

req_file() {
  local f="$1"; if [ ! -f "$f" ]; then fail "Missing required file: $f"; return 1; else ok "Found: $f"; fi
}

get_kv() { # file key
  local file="$1"; local key="$2"; awk -F'=' -v k="^"$key"=" '$0 ~ k {print substr($0, index($0,$2))}' "$file" | tail -n1
}

# 1) Files present
req_file "$BE_ENV" || true
if [ ! -r "$SVC_ENV" ]; then fail "Service env not readable: $SVC_ENV"; else ok "Service env readable: $SVC_ENV"; fi

# 2) Read values
BE_HOST=$(get_kv "$BE_ENV" HOST || true)
BE_PORT=$(get_kv "$BE_ENV" PORT || true)
BE_FRONT=$(get_kv "$BE_ENV" FRONTEND_URL || true)
BE_CORS=$(get_kv "$BE_ENV" CORS_ORIGINS || true)
BE_SOCKET=$(get_kv "$BE_ENV" SOCKET_IO_ENABLED || true)
BE_REFLECT=$(get_kv "$BE_ENV" ENABLE_REFLECTION_SYSTEM || true)
BE_SCHED=$(get_kv "$BE_ENV" ENABLE_ENHANCED_SCHEDULER || true)

SV_HOST=$(awk -F'=' '/^(HOST|HOST_IP)=/{print $2; exit}' "$SVC_ENV" 2>/dev/null || true)
SV_PORT=$(awk -F'=' '/^(SELO_AI_PORT|PORT)=/{print $2; exit}' "$SVC_ENV" 2>/dev/null || true)
SV_API=$(awk -F'=' '/^API_URL=/{print $2; exit}' "$SVC_ENV" 2>/dev/null || true)
SV_FRONT=$(awk -F'=' '/^FRONTEND_URL=/{print $2; exit}' "$SVC_ENV" 2>/dev/null || true)
SV_CORS=$(awk -F'=' '/^CORS_ORIGINS=/{print $2; exit}' "$SVC_ENV" 2>/dev/null || true)
SV_SOCKET=$(awk -F'=' '/^SOCKET_IO_ENABLED=/{print $2; exit}' "$SVC_ENV" 2>/dev/null || true)
SV_REFLECT=$(awk -F'=' '/^ENABLE_REFLECTION_SYSTEM=/{print $2; exit}' "$SVC_ENV" 2>/dev/null || true)
SV_SCHED=$(awk -F'=' '/^ENABLE_ENHANCED_SCHEDULER=/{print $2; exit}' "$SVC_ENV" 2>/dev/null || true)

# 3) Parity checks
[ -n "${BE_HOST:-}" ] && ok "backend HOST=$BE_HOST" || warn "backend HOST missing"
[ -n "${BE_PORT:-}" ] && ok "backend PORT=$BE_PORT" || warn "backend PORT missing"
[ -n "${BE_FRONT:-}" ] && ok "backend FRONTEND_URL=$BE_FRONT" || warn "backend FRONTEND_URL missing"

[ -n "${SV_PORT:-}" ] && ok "service SELO_AI_PORT=$SV_PORT" || warn "service SELO_AI_PORT missing"
[ -n "${SV_API:-}" ] && ok "service API_URL=$SV_API" || warn "service API_URL missing"
[ -n "${SV_FRONT:-}" ] && ok "service FRONTEND_URL=$SV_FRONT" || warn "service FRONTEND_URL missing"

# HOST/HOST_IP parity is advisory (backend often binds 0.0.0.0). Only ensure API_URL uses service host/port.
if [ -n "${SV_API:-}" ] && [ -n "${SV_PORT:-}" ]; then
  API_HOSTPORT=$(echo "$SV_API" | sed -E 's#^https?://([^/]+).*#\1#')
  API_PORT=${API_HOSTPORT##*:}
  if [ "$API_PORT" = "$SV_PORT" ]; then ok "API_URL port matches SELO_AI_PORT ($SV_PORT)"; else fail "API_URL port ($API_PORT) != SELO_AI_PORT ($SV_PORT)"; fi
fi

# Frontend URL should be consistent across files
if [ -n "${BE_FRONT:-}" ] && [ -n "${SV_FRONT:-}" ]; then
  if [ "$BE_FRONT" = "$SV_FRONT" ]; then ok "FRONTEND_URL consistent"; else fail "FRONTEND_URL mismatch: backend=$BE_FRONT vs service=$SV_FRONT"; fi
fi

# Feature flags mirrored
mirror_check() { # be_val sv_val name
  local be="$1" sv="$2" name="$3"
  if [ -n "$be" ] && [ -n "$sv" ]; then
    if [ "$be" = "$sv" ]; then ok "$name consistent ($be)"; else fail "$name mismatch: backend=$be vs service=$sv"; fi
  else
    warn "$name missing in one location (backend=$be service=$sv)"
  fi
}
mirror_check "$BE_SOCKET" "$SV_SOCKET" "SOCKET_IO_ENABLED"
mirror_check "$BE_REFLECT" "$SV_REFLECT" "ENABLE_REFLECTION_SYSTEM"
mirror_check "$BE_SCHED" "$SV_SCHED" "ENABLE_ENHANCED_SCHEDULER"

# CORS should include backend frontend URL and localhost:3000
if [ -n "${SV_CORS:-}" ]; then
  case ",$SV_CORS," in
    *",$SV_FRONT,"*) ok "CORS includes service FRONTEND_URL";;
    *) warn "CORS does not include service FRONTEND_URL ($SV_FRONT)";;
  esac
  case ",$SV_CORS," in
    *",http://localhost:3000,"*) ok "CORS includes localhost:3000";;
    *) warn "CORS missing localhost:3000";;
  esac
else
  warn "CORS_ORIGINS missing in service env"
fi

# 4) Health probe (backend)
BASE_URL="${SV_API:-http://localhost:${SV_PORT:-8000}}"
SIMPLE_URL="$BASE_URL/health/simple"
HEALTH_URL="$BASE_URL/health"
DETAILS_URL="$BASE_URL/health/details?probe_llm=false&probe_db=true"
if command -v curl >/dev/null 2>&1; then
  if curl -sf "$SIMPLE_URL" >/dev/null; then
    ok "Backend liveness OK: $SIMPLE_URL"
  elif curl -sf "$HEALTH_URL" >/dev/null; then
    ok "Backend health OK: $HEALTH_URL"
  elif curl -sf "$DETAILS_URL" >/dev/null; then
    ok "Backend detailed health OK: $DETAILS_URL"
  else
    fail "Backend health FAILED: tried $SIMPLE_URL, $HEALTH_URL, $DETAILS_URL"
  fi
else
  warn "curl not available; skipping backend health probe"
fi

# 5) Summary
echo ""
echo "Validation summary: $pass_count passed, $warn_count warnings, $fail_count failed"
if [ "$fail_count" -gt 0 ]; then exit 1; fi
