#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

# Get the script directory and navigate to frontend
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Starting SELO AI Frontend Production Server..."
echo "Current directory: $(pwd)"

# Ensure dependencies are installed (prefer clean lockfile install)
echo "Installing dependencies..."
if [ -f "package-lock.json" ]; then
  npm ci || npm install
else
  npm install
fi

# Resolve dynamic host and backend port
HOST_IP=${HOST_IP:-$(hostname -I | awk '{print $1}')}
# If backend.port file was produced by start-service, prefer it for coherence
if [ -f "../backend.port" ]; then
  BACKEND_PORT=$(cat ../backend.port)
else
  BACKEND_PORT=${SELO_AI_PORT:-${PORT:-8000}}
fi
API_BASE="${API_URL:-http://${HOST_IP}:${BACKEND_PORT}}"

# Export API URLs for CRA and Vite before build and persist to .env
export REACT_APP_API_URL="${REACT_APP_API_URL:-$API_BASE}"
export VITE_API_URL="${VITE_API_URL:-$API_BASE}"
{
  echo "REACT_APP_API_URL=$REACT_APP_API_URL"
  echo "VITE_API_URL=$VITE_API_URL"
} > .env
echo "Wrote frontend API config to $(pwd)/.env"

# Build the React app
echo "Building React app (API=${VITE_API_URL})..."
npm run build

# Check if build was successful
if [ ! -d "build" ]; then
    echo "Build failed! build directory not found."
    exit 1
fi

echo "Starting production server on all interfaces..."
echo "Frontend will be accessible at: http://${HOST_IP}:3000"
echo "API Backend should be running at: http://${HOST_IP}:${BACKEND_PORT}"
echo "Press Ctrl+C to stop the server"

# Start the production server using local binary or npx, binding to all interfaces
if command -v npx >/dev/null 2>&1; then
  npx --yes serve -s build -l tcp://0.0.0.0:3000
else
  if [ ! -x "node_modules/.bin/serve" ]; then
    echo "Installing local 'serve'..."
    npm install --no-save serve
  fi
  node node_modules/serve/bin/serve.js -s build -l tcp://0.0.0.0:3000
fi
