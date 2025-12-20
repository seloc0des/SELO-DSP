#!/bin/bash

set -euo pipefail
IFS=$'\n\t'

# SELO DSP Full Server Startup Script
# This script starts both the backend and frontend services

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "    SELO DSP Full Server Startup"
echo "========================================="
echo "Script directory: $SCRIPT_DIR"

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
        echo "Warning: neither 'lsof' nor 'ss' found; cannot verify port $port. Assuming free."
        return 1
    fi
}

# Function to kill process on port
kill_port() {
    local port=$1
    echo "Killing any existing process on port $port..."
    if command -v lsof >/dev/null 2>&1; then
        lsof -ti:$port | xargs kill -9 2>/dev/null || true
    elif command -v ss >/dev/null 2>&1; then
        PIDS=$(ss -ltnp "sport = :$port" 2>/dev/null | awk -F',' '/pid=/ {for(i=1;i<=NF;i++){if($i~^pid=){gsub("pid=","",$i);print $i}}}')
        if [ -n "$PIDS" ]; then
            echo "$PIDS" | xargs -r kill -9 2>/dev/null || true
        fi
    else
        echo "Warning: cannot determine PIDs to kill on port $port (no lsof/ss)."
    fi
    sleep 2
}

# Check for runtimes and LLM models
echo "Checking runtime dependencies..."
if ! command -v python3 >/dev/null 2>&1; then
    echo "Error: python3 is not installed. Install Python 3.10+ and retry."; exit 1
fi
if ! python3 -m venv --help >/dev/null 2>&1; then
    echo "Error: Python venv module is unavailable. Install python3-venv."; exit 1
fi
if ! command -v node >/dev/null 2>&1; then
    echo "Error: node is not installed. Install Node.js 18+ and retry."; exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
    echo "Error: npm is not installed. Install npm and retry."; exit 1
fi

echo "Checking Ollama model availability..."
OLLAMA_BIN="$(command -v ollama || echo /usr/local/bin/ollama)"
if [ ! -x "$OLLAMA_BIN" ]; then
    echo "Error: Ollama binary not found. Install Ollama and retry."; exit 1
fi
CONVERSATIONAL_MODEL="${CONVERSATIONAL_MODEL:-mistral:latest}"
if ! "$OLLAMA_BIN" list | grep -q "$CONVERSATIONAL_MODEL" 2>/dev/null; then
    echo "Pulling $CONVERSATIONAL_MODEL (this may take a few minutes)..."
    "$OLLAMA_BIN" pull "$CONVERSATIONAL_MODEL" || { echo "Failed to pull model $CONVERSATIONAL_MODEL"; exit 1; }
fi

# Clean up any existing processes
echo "Cleaning up existing processes..."

# Function to detect primary network IP (matches installer logic exactly)
detect_host_ip() {
    # Check environment files first
    if [ -f "/etc/selo-ai/environment" ]; then
        HOST_IP_FROM_ENV=$(grep '^HOST_IP=' /etc/selo-ai/environment 2>/dev/null | cut -d'=' -f2)
        if [ -n "$HOST_IP_FROM_ENV" ] && [ "$HOST_IP_FROM_ENV" != "0.0.0.0" ]; then
            echo "$HOST_IP_FROM_ENV"
            return
        fi
    fi
    
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
    
    # Final fallback
    echo "127.0.0.1"
}

# Determine host and ports using dynamic detection
HOST_IP=${HOST_IP:-$(detect_host_ip)}
BACKEND_PORT=${SELO_AI_PORT:-${PORT:-8000}}
kill_port "$BACKEND_PORT"  # Backend
kill_port 3000              # Frontend

# Start Backend
echo "Starting SELO DSP Backend..."
cd "$SCRIPT_DIR/backend"

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing dependencies..."
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install dependencies
echo "Installing Python dependencies..."
pip install -r requirements.txt

# Verify critical imports work
echo "Verifying dependencies..."
python -c "import fastapi, uvicorn, trafilatura, requests, socketio, pydantic, asyncio; print('✓ Core dependencies verified')"

# Start backend in background using uvicorn factory app
echo "Starting FastAPI backend on port ${BACKEND_PORT}..."
UVICORN_BIN="$(command -v uvicorn || echo "$SCRIPT_DIR/backend/venv/bin/uvicorn")"
"$UVICORN_BIN" "backend.main:get_socketio_app" --host 0.0.0.0 --port "$BACKEND_PORT" --factory &
BACKEND_PID=$!
echo "Backend started with PID: $BACKEND_PID"

# Wait a moment for backend to start
sleep 3

# Check if backend started successfully
if check_port "$BACKEND_PORT"; then
    echo "✓ Backend is running on http://${HOST_IP}:${BACKEND_PORT}"
else
    echo "✗ Backend failed to start on port ${BACKEND_PORT}"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start Frontend
echo "Starting SELO DSP Frontend..."
cd "$SCRIPT_DIR/frontend"

# Export API base URL for build (CRA and Vite), strictly derived from selected backend port
API_BASE="http://${HOST_IP}:${BACKEND_PORT}"
export REACT_APP_API_URL="$API_BASE"
export VITE_API_URL="$API_BASE"

# Install frontend dependencies (prefer clean lockfile install)
echo "Installing frontend dependencies..."
if [ -f "package-lock.json" ]; then
    npm ci || npm install
else
    npm install
fi

# Persist API URLs for clarity and local runs
{
    echo "REACT_APP_API_URL=$API_BASE"
    echo "VITE_API_URL=$API_BASE"
} > .env
echo "Wrote frontend API config to $(pwd)/.env"

# Build the React app
echo "Building React application (API=${VITE_API_URL})..."
npm run build

if [ ! -d "build" ]; then
    echo "✗ Frontend build failed!"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start frontend in background using npx or local binary with tcp:// listen
echo "Starting React frontend on port 3000..."
if command -v npx >/dev/null 2>&1; then
    npx --yes serve -s build -l tcp://0.0.0.0:3000 &
    FRONTEND_PID=$!
else
    # Ensure local 'serve' is available to avoid global npx cache issues
    if [ ! -x "node_modules/.bin/serve" ]; then
        echo "Installing local 'serve' static server..."
        npm install --no-save serve || true
    fi
    node node_modules/serve/bin/serve.js -s build -l tcp://0.0.0.0:3000 &
    FRONTEND_PID=$!
fi
echo "Frontend started with PID: $FRONTEND_PID"

# Wait a moment for frontend to start
sleep 3

# Check if frontend started successfully
if check_port 3000; then
    echo "✓ Frontend is running on http://${HOST_IP}:3000"
else
    echo "✗ Frontend failed to start on port 3000"
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    exit 1
fi

echo ""
echo "========================================="
echo "    SELO DSP Server Successfully Started!"
echo "========================================="
echo "Frontend URL: http://${HOST_IP}:3000"
echo "Backend API:  http://${HOST_IP}:${BACKEND_PORT}"
echo "Model:        Mistral (via Ollama)"
echo ""
echo "Access the chat interface from any device on your network!"
echo "Press Ctrl+C to stop both services..."
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down SELO DSP services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "Services stopped. Goodbye!"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM

# Keep script running and show status
while true; do
    sleep 10
    
    # Check if services are still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "⚠️  Backend process died unexpectedly!"
        kill $FRONTEND_PID 2>/dev/null
        exit 1
    fi
    
    if ! kill -0 $FRONTEND_PID 2>/dev/null; then
        echo "⚠️  Frontend process died unexpectedly!"
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
done
