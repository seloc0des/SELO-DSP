#!/bin/bash

# Exit on error
set -e

echo "🚀 Starting SELO AI Frontend Setup..."

REQUIRED_NODE_MAJOR=18
REQUIRED_NPM_MAJOR=9

ensure_nvm() {
  if [ -z "${NVM_DIR:-}" ]; then
    export NVM_DIR="$HOME/.nvm"
  fi

  if [ ! -s "$NVM_DIR/nvm.sh" ]; then
    echo "📥 Installing nvm (Node Version Manager)..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
  fi

  # shellcheck source=/dev/null
  [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
}

ensure_node() {
  local node_major=0

  if command -v node &> /dev/null; then
    local node_version
    node_version=$(node --version)
    node_major=${node_version#v}
    node_major=${node_major%%.*}
  fi

  if [ "$node_major" -lt "$REQUIRED_NODE_MAJOR" ]; then
    echo "❌ Node.js version $node_major detected. Requiring >= $REQUIRED_NODE_MAJOR."
    ensure_nvm
    echo "📦 Installing Node.js LTS via nvm..."
    nvm install --lts
    nvm use --lts
  else
    # Still load nvm if available to ensure npm upgrades use the same toolchain
    ensure_nvm
    if command -v nvm &> /dev/null; then
      local current
      current=$(node --version)
      echo "ℹ️  Using existing Node.js $current"
    fi
  fi
}

ensure_npm() {
  local npm_major=0

  if command -v npm &> /dev/null; then
    local npm_version
    npm_version=$(npm --version)
    npm_major=${npm_version%%.*}
  fi

  if [ "$npm_major" -lt "$REQUIRED_NPM_MAJOR" ]; then
    echo "❌ npm version $npm_major detected. Requiring >= $REQUIRED_NPM_MAJOR."
    ensure_nvm
    if ! command -v nvm &> /dev/null; then
      echo "⚠️  Unable to locate nvm for npm upgrade. Please install nvm manually." >&2
      exit 1
    fi
    echo "📦 Upgrading npm to latest stable..."
    npm install -g npm@latest
  fi
}

ensure_node
ensure_npm

# Verify Node.js and npm
NODE_VERSION=$(node --version)
NPM_VERSION=$(npm --version)
echo "✅ Node.js $NODE_VERSION and npm $NPM_VERSION are installed"

# Install project dependencies
echo "📦 Installing project dependencies..."
npm install

# Set default backend URL if not already set
if [ ! -f ".env" ]; then
    echo "🔧 Creating .env file with default configuration"
    {
      echo "REACT_APP_API_URL=http://localhost:8000"
      echo "VITE_API_URL=http://localhost:8000"
    } > .env
    echo "ℹ️  Please update the .env file with your backend URL if needed"
fi

echo ""
echo "✨ Setup complete!"
echo ""
echo "To start the development server, run:"
echo "  npm start"
echo ""
echo "To build for production:"
echo "  npm run build"
echo ""

# Make the script executable
chmod +x setup.sh
