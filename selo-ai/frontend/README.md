# SELO DSP Frontend

This is the frontend for the SELO DSP application, built with React and Tailwind CSS.

## Prerequisites

- Node.js 16+ and npm
- Backend server running (see backend setup)

## Setup

1. **Install Node.js** (if not already installed):
   ```bash
   curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
   export NVM_DIR="$HOME/.nvm"
   [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
   nvm install --lts
   nvm use --lts
   ```

2. **Run the setup script**:
   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Configure the backend URL** (if different from default):
   Use the BASE URL (no `/chat`). Both CRA and Vite variables are supported.
   ```bash
   cat > .env << 'EOF'
   REACT_APP_API_URL=http://<server-ip>:<port>
   VITE_API_URL=http://<server-ip>:<port>
   EOF
   ```
   - Default backend port is 8000; override via installer `--port` or env `SELO_AI_PORT`/`PORT`.

## Development

Start the development server:
```bash
npm start
```

The app will be available at `http://localhost:3000`

## Building for Production

Create a production build:
```bash
npm run build
```

Serve the production build:
```bash
npm install -g serve
serve -s build
```

---

## 🏭 Production Auto-Start (Recommended)

For unattended or server use, set up the frontend to auto-start at login or boot using systemd and the provided `start-production.sh` script:

1. Edit and use the script:
   ```bash
   ./start-production.sh
   ```
2. Set up the user systemd service as described in the main project README.

The service will:
- Build and serve the app on port 3000
- Restart automatically if it crashes
 - Export `REACT_APP_API_URL`/`VITE_API_URL` from `API_URL` or `HOST_IP`/`SELO_AI_PORT` dynamically

See the main [README](../../README.md) and [../OLLAMA_COMMANDS.md](../OLLAMA_COMMANDS.md) for full deployment and troubleshooting info.

---

Create a production build:
```bash
npm run build
```

Serve the production build:
```bash
npm install -g serve
serve -s build
```

## Troubleshooting

- **Port in use**: Change the port with `PORT=3001 npm start`
- **Dependency issues**: Try `rm -rf node_modules package-lock.json && npm install`
- **Blank page**: Check browser console for errors and verify backend is running
