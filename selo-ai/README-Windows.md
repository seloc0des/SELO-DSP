# SELO AI - Windows Installation Guide

This guide covers installing and running SELO AI on Windows 10/11.

## System Requirements

### Minimum Requirements
- **OS:** Windows 10 (1903+) or Windows 11
- **RAM:** 16GB (32GB recommended)
- **Storage:** 20GB free space (SSD recommended)
- **CPU:** Modern multi-core processor (Intel i5/AMD Ryzen 5 or better)

### Recommended for GPU Acceleration
- **GPU:** NVIDIA GPU with 8GB+ VRAM (RTX 3060 or better)
- **Drivers:** NVIDIA Driver 470+ with CUDA support

> **Note:** SELO AI can run without a GPU using CPU-only mode, but responses will be significantly slower.

## Prerequisites

Before running the installer, ensure you have:

1. **Python 3.10+**
   - Download from: https://python.org/downloads/
   - ⚠️ **Important:** Check "Add Python to PATH" during installation

2. **Node.js 18+**
   - Download LTS version from: https://nodejs.org/
   - The installer will add Node.js to PATH automatically

3. **Git** (optional, for cloning the repository)
   - Download from: https://git-scm.com/download/win

## Quick Start

### Option 1: Automated Installation (Recommended)

1. Open PowerShell as Administrator
2. Navigate to the `selo-ai` directory
3. Run the installer:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
.\windows-install.ps1
```

The installer will:
- Check system requirements
- Install Ollama (if not present)
- Download AI models (~15GB)
- Set up Python virtual environment
- Build the frontend
- Initialize the database
- Generate the AI persona

### Option 2: Manual Installation

If you prefer manual control, follow these steps:

1. **Install Ollama**
   - Download from: https://ollama.com/download
   - Run the installer

2. **Download Models**
   ```powershell
   ollama pull llama3:8b
   ollama pull qwen2.5:3b
   ollama pull qwen2.5:1.5b
   ollama pull nomic-embed-text
   ```

3. **Set Up Backend**
   ```powershell
   cd backend
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```

4. **Configure Environment**
   ```powershell
   .\windows-collect-env.ps1
   ```

5. **Initialize Database**
   ```powershell
   cd backend
   .\venv\Scripts\Activate.ps1
   python -m db.init_db
   ```

6. **Build Frontend**
   ```powershell
   cd frontend
   npm install
   npm run build
   ```

7. **Bootstrap Persona**
   ```powershell
   cd ..
   $env:PYTHONPATH = "$PWD\backend;$PWD"
   .\backend\venv\Scripts\python.exe -m backend.scripts.bootstrap_persona --verbose
   ```

## Starting SELO AI

After installation, start SELO AI by:

1. **Double-click** `start-selo.bat` in the selo-ai folder

   OR

2. Run in PowerShell:
   ```powershell
   .\start-selo.ps1
   ```

Then open your browser to: **http://localhost:3000**

## Configuration Files

| File | Purpose |
|------|---------|
| `backend\.env` | Backend configuration (API keys, models, database) |
| `frontend\.env` | Frontend configuration (API URL) |

### Key Configuration Options

Edit `backend\.env` to customize:

```env
# AI Models
CONVERSATIONAL_MODEL=llama3:8b
REFLECTION_LLM=qwen2.5:3b
ANALYTICAL_MODEL=qwen2.5:3b

# Server
HOST=0.0.0.0
PORT=8000

# Optional: Brave Search API for web search
BRAVE_SEARCH_API_KEY=your-api-key-here
```

## Windows Scripts Reference

| Script | Purpose |
|--------|---------|
| `windows-install.ps1` | Complete installation |
| `windows-collect-env.ps1` | Configure environment variables |
| `windows-detect-gpu-tier.ps1` | Detect GPU and set performance tier |
| `windows-configure-firewall.ps1` | Configure Windows Firewall |
| `windows-verify-installation.ps1` | Verify installation |
| `start-selo.bat` | Start SELO AI (double-click) |
| `start-selo.ps1` | Start SELO AI (PowerShell) |

## Troubleshooting

### PowerShell Execution Policy Error

If you get an error about script execution:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
```

### Ollama Not Starting

1. Check if Ollama is installed: `ollama --version`
2. Try starting manually: `ollama serve`
3. Check Windows Services for "Ollama" service

### Python Virtual Environment Issues

```powershell
# Remove and recreate venv
Remove-Item -Recurse -Force backend\venv
cd backend
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Frontend Build Fails

```powershell
cd frontend
Remove-Item -Recurse -Force node_modules
Remove-Item package-lock.json
npm install --legacy-peer-deps
npm run build
```

### Port Already in Use

If port 8000 or 3000 is in use:

```powershell
# Find process using port
netstat -ano | findstr :8000

# Kill process by PID (replace 1234 with actual PID)
taskkill /PID 1234 /F
```

### GPU Not Detected

1. Ensure NVIDIA drivers are installed
2. Check nvidia-smi works: `nvidia-smi`
3. Restart after driver installation

## Firewall Configuration

To access SELO AI from other devices on your network:

1. Run as Administrator:
   ```powershell
   .\windows-configure-firewall.ps1
   ```

2. Or manually allow ports 8000 and 3000 in Windows Firewall

## Updating SELO AI

To update to a new version:

1. Stop SELO AI (Ctrl+C in terminal)
2. Pull latest changes (if using Git): `git pull`
3. Update dependencies:
   ```powershell
   cd backend
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   
   cd ..\frontend
   npm install
   npm run build
   ```
4. Restart SELO AI

## Uninstallation

To remove SELO AI:

1. Delete the `selo-ai` folder
2. Optionally uninstall Ollama from Windows Settings > Apps
3. Optionally remove firewall rules in Windows Defender Firewall

## Support

For issues and feature requests, please open an issue on GitHub.

---

**Note:** This is a development installation. For production deployments, consider:
- Setting up proper Windows Service
- Configuring HTTPS/SSL
- Setting up PostgreSQL instead of SQLite
- Implementing proper backup procedures
