<#
.SYNOPSIS
    SELO AI Complete Installation Script for Windows

.DESCRIPTION
    This script handles the complete installation of SELO AI on Windows:
    - Checks and installs prerequisites (Python, Node.js, Git)
    - Installs Ollama and downloads required models
    - Sets up Python virtual environment and dependencies
    - Builds the frontend
    - Initializes the database
    - Bootstraps the AI persona
    - Configures startup options

.NOTES
    Run PowerShell as Administrator for full functionality.
    Minimum requirements: Windows 10/11, 16GB RAM, NVIDIA GPU (8GB+ VRAM recommended)

.EXAMPLE
    .\windows-install.ps1
    .\windows-install.ps1 -SkipModelDownload
    .\windows-install.ps1 -AutoConfirm
#>

[CmdletBinding()]
param(
    [switch]$SkipModelDownload,
    [switch]$SkipPersonaBootstrap,
    [switch]$SkipFirewall,
    [switch]$AutoConfirm,
    [switch]$Verbose
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"  # Speed up web requests

# Script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Configuration
$Config = @{
    PythonMinVersion = "3.10"
    NodeMinVersion = 18
    RequiredDiskGB = 20
    RequiredRAMGB = 16
    OllamaInstallerUrl = "https://ollama.com/download/OllamaSetup.exe"
    Models = @{
        Conversational = "llama3:8b"
        Analytical = "qwen2.5:3b"
        Reflection = "qwen2.5:3b"
        TraitsBootstrap = "qwen2.5:1.5b"
        Embedding = "nomic-embed-text"
    }
}

# Color output helpers
function Write-Step {
    param([string]$Message)
    Write-Host "`n=========================================" -ForegroundColor Cyan
    Write-Host "    $Message" -ForegroundColor Cyan
    Write-Host "=========================================" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "✅ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "⚠️  $Message" -ForegroundColor Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "❌ $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "ℹ️  $Message" -ForegroundColor White
}

# Check if running as Administrator
function Test-Administrator {
    $currentUser = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($currentUser)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

# Check system requirements
function Test-SystemRequirements {
    Write-Step "Checking System Requirements"
    
    $errors = @()
    $warnings = @()
    
    # Check Windows version
    $os = Get-WmiObject -Class Win32_OperatingSystem
    $osVersion = [Version]$os.Version
    if ($osVersion.Major -lt 10) {
        $errors += "Windows 10 or later is required (detected: $($os.Caption))"
    } else {
        Write-Success "Windows version: $($os.Caption)"
    }
    
    # Check RAM
    $totalRAM = [math]::Round((Get-WmiObject -Class Win32_ComputerSystem).TotalPhysicalMemory / 1GB)
    if ($totalRAM -lt $Config.RequiredRAMGB) {
        $warnings += "Recommended RAM: $($Config.RequiredRAMGB)GB (detected: ${totalRAM}GB)"
    } else {
        Write-Success "RAM: ${totalRAM}GB (minimum: $($Config.RequiredRAMGB)GB)"
    }
    
    # Check disk space
    $drive = (Get-Item $ScriptDir).PSDrive.Name
    $freeSpace = [math]::Round((Get-PSDrive $drive).Free / 1GB)
    if ($freeSpace -lt $Config.RequiredDiskGB) {
        $errors += "Insufficient disk space: ${freeSpace}GB free (need $($Config.RequiredDiskGB)GB)"
    } else {
        Write-Success "Disk space: ${freeSpace}GB free (minimum: $($Config.RequiredDiskGB)GB)"
    }
    
    # Check GPU
    $tierScript = Join-Path $ScriptDir "windows-detect-gpu-tier.ps1"
    if (Test-Path $tierScript) {
        . $tierScript
        $gpuInfo = Get-GpuMemory
        if ($gpuInfo.MemoryMB -gt 0) {
            Write-Success "GPU detected: $($gpuInfo.Name) ($($gpuInfo.MemoryMB)MB VRAM)"
        } else {
            $warnings += "No GPU detected - will use CPU-only mode (slower)"
        }
    }
    
    # Report warnings
    foreach ($warn in $warnings) {
        Write-Warning $warn
    }
    
    # Report errors
    if ($errors.Count -gt 0) {
        foreach ($err in $errors) {
            Write-Error $err
        }
        if (-not $AutoConfirm) {
            $continue = Read-Host "Continue anyway? (y/N)"
            if ($continue -notmatch '^[Yy]') {
                throw "Installation cancelled due to system requirements"
            }
        }
    }
    
    return $true
}

# Check and install Python
function Install-Python {
    Write-Step "Checking Python Installation"
    
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $pythonCmd) {
        $pythonCmd = Get-Command python3 -ErrorAction SilentlyContinue
    }
    
    if ($pythonCmd) {
        $version = & $pythonCmd.Source --version 2>&1
        if ($version -match "Python (\d+)\.(\d+)") {
            $major = [int]$Matches[1]
            $minor = [int]$Matches[2]
            if ($major -ge 3 -and $minor -ge 10) {
                Write-Success "Python $major.$minor is installed"
                return $pythonCmd.Source
            }
        }
    }
    
    Write-Warning "Python 3.10+ not found"
    Write-Info "Please install Python 3.10 or later from https://python.org"
    Write-Info "Make sure to check 'Add Python to PATH' during installation"
    
    if (-not $AutoConfirm) {
        $install = Read-Host "Open Python download page? (Y/n)"
        if ($install -notmatch '^[Nn]') {
            Start-Process "https://www.python.org/downloads/"
        }
    }
    
    throw "Python 3.10+ is required. Please install it and run this script again."
}

# Check and install Node.js
function Install-NodeJS {
    Write-Step "Checking Node.js Installation"
    
    $nodeCmd = Get-Command node -ErrorAction SilentlyContinue
    
    if ($nodeCmd) {
        $version = & node --version 2>&1
        if ($version -match "v(\d+)") {
            $major = [int]$Matches[1]
            if ($major -ge $Config.NodeMinVersion) {
                Write-Success "Node.js v$major is installed"
                return
            }
        }
    }
    
    Write-Warning "Node.js $($Config.NodeMinVersion)+ not found"
    Write-Info "Please install Node.js LTS from https://nodejs.org"
    
    if (-not $AutoConfirm) {
        $install = Read-Host "Open Node.js download page? (Y/n)"
        if ($install -notmatch '^[Nn]') {
            Start-Process "https://nodejs.org/"
        }
    }
    
    throw "Node.js $($Config.NodeMinVersion)+ is required. Please install it and run this script again."
}

# Check and install Ollama
function Install-Ollama {
    Write-Step "Checking Ollama Installation"
    
    $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
    
    if ($ollamaCmd) {
        Write-Success "Ollama is installed at: $($ollamaCmd.Source)"
        return $ollamaCmd.Source
    }
    
    Write-Info "Ollama not found. Installing..."
    
    $installerPath = Join-Path $env:TEMP "OllamaSetup.exe"
    
    try {
        Write-Info "Downloading Ollama installer..."
        Invoke-WebRequest -Uri $Config.OllamaInstallerUrl -OutFile $installerPath -UseBasicParsing
        
        Write-Info "Running Ollama installer (this may take a few minutes)..."
        $process = Start-Process -FilePath $installerPath -ArgumentList "/S" -Wait -PassThru
        
        if ($process.ExitCode -eq 0) {
            # Refresh PATH
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
            
            # Wait for Ollama service to start
            Start-Sleep -Seconds 5
            
            $ollamaCmd = Get-Command ollama -ErrorAction SilentlyContinue
            if ($ollamaCmd) {
                Write-Success "Ollama installed successfully"
                return $ollamaCmd.Source
            }
        }
        
        throw "Ollama installation failed"
    }
    finally {
        if (Test-Path $installerPath) {
            Remove-Item $installerPath -Force -ErrorAction SilentlyContinue
        }
    }
}

# Wait for Ollama service to be ready
function Wait-OllamaReady {
    param([int]$TimeoutSeconds = 60)
    
    Write-Info "Waiting for Ollama service to be ready..."
    
    $startTime = Get-Date
    while ((Get-Date) - $startTime -lt [TimeSpan]::FromSeconds($TimeoutSeconds)) {
        try {
            $response = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/version" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
            if ($response.StatusCode -eq 200) {
                Write-Success "Ollama service is ready"
                return $true
            }
        }
        catch {
            Start-Sleep -Seconds 2
        }
    }
    
    Write-Warning "Ollama service did not respond within ${TimeoutSeconds}s"
    return $false
}

# Download required models
function Install-Models {
    param([string]$OllamaBin)
    
    Write-Step "Downloading AI Models"
    
    if ($SkipModelDownload) {
        Write-Info "Skipping model download (--SkipModelDownload specified)"
        return
    }
    
    # Ensure Ollama is ready
    if (-not (Wait-OllamaReady -TimeoutSeconds 60)) {
        Write-Warning "Ollama service not responding, attempting to start..."
        Start-Process -FilePath $OllamaBin -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 5
        if (-not (Wait-OllamaReady -TimeoutSeconds 30)) {
            throw "Could not start Ollama service"
        }
    }
    
    $models = @(
        @{ Name = $Config.Models.Conversational; Description = "Conversational model" },
        @{ Name = $Config.Models.Reflection; Description = "Reflection model" },
        @{ Name = $Config.Models.Analytical; Description = "Analytical model" },
        @{ Name = $Config.Models.TraitsBootstrap; Description = "Traits bootstrap model" },
        @{ Name = $Config.Models.Embedding; Description = "Embedding model" }
    )
    
    foreach ($model in $models) {
        Write-Info "Checking $($model.Description): $($model.Name)..."
        
        # Check if model exists
        $showResult = & $OllamaBin show $model.Name 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Success "$($model.Description) already available"
            continue
        }
        
        # Download model with retries
        $maxAttempts = 3
        $attempt = 1
        $success = $false
        
        while ($attempt -le $maxAttempts -and -not $success) {
            Write-Info "Downloading $($model.Name) (attempt $attempt/$maxAttempts)..."
            
            & $OllamaBin pull $model.Name
            
            if ($LASTEXITCODE -eq 0) {
                Write-Success "$($model.Description) downloaded successfully"
                $success = $true
            }
            else {
                if ($attempt -lt $maxAttempts) {
                    Write-Warning "Download failed, retrying in 5 seconds..."
                    Start-Sleep -Seconds 5
                }
                $attempt++
            }
        }
        
        if (-not $success) {
            Write-Warning "Could not download $($model.Name) - you can install it later with: ollama pull $($model.Name)"
        }
    }
}

# Setup Python virtual environment and install dependencies
function Install-PythonDependencies {
    Write-Step "Setting Up Python Environment"
    
    $backendDir = Join-Path $ScriptDir "backend"
    $venvDir = Join-Path $backendDir "venv"
    $requirementsFile = Join-Path $backendDir "requirements.txt"
    
    # Create virtual environment
    if (-not (Test-Path $venvDir)) {
        Write-Info "Creating Python virtual environment..."
        & python -m venv $venvDir
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to create Python virtual environment"
        }
    }
    
    # Activate and install dependencies
    $activateScript = Join-Path $venvDir "Scripts\Activate.ps1"
    
    Write-Info "Installing Python dependencies..."
    
    # Use cmd to run pip in venv context
    $pipPath = Join-Path $venvDir "Scripts\pip.exe"
    $pythonPath = Join-Path $venvDir "Scripts\python.exe"
    
    # Upgrade pip
    & $pythonPath -m pip install --upgrade pip | Out-Null
    
    # Install core dependencies first
    Write-Info "Installing NumPy (pinned for FAISS compatibility)..."
    & $pipPath install "numpy>=1.24.0,<2.0.0" | Out-Null
    
    # Check for GPU and install appropriate PyTorch
    $tierScript = Join-Path $ScriptDir "windows-detect-gpu-tier.ps1"
    $hasGpu = $false
    if (Test-Path $tierScript) {
        . $tierScript
        $gpuInfo = Get-GpuMemory
        $hasGpu = $gpuInfo.MemoryMB -gt 0
    }
    
    if ($hasGpu) {
        Write-Info "Installing PyTorch with CUDA support..."
        & $pipPath install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    }
    else {
        Write-Info "Installing PyTorch (CPU version)..."
        & $pipPath install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    }
    
    # Install sentence-transformers
    Write-Info "Installing sentence-transformers..."
    & $pipPath install "sentence-transformers>=2.2.2" | Out-Null
    
    # Install FAISS
    Write-Info "Installing FAISS..."
    & $pipPath install faiss-cpu | Out-Null
    
    # Install remaining dependencies
    if (Test-Path $requirementsFile) {
        Write-Info "Installing remaining dependencies from requirements.txt..."
        # Filter out packages we already installed
        $tempReq = Join-Path $env:TEMP "selo-requirements-filtered.txt"
        Get-Content $requirementsFile | Where-Object { 
            $_ -notmatch "^(faiss|torch|sentence-transformers|numpy)" -and $_ -notmatch "^\s*#" -and $_.Trim() -ne ""
        } | Set-Content $tempReq
        
        & $pipPath install -r $tempReq
        Remove-Item $tempReq -Force -ErrorAction SilentlyContinue
    }
    
    Write-Success "Python dependencies installed"
    
    return @{
        VenvDir = $venvDir
        PythonPath = $pythonPath
        PipPath = $pipPath
    }
}

# Build frontend
function Build-Frontend {
    Write-Step "Building Frontend"
    
    $frontendDir = Join-Path $ScriptDir "frontend"
    
    if (-not (Test-Path $frontendDir)) {
        Write-Warning "Frontend directory not found at $frontendDir"
        return
    }
    
    Push-Location $frontendDir
    
    try {
        # Create .npmrc for reliable installs
        if (-not (Test-Path ".npmrc")) {
            @"
legacy-peer-deps=true
fund=false
audit=false
progress=false
"@ | Set-Content ".npmrc"
        }
        
        # Install dependencies
        Write-Info "Installing frontend dependencies (this may take a few minutes)..."
        
        if (Test-Path "package-lock.json") {
            npm ci 2>&1 | Out-Null
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "npm ci failed, trying npm install..."
                Remove-Item "node_modules" -Recurse -Force -ErrorAction SilentlyContinue
                npm install --legacy-peer-deps 2>&1 | Out-Null
            }
        }
        else {
            npm install --legacy-peer-deps 2>&1 | Out-Null
        }
        
        if ($LASTEXITCODE -ne 0) {
            throw "Failed to install frontend dependencies"
        }
        
        # Set API URL for build
        $env:REACT_APP_API_URL = "http://localhost:8000"
        $env:VITE_API_URL = "http://localhost:8000"
        
        # Create frontend .env
        @"
REACT_APP_API_URL=http://localhost:8000
VITE_API_URL=http://localhost:8000
"@ | Set-Content ".env"
        
        # Build
        Write-Info "Building frontend..."
        npm run build 2>&1 | Out-Null
        
        if ($LASTEXITCODE -ne 0) {
            throw "Frontend build failed"
        }
        
        Write-Success "Frontend built successfully"
    }
    finally {
        Pop-Location
    }
}

# Initialize database
function Initialize-Database {
    param([hashtable]$PythonEnv)
    
    Write-Step "Initializing Database"
    
    $backendDir = Join-Path $ScriptDir "backend"
    
    Push-Location $backendDir
    
    try {
        # Set PYTHONPATH
        $env:PYTHONPATH = "$backendDir;$ScriptDir"
        
        Write-Info "Running database initialization..."
        
        & $PythonEnv.PythonPath -m db.init_db 2>&1
        
        if ($LASTEXITCODE -eq 0) {
            Write-Success "Database initialized successfully"
        }
        else {
            Write-Warning "Database initialization returned non-zero exit code"
        }
    }
    finally {
        Pop-Location
    }
}

# Bootstrap persona
function Initialize-Persona {
    param([hashtable]$PythonEnv)
    
    Write-Step "Bootstrapping AI Persona"
    
    if ($SkipPersonaBootstrap) {
        Write-Info "Skipping persona bootstrap (--SkipPersonaBootstrap specified)"
        return
    }
    
    Write-Info "This step generates the AI persona and mantra before service creation."
    Write-Info "This may take 5-25 minutes depending on your system tier and hardware."
    Write-Info "This is an essential step that creates your SELO's personality and core values."
    Write-Info "The SELO will have a unique name, mantra, and behavioral traits."
    Write-Info "This process establishes your SELO's fundamental character and guiding principles."
    Write-Info "Your SELO will be ready with a distinct identity when the service starts."
    Write-Info "Standard tier systems typically take 15-20 minutes. Please be patient..."
    Write-Info ""
    
    Push-Location $ScriptDir
    
    try {
        $env:PYTHONPATH = "$ScriptDir\backend;$ScriptDir"
        
        $logFile = Join-Path $ScriptDir "logs\persona_bootstrap.log"
        New-Item -ItemType Directory -Path (Split-Path $logFile) -Force -ErrorAction SilentlyContinue | Out-Null
        
        # Timeout after 30 minutes to prevent indefinite hangs (increased from 20 for standard tier systems)
        $timeoutSeconds = 1800  # 30 minutes
        $maxAttempts = 2
        $attempt = 1
        $success = $false
        
        while ($attempt -le $maxAttempts) {
            Write-Info "Persona bootstrap attempt $attempt of $maxAttempts..."
            
            # Start the bootstrap process with timeout
            $job = Start-Job -ScriptBlock {
                param($PythonPath, $ScriptDir, $LogFile)
                $env:PYTHONPATH = "$ScriptDir\backend;$ScriptDir"
                Set-Location $ScriptDir
                & $PythonPath -u -m backend.scripts.bootstrap_persona --verbose 2>&1 | Tee-Object -FilePath $LogFile
                return $LASTEXITCODE
            } -ArgumentList $PythonEnv.PythonPath, $ScriptDir, $logFile
            
            # Wait for job to complete or timeout
            $completed = Wait-Job -Job $job -Timeout $timeoutSeconds
            
            if ($completed) {
                $exitCode = Receive-Job -Job $job
                Remove-Job -Job $job -Force
                
                if ($exitCode -eq 0) {
                    Write-Success "Persona bootstrap completed successfully on attempt $attempt"
                    Write-Info "  Log: $logFile"
                    $success = $true
                    break
                }
                elseif ($attempt -lt $maxAttempts) {
                    Write-Warning "Persona bootstrap failed (exit code $exitCode), retrying..."
                    Start-Sleep -Seconds 5
                }
                else {
                    Write-Host ""
                    Write-Error "ERROR: Persona bootstrap failed with exit code $exitCode"
                    Write-Info "   This is a critical step required before service creation."
                    Write-Info "   Review the log for details: $logFile"
                    Write-Host ""
                    Write-Info "Common causes:"
                    Write-Info "  • Ollama service not running or models not available"
                    Write-Info "  • Database connection issues"
                    Write-Info "  • Missing dependencies in backend environment"
                    Write-Info "  • Import errors (check PYTHONPATH)"
                    Write-Host ""
                    Write-Info "Installation cannot continue without a valid persona."
                    throw "Persona bootstrap failed"
                }
            }
            else {
                # Timeout occurred
                Stop-Job -Job $job -PassThru | Remove-Job -Force
                
                if ($attempt -lt $maxAttempts) {
                    Write-Warning "Persona bootstrap timed out, retrying..."
                    Start-Sleep -Seconds 5
                }
                else {
                    Write-Host ""
                    Write-Error "ERROR: Persona bootstrap timed out after 30 minutes"
                    Write-Info "   This indicates a serious issue with the system."
                    Write-Info "   Review the log for details: $logFile"
                    Write-Host ""
                    Write-Info "Common causes:"
                    Write-Info "  • Ollama service not responding"
                    Write-Info "  • Models not loaded (check: ollama list)"
                    Write-Info "  • Insufficient system resources (CPU/RAM/GPU)"
                    Write-Info "  • Database connection hanging"
                    Write-Host ""
                    Write-Info "Installation cannot continue without a valid persona."
                    throw "Persona bootstrap timed out"
                }
            }
            
            $attempt++
        }
        
        if (-not $success) {
            throw "Persona bootstrap failed after $maxAttempts attempts"
        }
    }
    finally {
        Pop-Location
    }
}

# Collect environment variables
function Initialize-Environment {
    Write-Step "Configuring Environment"
    
    $envScript = Join-Path $ScriptDir "windows-collect-env.ps1"
    
    if (Test-Path $envScript) {
        Write-Info "Running environment configuration..."
        & $envScript -ProjectRoot $ScriptDir
    }
    else {
        Write-Warning "Environment collection script not found, creating minimal configuration..."
        
        $backendEnv = Join-Path $ScriptDir "backend\.env"
        
        # Generate API key
        $bytes = New-Object byte[] 32
        $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
        $rng.GetBytes($bytes)
        $apiKey = [System.BitConverter]::ToString($bytes).Replace('-', '').ToLower()
        
        $reportsDir = Join-Path (Split-Path $ScriptDir -Parent) 'Reports'
        @"
# SELO AI Backend Configuration
DATABASE_URL=sqlite+aiosqlite:///selo_ai.db
SELO_SYSTEM_API_KEY=$apiKey
OLLAMA_BASE_URL=http://localhost:11434

CONVERSATIONAL_MODEL=$($Config.Models.Conversational)
ANALYTICAL_MODEL=$($Config.Models.Analytical)
REFLECTION_LLM=$($Config.Models.Reflection)
EMBEDDING_MODEL=$($Config.Models.Embedding)

HOST=0.0.0.0
PORT=8000
CORS_ORIGINS=http://localhost:3000

SOCKET_IO_ENABLED=true

# Boot seed directives location
SELO_REPORTS_DIR=$reportsDir
"@ | Set-Content $backendEnv
        
        Write-Success "Created minimal environment configuration"
    }
}

# Configure firewall
function Configure-Firewall {
    if ($SkipFirewall) {
        Write-Info "Skipping firewall configuration (--SkipFirewall specified)"
        return
    }
    
    $firewallScript = Join-Path $ScriptDir "windows-configure-firewall.ps1"
    
    if (Test-Path $firewallScript) {
        if ($AutoConfirm -or (Test-Administrator)) {
            & $firewallScript -BackendPort 8000 -FrontendPort 3000
        }
        else {
            Write-Info "Firewall configuration requires Administrator privileges"
            Write-Info "Run windows-configure-firewall.ps1 as Administrator to configure firewall"
        }
    }
}

# Create startup script
function New-StartupScript {
    Write-Step "Creating Startup Scripts"
    
    $startScript = Join-Path $ScriptDir "start-selo.bat"
    $startPsScript = Join-Path $ScriptDir "start-selo.ps1"
    
    # Create batch file for easy double-click start
    @"
@echo off
echo Starting SELO AI...
echo.
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0start-selo.ps1"
pause
"@ | Set-Content $startScript
    
    # Create PowerShell startup script
    @"
# SELO AI Startup Script
`$ErrorActionPreference = "Continue"
`$ScriptDir = Split-Path -Parent `$MyInvocation.MyCommand.Path

Write-Host "=========================================" -ForegroundColor Cyan
Write-Host "    Starting SELO AI" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan

# Check Ollama
Write-Host "Checking Ollama service..." -ForegroundColor Yellow
`$ollamaRunning = `$false
try {
    `$response = Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/version" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
    if (`$response.StatusCode -eq 200) {
        `$ollamaRunning = `$true
        Write-Host "✅ Ollama is running" -ForegroundColor Green
    }
}
catch {
    Write-Host "Starting Ollama..." -ForegroundColor Yellow
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 5
}

# Start backend
Write-Host "Starting backend..." -ForegroundColor Yellow
`$backendDir = Join-Path `$ScriptDir "backend"
`$venvPython = Join-Path `$backendDir "venv\Scripts\python.exe"

`$env:PYTHONPATH = "`$backendDir;`$ScriptDir"
`$backendJob = Start-Job -ScriptBlock {
    param(`$python, `$dir, `$pythonPath)
    Set-Location `$dir
    `$env:PYTHONPATH = `$pythonPath
    & `$python -m uvicorn main:app --host 0.0.0.0 --port 8000
} -ArgumentList `$venvPython, `$backendDir, "`$backendDir;`$ScriptDir"

Write-Host "✅ Backend starting on http://localhost:8000" -ForegroundColor Green

# Start frontend
Write-Host "Starting frontend..." -ForegroundColor Yellow
`$frontendDir = Join-Path `$ScriptDir "frontend"

`$frontendJob = Start-Job -ScriptBlock {
    param(`$dir)
    Set-Location `$dir
    npm run dev
} -ArgumentList `$frontendDir

Write-Host "✅ Frontend starting on http://localhost:3000" -ForegroundColor Green

Write-Host ""
Write-Host "=========================================" -ForegroundColor Green
Write-Host "    SELO AI is starting!" -ForegroundColor Green
Write-Host "=========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Open your browser to: http://localhost:3000" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the servers" -ForegroundColor Yellow
Write-Host ""

# Keep running and show logs
try {
    while (`$true) {
        Receive-Job `$backendJob -ErrorAction SilentlyContinue
        Receive-Job `$frontendJob -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
}
finally {
    Write-Host "Stopping services..." -ForegroundColor Yellow
    Stop-Job `$backendJob -ErrorAction SilentlyContinue
    Stop-Job `$frontendJob -ErrorAction SilentlyContinue
    Remove-Job `$backendJob -Force -ErrorAction SilentlyContinue
    Remove-Job `$frontendJob -Force -ErrorAction SilentlyContinue
}
"@ | Set-Content $startPsScript
    
    Write-Success "Created start-selo.bat and start-selo.ps1"
    Write-Info "Double-click start-selo.bat to start SELO AI"
}

# Run verification
function Test-Installation {
    Write-Step "Verifying Installation"
    
    $verifyScript = Join-Path $ScriptDir "windows-verify-installation.ps1"
    
    if (Test-Path $verifyScript) {
        & $verifyScript -ProjectRoot $ScriptDir
    }
    else {
        Write-Info "Verification script not found, performing basic checks..."
        
        # Check backend directory
        if (Test-Path (Join-Path $ScriptDir "backend\venv")) {
            Write-Success "Backend virtual environment exists"
        }
        else {
            Write-Warning "Backend virtual environment not found"
        }
        
        # Check frontend build
        if (Test-Path (Join-Path $ScriptDir "frontend\dist")) {
            Write-Success "Frontend build exists"
        }
        elseif (Test-Path (Join-Path $ScriptDir "frontend\build")) {
            Write-Success "Frontend build exists"
        }
        else {
            Write-Warning "Frontend build not found"
        }
        
        # Check .env
        if (Test-Path (Join-Path $ScriptDir "backend\.env")) {
            Write-Success "Backend configuration exists"
        }
        else {
            Write-Warning "Backend configuration not found"
        }
    }
}

# Show completion message
function Show-Completion {
    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host "    Installation Complete!" -ForegroundColor Green
    Write-Host "=========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "SELO AI has been successfully installed!" -ForegroundColor White
    Write-Host ""
    Write-Host "To start SELO AI:" -ForegroundColor Cyan
    Write-Host "  • Double-click: start-selo.bat" -ForegroundColor White
    Write-Host "  • Or run: .\start-selo.ps1" -ForegroundColor White
    Write-Host ""
    Write-Host "Then open your browser to: http://localhost:3000" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "Configuration files:" -ForegroundColor Yellow
    Write-Host "  • Backend: $ScriptDir\backend\.env" -ForegroundColor White
    Write-Host "  • Frontend: $ScriptDir\frontend\.env" -ForegroundColor White
    Write-Host ""
    Write-Host "For support, see the documentation or open an issue on GitHub." -ForegroundColor Gray
    Write-Host ""
}

# Main installation flow
function Main {
    Write-Host ""
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host "    SELO AI Windows Installer" -ForegroundColor Cyan
    Write-Host "=========================================" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "This script will:" -ForegroundColor White
    Write-Host "  • Check and install prerequisites" -ForegroundColor White
    Write-Host "  • Download AI models (~15GB)" -ForegroundColor White
    Write-Host "  • Set up Python and Node.js environments" -ForegroundColor White
    Write-Host "  • Build the frontend" -ForegroundColor White
    Write-Host "  • Initialize the database" -ForegroundColor White
    Write-Host "  • Generate AI persona" -ForegroundColor White
    Write-Host ""
    
    if (-not $AutoConfirm) {
        $proceed = Read-Host "Proceed with installation? (Y/n)"
        if ($proceed -match '^[Nn]') {
            Write-Host "Installation cancelled." -ForegroundColor Yellow
            return
        }
    }
    
    # Create logs directory
    New-Item -ItemType Directory -Path (Join-Path $ScriptDir "logs") -Force -ErrorAction SilentlyContinue | Out-Null
    
    # Step 1: Check system requirements
    Test-SystemRequirements
    
    # Step 2: Check/install prerequisites
    $pythonPath = Install-Python
    Install-NodeJS
    $ollamaBin = Install-Ollama
    
    # Step 3: Configure environment
    Initialize-Environment
    
    # Step 4: Download models
    Install-Models -OllamaBin $ollamaBin
    
    # Step 5: Setup Python environment
    $pythonEnv = Install-PythonDependencies
    
    # Step 6: Build frontend
    Build-Frontend
    
    # Step 7: Initialize database
    Initialize-Database -PythonEnv $pythonEnv
    
    # Step 8: Bootstrap persona
    Initialize-Persona -PythonEnv $pythonEnv
    
    # Step 9: Configure firewall (optional)
    Configure-Firewall
    
    # Step 10: Create startup scripts
    New-StartupScript
    
    # Step 11: Verify installation
    Test-Installation
    
    # Done!
    Show-Completion
}

# Run main
try {
    Main
}
catch {
    Write-Host ""
    Write-Error "Installation failed: $_"
    Write-Host ""
    Write-Host "If you need help, please:" -ForegroundColor Yellow
    Write-Host "  1. Check the error message above" -ForegroundColor White
    Write-Host "  2. Review logs in the 'logs' directory" -ForegroundColor White
    Write-Host "  3. Open an issue on GitHub with the error details" -ForegroundColor White
    Write-Host ""
    exit 1
}
