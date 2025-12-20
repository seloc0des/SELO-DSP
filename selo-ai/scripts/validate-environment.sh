#!/bin/bash

# Environment Validation Script for SELO DSP
# Validates system dependencies, Python environment, and configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Validation counters
CHECKS_PASSED=0
CHECKS_FAILED=0
WARNINGS=0

# Track validation results
check_result() {
    if [ $1 -eq 0 ]; then
        log_success "$2"
        ((CHECKS_PASSED++))
    else
        log_error "$2"
        ((CHECKS_FAILED++))
    fi
}

warning_result() {
    log_warning "$1"
    ((WARNINGS++))
}

log_info "🚀 Starting SELO DSP Environment Validation..."

# 1. Check Python version and virtual environment
log_info "Checking Python environment..."

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    log_info "Python version: $PYTHON_VERSION"
    
    # Check if we're in a virtual environment
    if [[ "$VIRTUAL_ENV" != "" ]]; then
        log_success "Virtual environment active: $VIRTUAL_ENV"
        ((CHECKS_PASSED++))
    else
        warning_result "No virtual environment detected - recommended for isolation"
    fi
    
    # Check Python version compatibility (3.9+)
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 9 ]]; then
        log_success "Python version compatible (3.9+)"
        ((CHECKS_PASSED++))
    else
        log_error "Python version incompatible. Need 3.9+, found $PYTHON_VERSION"
        ((CHECKS_FAILED++))
    fi
else
    log_error "Python3 not found in PATH"
    ((CHECKS_FAILED++))
fi

# 2. Check critical Python packages
log_info "Checking Python package dependencies..."

REQUIRED_PACKAGES=(
    "fastapi"
    "uvicorn"
    "sqlalchemy"
    "asyncpg" 
    "pydantic"
    "torch"
    "sentence-transformers"
    "faiss-cpu,faiss-gpu"
    "numpy"
    "pandas"
)

for package in "${REQUIRED_PACKAGES[@]}"; do
    # Handle packages with alternatives (faiss-cpu,faiss-gpu)
    if [[ $package == *","* ]]; then
        IFS=',' read -ra ALTERNATIVES <<< "$package"
        found=false
        for alt in "${ALTERNATIVES[@]}"; do
            if python3 -c "import $alt" &> /dev/null; then
                log_success "Package '$alt' installed"
                ((CHECKS_PASSED++))
                found=true
                break
            fi
        done
        if [[ $found == false ]]; then
            log_error "None of the alternatives found: $package"
            ((CHECKS_FAILED++))
        fi
    else
        if python3 -c "import $package" &> /dev/null; then
            # Get version if available
            VERSION=$(python3 -c "import $package; print(getattr($package, '__version__', 'unknown'))" 2>/dev/null || echo "unknown")
            log_success "Package '$package' installed (version: $VERSION)"
            ((CHECKS_PASSED++))
        else
            log_error "Package '$package' not installed"
            ((CHECKS_FAILED++))
        fi
    fi
done

# 3. Check FAISS GPU vs CPU installation
log_info "Checking FAISS installation type..."

if python3 -c "import faiss" &> /dev/null; then
    # Check if GPU version is available
    GPU_AVAILABLE=$(python3 -c "
import faiss
try:
    faiss.StandardGpuResources()
    print('gpu')
except:
    print('cpu')
" 2>/dev/null)

    if [[ $GPU_AVAILABLE == "gpu" ]]; then
        log_success "FAISS GPU support available"
        ((CHECKS_PASSED++))
    else
        warning_result "FAISS CPU-only version detected (GPU acceleration unavailable)"
    fi
else
    log_error "FAISS not installed"
    ((CHECKS_FAILED++))
fi

# 4. Check Ollama installation and models
log_info "Checking Ollama installation..."

if command -v ollama &> /dev/null; then
    log_success "Ollama CLI installed"
    ((CHECKS_PASSED++))
    
    # Check if Ollama service is running
    if pgrep -x "ollama" > /dev/null; then
        log_success "Ollama service running"
        ((CHECKS_PASSED++))
        
        # Check required models
        REQUIRED_MODELS=(
            "humanish-llama3:8b-q4"
            "phi3:mini-4k-instruct" 
            "qwen2.5-coder:3b"
            "nomic-embed-text"
        )
        
        log_info "Checking Ollama models..."
        AVAILABLE_MODELS=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' || echo "")
        
        for model in "${REQUIRED_MODELS[@]}"; do
            if echo "$AVAILABLE_MODELS" | grep -q "^$model"; then
                log_success "Model '$model' available"
                ((CHECKS_PASSED++))
            else
                log_error "Model '$model' not found"
                ((CHECKS_FAILED++))
            fi
        done
    else
        warning_result "Ollama service not running - start with 'ollama serve'"
    fi
else
    log_error "Ollama CLI not installed"
    ((CHECKS_FAILED++))
fi

# 5. Check PostgreSQL
log_info "Checking PostgreSQL..."

if command -v psql &> /dev/null; then
    log_success "PostgreSQL client installed"
    ((CHECKS_PASSED++))
    
    # Check if PostgreSQL service is running
    if systemctl is-active --quiet postgresql; then
        log_success "PostgreSQL service running"
        ((CHECKS_PASSED++))
    else
        warning_result "PostgreSQL service not running"
    fi
else
    log_error "PostgreSQL client not installed"
    ((CHECKS_FAILED++))
fi

# 6. Check environment variables
log_info "Checking environment configuration..."

REQUIRED_ENV_VARS=(
    "DATABASE_URL"
    "CONVERSATIONAL_MODEL"
    "ANALYTICAL_MODEL"
    "REFLECTION_LLM"
    "HOST"
    "PORT"
)

# Check if .env file exists
if [[ -f "backend/.env" ]]; then
    log_success "Backend .env file found"
    ((CHECKS_PASSED++))
    
    # Source .env file for checking
    set -a
    source backend/.env
    set +a
    
    for var in "${REQUIRED_ENV_VARS[@]}"; do
        if [[ -n "${!var}" ]]; then
            log_success "Environment variable '$var' set"
            ((CHECKS_PASSED++))
        else
            log_error "Environment variable '$var' not set"
            ((CHECKS_FAILED++))
        fi
    done
else
    log_error "Backend .env file not found"
    ((CHECKS_FAILED++))
fi

# 7. Check systemd service configuration
log_info "Checking systemd service..."

SERVICE_FILE="/etc/systemd/system/seloai-backend.service"
if [[ -f "$SERVICE_FILE" ]]; then
    log_success "Systemd service file exists"
    ((CHECKS_PASSED++))
    
    # Check if service uses virtual environment
    if grep -q "Environment=PATH=" "$SERVICE_FILE"; then
        log_success "Service configured with virtual environment"
        ((CHECKS_PASSED++))
    else
        warning_result "Service may not be using virtual environment"
    fi
    
    # Check service status
    if systemctl is-enabled --quiet seloai-backend; then
        log_success "Service enabled for startup"
        ((CHECKS_PASSED++))
    else
        warning_result "Service not enabled for automatic startup"
    fi
else
    warning_result "Systemd service file not found"
fi

# 8. Check file permissions and ownership
log_info "Checking file permissions..."

if [[ -r "backend/main.py" ]]; then
    log_success "Backend files readable"
    ((CHECKS_PASSED++))
else
    log_error "Backend files not accessible"
    ((CHECKS_FAILED++))
fi

# 9. Check network ports
log_info "Checking network configuration..."

BACKEND_PORT=${PORT:-8000}
FRONTEND_PORT=3000

# Check if ports are available or in use by our services
if netstat -tuln 2>/dev/null | grep -q ":$BACKEND_PORT "; then
    warning_result "Port $BACKEND_PORT already in use"
else
    log_success "Backend port $BACKEND_PORT available"
    ((CHECKS_PASSED++))
fi

if netstat -tuln 2>/dev/null | grep -q ":$FRONTEND_PORT "; then
    warning_result "Port $FRONTEND_PORT already in use"
else
    log_success "Frontend port $FRONTEND_PORT available"
    ((CHECKS_PASSED++))
fi

# 10. Memory and disk space checks
log_info "Checking system resources..."

# Check available memory (need at least 4GB for models)
AVAILABLE_MEM=$(free -g | awk 'NR==2{print $7}')
if [[ $AVAILABLE_MEM -ge 4 ]]; then
    log_success "Sufficient memory available (${AVAILABLE_MEM}GB)"
    ((CHECKS_PASSED++))
else
    warning_result "Low memory available (${AVAILABLE_MEM}GB) - may affect model performance"
fi

# Check disk space (need at least 10GB for models)
AVAILABLE_DISK=$(df -BG . | awk 'NR==2{print $4}' | sed 's/G//')
if [[ $AVAILABLE_DISK -ge 10 ]]; then
    log_success "Sufficient disk space (${AVAILABLE_DISK}GB)"
    ((CHECKS_PASSED++))
else
    warning_result "Low disk space (${AVAILABLE_DISK}GB) - may not fit all models"
fi

# Final summary
echo ""
log_info "=== VALIDATION SUMMARY ==="
log_success "Checks passed: $CHECKS_PASSED"
if [[ $CHECKS_FAILED -gt 0 ]]; then
    log_error "Checks failed: $CHECKS_FAILED"
fi
if [[ $WARNINGS -gt 0 ]]; then
    log_warning "Warnings: $WARNINGS"
fi

echo ""
if [[ $CHECKS_FAILED -eq 0 ]]; then
    log_success "✅ Environment validation PASSED - SELO DSP ready for deployment"
    exit 0
else
    log_error "❌ Environment validation FAILED - $CHECKS_FAILED critical issues found"
    echo ""
    log_info "Please fix the failed checks before deploying SELO DSP"
    exit 1
fi
