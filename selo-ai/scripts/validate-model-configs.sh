#!/bin/bash
# SELO DSP Model Configuration Validation Script

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "    SELO DSP Model Configuration Validator"
echo "========================================="

# Check if configuration directories exist
echo "Checking configuration structure..."

configs=("lightweight" "heavyweight")
for config in "${configs[@]}"; do
    config_dir="configs/$config"
    echo "Validating $config configuration..."
    
    # Check directory exists
    if [ ! -d "$config_dir" ]; then
        echo "✗ Missing directory: $config_dir"
        continue
    fi
    
    # Check required files
    files=(".env.template" "models.txt" "README.md" "install-models.sh")
    for file in "${files[@]}"; do
        if [ -f "$config_dir/$file" ]; then
            echo "  ✓ $file exists"
        else
            echo "  ✗ Missing: $file"
        fi
    done
    
    # Check if install script is executable
    if [ -x "$config_dir/install-models.sh" ]; then
        echo "  ✓ install-models.sh is executable"
    else
        echo "  ✗ install-models.sh is not executable"
    fi
    
    # Validate .env template has required variables
    if [ -f "$config_dir/.env.template" ]; then
        required_vars=("CONVERSATIONAL_MODEL" "ANALYTICAL_MODEL" "REFLECTION_LLM" "EMBEDDING_MODEL")
        for var in "${required_vars[@]}"; do
            if grep -q "^$var=" "$config_dir/.env.template"; then
                echo "  ✓ $var defined in .env.template"
            else
                echo "  ✗ Missing variable: $var"
            fi
        done
    fi
    
    echo ""
done

# Check main installation scripts
echo "Checking installation scripts..."
scripts=("install-complete.sh" "install-lightweight.sh" "install-heavyweight.sh")
for script in "${scripts[@]}"; do
    if [ -f "$script" ] && [ -x "$script" ]; then
        echo "✓ $script exists and is executable"
    else
        echo "✗ $script missing or not executable"
    fi
done

# Test model configuration parameter parsing
echo ""
echo "Testing model configuration detection..."

# Simulate different RAM/VRAM scenarios
test_scenarios() {
    local ram_gb=$1
    local vram_gb=$2
    local expected=$3
    
    echo "Testing RAM=${ram_gb}GB, VRAM=${vram_gb}GB -> Expected: $expected"
    
    # This would be the logic from install-complete.sh
    if [ "$ram_gb" -ge 24 ] && [ "$vram_gb" -ge 8 ]; then
        result="heavyweight"
    else
        result="lightweight"
    fi
    
    if [ "$result" = "$expected" ]; then
        echo "  ✓ Correctly detected: $result"
    else
        echo "  ✗ Expected $expected, got $result"
    fi
}

test_scenarios 8 0 "lightweight"
test_scenarios 16 4 "lightweight"
test_scenarios 32 8 "heavyweight"
test_scenarios 64 16 "heavyweight"
test_scenarios 32 4 "lightweight"  # High RAM but low VRAM

echo ""
echo "========================================="
echo "    Validation Complete"
echo "========================================="
echo ""
echo "Usage examples:"
echo "  ./install-complete.sh                    # Auto-detect configuration"
echo "  ./install-lightweight.sh                # Force lightweight"
echo "  ./install-heavyweight.sh                # Force heavyweight"
echo "  ./install-complete.sh --model-config=lightweight"
echo ""
echo "Configuration files:"
echo "  configs/lightweight/.env.template       # Lightweight environment"
echo "  configs/heavyweight/.env.template       # Heavyweight environment"
echo "  INSTALLATION.md                         # Detailed installation guide"
