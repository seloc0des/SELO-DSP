#!/bin/bash
# Configure Ollama systemd service for optimal SELO-DSP performance

set -e

echo "Configuring Ollama systemd service for SELO-DSP..."
echo "=================================================="

# Create systemd override directory
sudo mkdir -p /etc/systemd/system/ollama.service.d/

# Create override configuration
sudo tee /etc/systemd/system/ollama.service.d/selo-optimization.conf > /dev/null <<EOF
[Service]
# CRITICAL: Enable concurrent model execution
# Allows reflection and chat models to run simultaneously
Environment="OLLAMA_NUM_PARALLEL=2"

# Keep models loaded for 30 minutes to avoid reload overhead
Environment="OLLAMA_KEEP_ALIVE=30m"

# Context length for better quality
Environment="OLLAMA_CONTEXT_LENGTH=8192"

# GPU configuration
Environment="OLLAMA_NUM_GPU=1"

# Thread optimization (adjust based on CPU cores)
Environment="OLLAMA_NUM_THREAD=8"

# Flash attention for performance
Environment="OLLAMA_FLASH_ATTENTION=true"

# Max queue and loaded models
Environment="OLLAMA_MAX_QUEUE=512"
Environment="OLLAMA_MAX_LOADED_MODELS=0"
EOF

echo "✓ Created systemd override configuration"

# Reload systemd daemon
sudo systemctl daemon-reload
echo "✓ Reloaded systemd daemon"

# Restart Ollama service
sudo systemctl restart ollama
echo "✓ Restarted Ollama service"

# Wait for Ollama to start
sleep 2

# Verify Ollama is running
if systemctl is-active --quiet ollama; then
    echo "✓ Ollama service is running"
else
    echo "✗ Ollama service failed to start"
    sudo systemctl status ollama
    exit 1
fi

# Show current configuration
echo ""
echo "Current Ollama Configuration:"
echo "=================================================="
curl -s http://127.0.0.1:11434/api/ps | jq '.' || echo "Ollama API not responding yet"

echo ""
echo "✅ Ollama configuration complete!"
echo ""
echo "Next steps:"
echo "1. Restart SELO backend to apply changes"
echo "2. Monitor logs for improved performance"
echo "3. Reflection should now complete in 3-10 seconds instead of 5+ minutes"
