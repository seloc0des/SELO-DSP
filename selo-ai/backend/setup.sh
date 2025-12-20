#!/bin/bash

# Update package list and install required packages
sudo apt update
sudo apt install -y python3-pip python3-venv

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Seed environment file with latest defaults if not present
if [ ! -f .env ]; then
  if [ -f .env.example ]; then
    cp .env.example .env
    echo "Created .env from .env.example with current default settings."
  else
    echo "Warning: .env.example not found; skipping .env creation."
  fi
fi

# Create a systemd service file
cat <<EOL | sudo tee /etc/systemd/system/selo-ai.service
[Unit]
Description=SELO DSP Backend Service
After=network.target

[Service]
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=$(pwd)/venv/bin"
ExecStart=$(pwd)/venv/bin/uvicorn backend.main:app --host 0.0.0.0 --port 8000
Restart=always

[Install]
WantedBy=multi-user.target
EOL

# Reload systemd and enable the service
sudo systemctl daemon-reload
sudo systemctl enable selo-ai
sudo systemctl start selo-ai

echo "SELO DSP backend has been installed and started as a system service."
echo "You can check the status with: sudo systemctl status selo-ai"
echo "View logs with: journalctl -u selo-ai -f"
