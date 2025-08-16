#!/bin/bash

# Simplified setup script for when already in the correct directory
# This script assumes you're running from /home/ef/english-corner-ai-backend

set -e  # Exit on any error

echo "🚀 Setting up English Corner AI Backend as a service (simplified)..."

# Configuration variables
SERVICE_NAME="english-corner-ai"
SERVICE_FILE="${SERVICE_NAME}.service"
USER="ef"
BACKEND_DIR="/home/ef/english-corner-ai-backend"
VENV_DIR="${BACKEND_DIR}/venv"

# Check if running as root for systemd operations
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script needs to be run with sudo for systemd operations"
    echo "Usage: sudo ./setup_service_local.sh"
    exit 1
fi

# Check if we're in the right directory
CURRENT_DIR=$(pwd)
if [ "$CURRENT_DIR" != "$BACKEND_DIR" ]; then
    echo "❌ Please run this script from $BACKEND_DIR"
    echo "Current directory: $CURRENT_DIR"
    exit 1
fi

# Check if required files exist
for file in "$SERVICE_FILE" "rag_backend.py" "requirements.txt"; do
    if [ ! -f "$file" ]; then
        echo "❌ Required file $file not found in current directory"
        exit 1
    fi
done

echo "✅ All required files found"

# Set proper ownership
echo "🔧 Setting proper ownership..."
chown -R "$USER:$USER" "$BACKEND_DIR"

echo "🐍 Setting up Python virtual environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    sudo -u "$USER" python3 -m venv "$VENV_DIR"
fi

# Install/upgrade requirements
echo "📦 Installing Python dependencies..."
sudo -u "$USER" "$VENV_DIR/bin/pip" install --upgrade pip
sudo -u "$USER" "$VENV_DIR/bin/pip" install -r "requirements.txt"

echo "🔧 Installing systemd service..."

# Copy service file to systemd directory
cp "$SERVICE_FILE" "/etc/systemd/system/"

# Reload systemd daemon
systemctl daemon-reload

# Enable service to start on boot
systemctl enable "$SERVICE_NAME"

echo "🔥 Starting service..."

# Start the service
systemctl start "$SERVICE_NAME"

# Check service status
sleep 3
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "✅ Service started successfully!"
    echo ""
    echo "📊 Service Status:"
    systemctl status "$SERVICE_NAME" --no-pager -l
    echo ""
    echo "🔍 Useful commands:"
    echo "  Check status:    ./service.sh status"
    echo "  View logs:       ./service.sh logs"
    echo "  Restart:         ./service.sh restart"
    echo "  Stop:            ./service.sh stop"
    echo ""
    echo "🌐 Service should be running at: https://api.englishcorner.cyou:8443"
else
    echo "❌ Service failed to start!"
    echo "Check logs with: sudo journalctl -u $SERVICE_NAME -f"
    exit 1
fi

echo "🎉 Setup complete! The English Corner AI Backend is now running as a service."
