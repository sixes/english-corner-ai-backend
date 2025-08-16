#!/bin/bash

# English Corner AI Backend Service Setup Script
# This script sets up the backend to run as a systemd service

set -e  # Exit on any error

echo "🚀 Setting up English Corner AI Backend as a service..."

# Configuration variables
SERVICE_NAME="english-corner-ai"
SERVICE_FILE="${SERVICE_NAME}.service"
USER="ef"
BACKEND_DIR="/home/ef/english-corner-ai-backend"
VENV_DIR="${BACKEND_DIR}/venv"

# Check if running as root for systemd operations
if [ "$EUID" -ne 0 ]; then
    echo "❌ This script needs to be run with sudo for systemd operations"
    echo "Usage: sudo ./setup_service.sh"
    exit 1
fi

# Check if service file exists
if [ ! -f "$SERVICE_FILE" ]; then
    echo "❌ Service file $SERVICE_FILE not found in current directory"
    exit 1
fi

echo "📁 Checking directories..."

# Create backend directory if it doesn't exist
if [ ! -d "$BACKEND_DIR" ]; then
    echo "📁 Creating backend directory: $BACKEND_DIR"
    mkdir -p "$BACKEND_DIR"
    chown "$USER:$USER" "$BACKEND_DIR"
fi

# Copy service files to backend directory
echo "📋 Copying service files..."
cp "$SERVICE_FILE" "$BACKEND_DIR/"
cp "rag_backend.py" "$BACKEND_DIR/" 2>/dev/null || echo "⚠️  rag_backend.py not found, make sure to copy it manually"
cp ".env" "$BACKEND_DIR/" 2>/dev/null || echo "⚠️  .env not found, make sure to copy it manually"
cp "requirements.txt" "$BACKEND_DIR/" 2>/dev/null || echo "⚠️  requirements.txt not found, make sure to copy it manually"

# Set proper ownership
chown -R "$USER:$USER" "$BACKEND_DIR"

echo "🐍 Setting up Python virtual environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "📦 Creating virtual environment..."
    sudo -u "$USER" python3 -m venv "$VENV_DIR"
fi

# Install/upgrade requirements
if [ -f "$BACKEND_DIR/requirements.txt" ]; then
    echo "📦 Installing Python dependencies..."
    sudo -u "$USER" "$VENV_DIR/bin/pip" install --upgrade pip
    sudo -u "$USER" "$VENV_DIR/bin/pip" install -r "$BACKEND_DIR/requirements.txt"
else
    echo "⚠️  No requirements.txt found, installing basic dependencies..."
    sudo -u "$USER" "$VENV_DIR/bin/pip" install fastapi uvicorn python-dotenv
fi

echo "🔧 Installing systemd service..."

# Copy service file to systemd directory
cp "$BACKEND_DIR/$SERVICE_FILE" "/etc/systemd/system/"

# Reload systemd daemon
systemctl daemon-reload

# Enable service to start on boot
systemctl enable "$SERVICE_NAME"

echo "🔥 Starting service..."

# Start the service
systemctl start "$SERVICE_NAME"

# Check service status
sleep 2
if systemctl is-active --quiet "$SERVICE_NAME"; then
    echo "✅ Service started successfully!"
    echo ""
    echo "📊 Service Status:"
    systemctl status "$SERVICE_NAME" --no-pager -l
    echo ""
    echo "🔍 Useful commands:"
    echo "  Check status:    sudo systemctl status $SERVICE_NAME"
    echo "  View logs:       sudo journalctl -u $SERVICE_NAME -f"
    echo "  Restart:         sudo systemctl restart $SERVICE_NAME"
    echo "  Stop:            sudo systemctl stop $SERVICE_NAME"
    echo "  Disable:         sudo systemctl disable $SERVICE_NAME"
    echo ""
    echo "🌐 Service should be running at: https://api.englishcorner.cyou:8443"
else
    echo "❌ Service failed to start!"
    echo "Check logs with: sudo journalctl -u $SERVICE_NAME -f"
    exit 1
fi

echo "🎉 Setup complete! The English Corner AI Backend is now running as a service."
