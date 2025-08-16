#!/bin/bash

# Enhanced diagnostic script for service startup issues
echo "🔍 English Corner AI Service Startup Diagnostics"
echo "==============================================="
echo ""

USER="ef"
BACKEND_DIR="/home/ef/english-corner-ai-backend"
VENV_DIR="${BACKEND_DIR}/venv"
SERVICE_NAME="english-corner-ai"

echo "📊 Service Status Check..."
echo ""

# Check service status
if systemctl is-active --quiet $SERVICE_NAME; then
    echo "✅ Service is currently running"
    systemctl status $SERVICE_NAME --no-pager
else
    echo "❌ Service is not running"
    echo ""
    echo "🔍 Last 20 log lines:"
    sudo journalctl -u $SERVICE_NAME -n 20 --no-pager
fi

echo ""
echo "🌐 Port Check..."

# Check if port 8443 is in use
if netstat -tulnp | grep -q :8443; then
    echo "⚠️  Port 8443 is already in use:"
    netstat -tulnp | grep :8443
    echo ""
    echo "💡 Port 8443 conflicts - this might be the issue!"
    echo "   Options:"
    echo "   1. Kill the process using port 8443"
    echo "   2. Change the service to use a different port"
    echo "   3. Check if nginx/reverse proxy is running on 8443"
else
    echo "✅ Port 8443 is available"
fi

echo ""
echo "🐍 Python Environment Check..."

# Test if we can run the app manually
echo "Testing manual execution..."
cd "$BACKEND_DIR"

if [ -f "$VENV_DIR/bin/python" ]; then
    echo "✅ Virtual environment found"
    
    # Test basic imports
    echo "🧪 Testing Python imports..."
    if sudo -u $USER "$VENV_DIR/bin/python" -c "
import sys
print(f'Python version: {sys.version}')
try:
    import fastapi
    print('✅ FastAPI imported successfully')
except ImportError as e:
    print(f'❌ FastAPI import failed: {e}')

try:
    import uvicorn
    print('✅ Uvicorn imported successfully')
except ImportError as e:
    print(f'❌ Uvicorn import failed: {e}')

try:
    import google.genai
    print('✅ Google GenAI imported successfully')
except ImportError as e:
    print(f'❌ Google GenAI import failed: {e}')

try:
    import rag_backend
    print('✅ rag_backend imported successfully')
except Exception as e:
    print(f'❌ rag_backend import failed: {e}')
" 2>&1; then
        echo "✅ Basic imports successful"
    else
        echo "❌ Import test failed"
    fi
    
    echo ""
    echo "🧪 Testing FastAPI app creation..."
    if sudo -u $USER "$VENV_DIR/bin/python" -c "
try:
    from rag_backend import app
    print('✅ FastAPI app created successfully')
    print(f'App type: {type(app)}')
except Exception as e:
    print(f'❌ App creation failed: {e}')
    import traceback
    traceback.print_exc()
" 2>&1; then
        echo "✅ App creation successful"
    else
        echo "❌ App creation failed"
    fi
    
else
    echo "❌ Virtual environment not found at $VENV_DIR"
fi

echo ""
echo "📝 Environment Variables Check..."

# Check if .env file exists and has required variables
if [ -f "$BACKEND_DIR/.env" ]; then
    echo "✅ .env file found"
    if grep -q "GOOGLE_API_KEY" "$BACKEND_DIR/.env"; then
        echo "✅ GOOGLE_API_KEY found in .env"
    else
        echo "❌ GOOGLE_API_KEY missing from .env"
    fi
else
    echo "❌ .env file not found"
    echo "   Create it with: echo 'GOOGLE_API_KEY=your_key_here' > .env"
fi

echo ""
echo "🔧 Manual Test Command..."
echo "Try running this manually to see the exact error:"
echo ""
echo "cd $BACKEND_DIR"
echo "sudo -u $USER $VENV_DIR/bin/uvicorn rag_backend:app --host 0.0.0.0 --port 8443 --log-level debug"
echo ""

echo "🚀 Suggested Fixes:"
echo ""

# Check common issues
if netstat -tulnp | grep -q :8443; then
    echo "1. 🔌 Port 8443 is in use - change port in service file:"
    echo "   sudo nano /etc/systemd/system/english-corner-ai.service"
    echo "   Change --port 8443 to --port 8444 (or another free port)"
    echo ""
fi

if [ ! -f "$BACKEND_DIR/.env" ]; then
    echo "2. 📝 Create .env file with your API keys"
    echo ""
fi

echo "3. 🔄 Restart service after fixes:"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl restart english-corner-ai"
echo ""

echo "4. 📋 View live logs:"
echo "   sudo journalctl -u english-corner-ai -f"
