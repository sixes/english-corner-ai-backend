#!/bin/bash

# Diagnostic script to check service setup
echo "🔍 English Corner AI Service Diagnostics"
echo "========================================"
echo ""

USER="ef"
BACKEND_DIR="/home/ef/english-corner-ai-backend"
VENV_DIR="${BACKEND_DIR}/venv"

echo "📁 Checking directories and permissions..."
echo ""

# Check if main directory exists and permissions
if [ -d "$BACKEND_DIR" ]; then
    echo "✅ Backend directory exists: $BACKEND_DIR"
    ls -la "$BACKEND_DIR" | head -5
else
    echo "❌ Backend directory missing: $BACKEND_DIR"
fi

echo ""

# Check if virtual environment exists
if [ -d "$VENV_DIR" ]; then
    echo "✅ Virtual environment exists: $VENV_DIR"
    if [ -f "$VENV_DIR/bin/uvicorn" ]; then
        echo "✅ uvicorn executable found"
    else
        echo "❌ uvicorn executable missing"
    fi
else
    echo "❌ Virtual environment missing: $VENV_DIR"
fi

echo ""

# Check if main app file exists
if [ -f "$BACKEND_DIR/rag_backend.py" ]; then
    echo "✅ Main application file exists: rag_backend.py"
else
    echo "❌ Main application file missing: rag_backend.py"
fi

echo ""

# Check if .env file exists
if [ -f "$BACKEND_DIR/.env" ]; then
    echo "✅ Environment file exists: .env"
else
    echo "❌ Environment file missing: .env"
    echo "   Create it with: cp .env.example .env (if you have a template)"
fi

echo ""

# Check Python version in venv
if [ -f "$VENV_DIR/bin/python" ]; then
    echo "🐍 Python version in virtual environment:"
    "$VENV_DIR/bin/python" --version
else
    echo "❌ Python executable not found in virtual environment"
fi

echo ""

# Check installed packages
if [ -f "$VENV_DIR/bin/pip" ]; then
    echo "📦 Key packages installed:"
    "$VENV_DIR/bin/pip" list | grep -E "(fastapi|uvicorn|google-genai)" || echo "   No key packages found"
else
    echo "❌ pip not found in virtual environment"
fi

echo ""

# Test manual execution
echo "🧪 Testing manual execution..."
if [ -f "$VENV_DIR/bin/uvicorn" ] && [ -f "$BACKEND_DIR/rag_backend.py" ]; then
    echo "   Command that will be executed:"
    echo "   cd $BACKEND_DIR && $VENV_DIR/bin/uvicorn rag_backend:app --host 0.0.0.0 --port 8443 --workers 1"
    echo ""
    echo "   Testing Python import..."
    cd "$BACKEND_DIR"
    if "$VENV_DIR/bin/python" -c "import rag_backend; print('✅ rag_backend imports successfully')" 2>/dev/null; then
        echo "✅ Application imports successfully"
    else
        echo "❌ Application import failed:"
        "$VENV_DIR/bin/python" -c "import rag_backend" 2>&1 | head -3
    fi
else
    echo "❌ Cannot test execution - missing files"
fi

echo ""

# Check systemd service
echo "🔧 Systemd service status:"
if systemctl list-unit-files | grep -q english-corner-ai; then
    echo "✅ Service is installed"
    systemctl is-enabled english-corner-ai
    systemctl is-active english-corner-ai
else
    echo "❌ Service not found in systemd"
fi

echo ""
echo "🎯 Recommended next steps:"
echo ""

if [ ! -d "$VENV_DIR" ]; then
    echo "1. Create virtual environment:"
    echo "   python3 -m venv $VENV_DIR"
    echo ""
fi

if [ ! -f "$VENV_DIR/bin/uvicorn" ]; then
    echo "2. Install dependencies:"
    echo "   $VENV_DIR/bin/pip install -r requirements.txt"
    echo ""
fi

if [ ! -f "$BACKEND_DIR/.env" ]; then
    echo "3. Create .env file with your API keys"
    echo ""
fi

echo "4. Update and restart the service:"
echo "   sudo cp english-corner-ai.service /etc/systemd/system/"
echo "   sudo systemctl daemon-reload"
echo "   sudo systemctl restart english-corner-ai"
echo ""

echo "5. Check service logs:"
echo "   sudo journalctl -u english-corner-ai -f"
