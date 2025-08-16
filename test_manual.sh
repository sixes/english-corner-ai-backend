#!/bin/bash

# Manual test script to run the server and debug issues
echo "🧪 Manual Server Test for English Corner AI Backend"
echo "=================================================="
echo ""

USER="ef"
BACKEND_DIR="/home/ef/english-corner-ai-backend"
VENV_DIR="${BACKEND_DIR}/venv"

# Check if we're in the right directory
if [ "$(pwd)" != "$BACKEND_DIR" ]; then
    echo "📁 Changing to backend directory: $BACKEND_DIR"
    cd "$BACKEND_DIR" || exit 1
fi

echo "🔍 Current directory: $(pwd)"
echo "🔍 Running as user: $(whoami)"
echo ""

# Check if virtual environment exists
if [ ! -d "$VENV_DIR" ]; then
    echo "❌ Virtual environment not found at $VENV_DIR"
    echo "   Create it with: python3 -m venv venv"
    exit 1
fi

echo "✅ Virtual environment found"
echo ""

# Activate virtual environment and test
echo "🐍 Testing with virtual environment..."
source "$VENV_DIR/bin/activate"

echo "🔍 Python version: $(python --version)"
echo "🔍 Pip version: $(pip --version)"
echo ""

# Test imports
echo "🧪 Testing key imports..."
python -c "
import sys
print(f'Python executable: {sys.executable}')

try:
    import fastapi
    print('✅ FastAPI imported')
except Exception as e:
    print(f'❌ FastAPI import failed: {e}')

try:
    import uvicorn
    print('✅ Uvicorn imported')
except Exception as e:
    print(f'❌ Uvicorn import failed: {e}')

try:
    import google.genai
    print('✅ Google GenAI imported')
except Exception as e:
    print(f'❌ Google GenAI import failed: {e}')

try:
    from rag_backend import app
    print('✅ rag_backend app imported')
except Exception as e:
    print(f'❌ rag_backend import failed: {e}')
    import traceback
    traceback.print_exc()
"

echo ""
echo "🌐 Checking available ports..."
netstat -tulnp | grep -E ':8443|:8444|:8080' || echo "No services on common ports"

echo ""
echo "🚀 Attempting to start server manually..."
echo "   If this fails, you'll see the exact error message"
echo "   Press Ctrl+C to stop the server once it starts"
echo ""

# Try to start the server
echo "Command: uvicorn rag_backend:app --host 0.0.0.0 --port 8444 --log-level debug"
echo ""

uvicorn rag_backend:app --host 0.0.0.0 --port 8444 --log-level debug
