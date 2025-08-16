#!/bin/bash

# CORS Test Script for English Corner AI Backend
echo "🌐 Testing CORS Configuration for English Corner AI Backend"
echo "=========================================================="
echo ""

BACKEND_URL="https://api.englishcorner.cyou:8443"
FRONTEND_ORIGIN="https://www.englishcorner.cyou"

echo "🔍 Testing backend availability..."
echo "Backend URL: $BACKEND_URL (via Nginx reverse proxy)"
echo "Frontend Origin: $FRONTEND_ORIGIN"
echo ""

# Test 1: Basic health check
echo "1. 📊 Testing basic health endpoint..."
curl -s -w "\nHTTP Status: %{http_code}\n" "$BACKEND_URL/health" || echo "❌ Health check failed"
echo ""

# Test 2: Preflight OPTIONS request (what browsers send first)
echo "2. ✈️  Testing preflight OPTIONS request..."
curl -s -w "\nHTTP Status: %{http_code}\n" \
  -X OPTIONS \
  -H "Origin: $FRONTEND_ORIGIN" \
  -H "Access-Control-Request-Method: POST" \
  -H "Access-Control-Request-Headers: Content-Type" \
  "$BACKEND_URL/chat"
echo ""

# Test 3: Actual POST request with CORS headers
echo "3. 📨 Testing actual POST request with CORS..."
curl -s -w "\nHTTP Status: %{http_code}\n" \
  -X POST \
  -H "Origin: $FRONTEND_ORIGIN" \
  -H "Content-Type: application/json" \
  -d '{"question": "Hello, testing CORS", "session_id": "test"}' \
  "$BACKEND_URL/chat"
echo ""

# Test 4: Check if service is running
echo "4. 🔧 Checking if service is running..."
if command -v systemctl &> /dev/null; then
    systemctl is-active english-corner-ai 2>/dev/null
    if [ $? -eq 0 ]; then
        echo "✅ Service is running"
    else
        echo "❌ Service is not running"
        echo "   Try: sudo systemctl start english-corner-ai"
    fi
else
    echo "⚠️  systemctl not available, checking manually..."
    netstat -tulnp | grep :8443 || echo "❌ Port 8443 not listening"
fi
echo ""

# Test 5: Check current CORS headers
echo "5. 🔍 Checking CORS headers from server..."
curl -s -I \
  -H "Origin: $FRONTEND_ORIGIN" \
  "$BACKEND_URL/health" | grep -i "access-control"
echo ""

echo "🎯 Common CORS Issues and Solutions:"
echo ""
echo "If you see CORS errors:"
echo "1. Make sure the service is running: sudo systemctl status english-corner-ai"
echo "2. Check logs: sudo journalctl -u english-corner-ai -f"
echo "3. Restart service: sudo systemctl restart english-corner-ai"
echo "4. Verify nginx/reverse proxy settings if applicable"
echo ""
echo "Expected CORS headers:"
echo "  Access-Control-Allow-Origin: https://www.englishcorner.cyou"
echo "  Access-Control-Allow-Methods: GET, POST, PUT, DELETE, OPTIONS"
echo "  Access-Control-Allow-Headers: *"
echo "  Access-Control-Allow-Credentials: true"
echo ""
echo "🔧 Port Information:"
echo "  Port 8443: Nginx HTTPS reverse proxy (public access)"
echo "  Port 8000: FastAPI backend service (internal only)"
echo "  Frontend should use: https://api.englishcorner.cyou:8443"
echo ""
echo "🏗️ Architecture:"
echo "  Frontend → Nginx (8443) → FastAPI (8000)"
