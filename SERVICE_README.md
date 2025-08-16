# English Corner AI Backend Service

This directory contains the configuration and scripts to run the English Corner AI Backend as a persistent system service.

## 🚀 Quick Setup

1. **Copy files to server:**
   ```bash
   scp -r . user@your-server:/home/ef/english-corner-ai-backend/
   ```

2. **Run setup script:**
   ```bash
   cd /home/ef/english-corner-ai-backend
   chmod +x setup_service.sh service.sh
   sudo ./setup_service.sh
   ```

## 📁 Files Included

- `english-corner-ai.service` - Systemd service configuration
- `setup_service.sh` - Automated setup script
- `service.sh` - Quick service management commands
- `rag_backend.py` - Main FastAPI application
- `requirements.txt` - Python dependencies

## 🔧 Manual Setup (if needed)

### 1. Install Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Setup Environment
```bash
# Copy your .env file with API keys
cp .env.example .env
# Edit .env with your actual API keys
nano .env
```

### 3. Install Service
```bash
# Copy service file
sudo cp english-corner-ai.service /etc/systemd/system/

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable english-corner-ai
sudo systemctl start english-corner-ai
```

## 🎛️ Service Management

### Using the service script:
```bash
# Start service
./service.sh start

# Stop service
./service.sh stop

# Restart service
./service.sh restart

# Check status
./service.sh status

# View live logs
./service.sh logs

# Enable auto-start on boot
./service.sh enable
```

### Using systemctl directly:
```bash
# Check service status
sudo systemctl status english-corner-ai

# View logs
sudo journalctl -u english-corner-ai -f

# Restart service
sudo systemctl restart english-corner-ai

# Stop service
sudo systemctl stop english-corner-ai

# Start service
sudo systemctl start english-corner-ai
```

## 🌐 Service Configuration

The service runs on:
- **Host:** 0.0.0.0 (all interfaces)
- **Port:** 8443 (HTTPS port)
- **Workers:** 1 (single worker for memory consistency)
- **User:** ef
- **Auto-restart:** Yes
- **Log location:** System journal (`journalctl`)

## 🔍 Troubleshooting

### Check if service is running:
```bash
sudo systemctl is-active english-corner-ai
```

### View detailed logs:
```bash
sudo journalctl -u english-corner-ai -n 50
```

### Test API directly:
```bash
curl https://api.englishcorner.cyou:8443/health
```

### Common issues:

1. **Permission denied:** Make sure files are owned by the `ef` user
2. **Port already in use:** Check if another service is using port 8443
3. **Missing dependencies:** Ensure all requirements are installed in the virtual environment
4. **Environment variables:** Verify `.env` file contains all required API keys

## 🔒 Security Features

The service includes several security settings:
- Runs as non-root user (`ef`)
- Read-only filesystem protection
- Private temporary directory
- No privilege escalation
- Protected home directory access

## 📊 Monitoring

The service logs all activities to the system journal. You can:
- Monitor real-time logs: `sudo journalctl -u english-corner-ai -f`
- Check service health: `curl https://api.englishcorner.cyou:8443/health`
- View service status: `sudo systemctl status english-corner-ai`

## 🔄 Updates

To update the service:
1. Stop the service: `./service.sh stop`
2. Update code files
3. Install new dependencies if needed
4. Restart the service: `./service.sh start`

## 🆘 Emergency Recovery

If the service fails to start:
1. Check logs: `sudo journalctl -u english-corner-ai -n 50`
2. Test manually: `cd /home/ef/english-corner-ai-backend && venv/bin/uvicorn rag_backend:app --host 0.0.0.0 --port 8443`
3. Check permissions: `ls -la /home/ef/english-corner-ai-backend`
4. Verify environment: `sudo -u ef venv/bin/python -c "import fastapi; print('OK')"`
