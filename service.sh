#!/bin/bash

# Quick service management script for English Corner AI Backend

SERVICE_NAME="english-corner-ai"

case "$1" in
    start)
        echo "🚀 Starting English Corner AI Backend service..."
        sudo systemctl start $SERVICE_NAME
        ;;
    stop)
        echo "🛑 Stopping English Corner AI Backend service..."
        sudo systemctl stop $SERVICE_NAME
        ;;
    restart)
        echo "🔄 Restarting English Corner AI Backend service..."
        sudo systemctl restart $SERVICE_NAME
        ;;
    status)
        echo "📊 English Corner AI Backend service status:"
        sudo systemctl status $SERVICE_NAME --no-pager
        ;;
    logs)
        echo "📝 Showing live logs (Ctrl+C to exit):"
        sudo journalctl -u $SERVICE_NAME -f
        ;;
    enable)
        echo "🔧 Enabling service to start on boot..."
        sudo systemctl enable $SERVICE_NAME
        ;;
    disable)
        echo "🚫 Disabling service from starting on boot..."
        sudo systemctl disable $SERVICE_NAME
        ;;
    *)
        echo "🤖 English Corner AI Backend Service Manager"
        echo ""
        echo "Usage: $0 {start|stop|restart|status|logs|enable|disable}"
        echo ""
        echo "Commands:"
        echo "  start    - Start the service"
        echo "  stop     - Stop the service"
        echo "  restart  - Restart the service"
        echo "  status   - Show service status"
        echo "  logs     - Show live logs"
        echo "  enable   - Enable auto-start on boot"
        echo "  disable  - Disable auto-start on boot"
        exit 1
        ;;
esac

# Show status after operations (except for logs)
if [ "$1" != "logs" ] && [ "$1" != "status" ]; then
    echo ""
    echo "📊 Current status:"
    sudo systemctl is-active $SERVICE_NAME > /dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✅ Service is running"
    else
        echo "❌ Service is not running"
    fi
fi
