#!/bin/bash

# SREE Server Update Script
# Updates the server with the latest code changes

set -e

echo "🚀 Updating SREE Server..."

# VPS Configuration
VPS_IP="92.243.64.55"
VPS_USER="root"
VPS_PASS="${VPS_PASSWORD:-}"  # Set via environment variable
SSH_PORT="22"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to run commands on VPS
run_on_vps() {
    sshpass -p "$VPS_PASS" ssh -o StrictHostKeyChecking=no -p $SSH_PORT $VPS_USER@$VPS_IP "$1"
}

# Security check
if [ -z "$VPS_PASS" ]; then
    print_error "❌ VPS_PASSWORD environment variable not set!"
    print_error "Please set it with: export VPS_PASSWORD='your_password'"
    exit 1
fi

print_status "Connecting to VPS at $VPS_IP..."

# 1. Stop the current service
print_status "Stopping current SREE dashboard service..."
run_on_vps "systemctl stop sree-dashboard || true"

# 2. Update the repository
print_status "Updating SREE repository..."
run_on_vps "
    cd /home/app/sree &&
    if [ -d .git ]; then
        git fetch origin &&
        git reset --hard origin/main
    else
        rm -rf * .* 2>/dev/null || true &&
        git clone https://github.com/centerserv/Sree.git .
    fi
"

# 3. Update Python dependencies
print_status "Updating Python dependencies..."
run_on_vps "
    cd /home/app/sree &&
    if [ ! -d venv ]; then
        python3 -m venv venv
    fi &&
    source venv/bin/activate &&
    pip install --upgrade pip &&
    pip install -r requirements.txt
"

# 4. Generate visualizations if needed
print_status "Generating visualizations..."
run_on_vps "
    cd /home/app/sree &&
    source venv/bin/activate &&
    python3 visualization.py
"

# 5. Restart the service
print_status "Restarting SREE dashboard service..."
run_on_vps "
    systemctl daemon-reload &&
    systemctl start sree-dashboard
"

# 6. Wait for service to start
print_status "Waiting for service to start..."
sleep 10

# 7. Check service status
print_status "Checking service status..."
SERVICE_STATUS=$(run_on_vps "systemctl is-active sree-dashboard")
if [ "$SERVICE_STATUS" = "active" ]; then
    print_status "✅ SREE Dashboard service is running"
else
    print_error "❌ SREE Dashboard service is not running"
    run_on_vps "journalctl -u sree-dashboard -n 20 --no-pager"
    exit 1
fi

# 8. Test HTTP access
print_status "Testing HTTP access..."
HTTP_STATUS=$(run_on_vps "curl -s -o /dev/null -w '%{http_code}' http://localhost:8501")
if [ "$HTTP_STATUS" = "200" ]; then
    print_status "✅ SREE Dashboard is accessible on port 8501"
else
    print_error "❌ SREE Dashboard is not accessible on port 8501 (HTTP status: $HTTP_STATUS)"
    exit 1
fi

# 9. Display final information
print_status "🎉 SREE Server update completed successfully!"
echo ""
echo "=== Server Information ==="
echo "VPS IP: $VPS_IP"
echo "Dashboard URL: http://$VPS_IP:8501"
echo "Nginx URL: http://$VPS_IP (port 80)"
echo ""
echo "=== Useful Commands ==="
echo "Check service status: ssh $VPS_USER@$VPS_IP 'systemctl status sree-dashboard'"
echo "View logs: ssh $VPS_USER@$VPS_IP 'journalctl -u sree-dashboard -f'"
echo "Monitor system: ssh $VPS_USER@$VPS_IP '/home/app/sree/monitor.sh'"
echo "Health check: ssh $VPS_USER@$VPS_IP '/home/app/sree/health_check.sh'"
echo ""
print_status "Server is ready for client testing! 🚀" 