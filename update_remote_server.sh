#!/bin/bash

# Update Remote Server Script
# This script connects to the remote server, pulls latest code, and restarts the service

set -e  # Exit on any error

echo "🚀 Starting Remote Server Update..."

# VPS Configuration
VPS_IP="92.243.64.55"
VPS_USER="root"
SSH_PORT="22"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_status "Connecting to remote server at $VPS_IP..."

# 1. Connect to VPS and update repository
print_status "Updating SREE repository..."
ssh -o StrictHostKeyChecking=no $VPS_USER@$VPS_IP << 'EOF'
    echo "=== Updating SREE Repository ==="
    cd /home/app/sree
    
    # Check current status
    echo "Current git status:"
    git status --short
    
    # Pull latest changes
    echo "Pulling latest changes..."
    git pull origin main
    
    # Check new status
    echo "Updated git status:"
    git status --short
    
    # Show latest commit
    echo "Latest commit:"
    git log --oneline -1
    
    echo "=== Repository Update Complete ==="
EOF

# 2. Restart the SREE dashboard service
print_status "Restarting SREE dashboard service..."
ssh -o StrictHostKeyChecking=no $VPS_USER@$VPS_IP << 'EOF'
    echo "=== Restarting SREE Dashboard Service ==="
    
    # Stop the service
    echo "Stopping service..."
    systemctl stop sree-dashboard
    
    # Wait a moment
    sleep 2
    
    # Start the service
    echo "Starting service..."
    systemctl start sree-dashboard
    
    # Check service status
    echo "Service status:"
    systemctl status sree-dashboard --no-pager -l
    
    echo "=== Service Restart Complete ==="
EOF

# 3. Verify the deployment
print_status "Verifying deployment..."
ssh -o StrictHostKeyChecking=no $VPS_USER@$VPS_IP << 'EOF'
    echo "=== Verifying Deployment ==="
    
    # Check if service is running
    SERVICE_STATUS=$(systemctl is-active sree-dashboard)
    echo "Service status: $SERVICE_STATUS"
    
    if [ "$SERVICE_STATUS" = "active" ]; then
        echo "✅ SREE Dashboard service is running"
    else
        echo "❌ SREE Dashboard service is not running"
        echo "Recent logs:"
        journalctl -u sree-dashboard -n 10 --no-pager
        exit 1
    fi
    
    # Check if port is accessible
    sleep 5  # Wait for service to fully start
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501)
    echo "HTTP status: $HTTP_STATUS"
    
    if [ "$HTTP_STATUS" = "200" ]; then
        echo "✅ SREE Dashboard is accessible on port 8501"
    else
        echo "❌ SREE Dashboard is not accessible on port 8501"
        echo "Recent logs:"
        journalctl -u sree-dashboard -n 10 --no-pager
        exit 1
    fi
    
    # Run block count verification
    echo "=== Running Block Count Verification ==="
    cd /home/app/sree
    
    if [ -f "verify_block_count_consistency.py" ]; then
        echo "Running block count verification script..."
        python3 verify_block_count_consistency.py
    else
        echo "⚠️  Block count verification script not found"
    fi
    
    echo "=== Verification Complete ==="
EOF

# 4. Display final information
print_status "🎉 Remote server update completed successfully!"
echo ""
echo "=== Update Summary ==="
echo "VPS IP: $VPS_IP"
echo "Dashboard URL: http://$VPS_IP:8501"
echo "Nginx URL: http://$VPS_IP (port 80)"
echo ""
echo "=== Useful Commands ==="
echo "Check service status: ssh $VPS_USER@$VPS_IP 'systemctl status sree-dashboard'"
echo "View logs: ssh $VPS_USER@$VPS_IP 'journalctl -u sree-dashboard -f'"
echo "Monitor system: ssh $VPS_USER@$VPS_IP '/opt/sree/monitor.sh'"
echo "Health check: ssh $VPS_USER@$VPS_IP '/opt/sree/health_check.sh'"
echo "Verify block count: ssh $VPS_USER@$VPS_IP 'cd /opt/sree && python3 verify_block_count_consistency.py'"
echo ""
print_status "Remote server update completed! 🚀" 