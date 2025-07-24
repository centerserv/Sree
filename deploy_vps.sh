#!/bin/bash

# SREE Phase 1 Demo - VPS Deployment Script
# This script deploys the SREE demo on Ubuntu 20.04 VPS

set -e  # Exit on any error

echo "🚀 Starting SREE VPS Deployment..."

# VPS Configuration
VPS_IP="92.243.64.55"
VPS_USER="root"
VPS_PASS="${VPS_PASSWORD:-}"  # Set via environment variable
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

# Function to run commands on VPS
run_on_vps() {
    sshpass -p "$VPS_PASS" ssh -o StrictHostKeyChecking=no -p $SSH_PORT $VPS_USER@$VPS_IP "$1"
}

# Function to copy files to VPS
copy_to_vps() {
    sshpass -p "$VPS_PASS" scp -o StrictHostKeyChecking=no -P $SSH_PORT -r "$1" $VPS_USER@$VPS_IP:"$2"
}

# Security check
if [ -z "$VPS_PASS" ]; then
    print_error "❌ VPS_PASSWORD environment variable not set!"
    print_error "Please set it with: export VPS_PASSWORD='your_password'"
    exit 1
fi

print_status "Connecting to VPS at $VPS_IP..."

# 1. Update system and install dependencies
print_status "Updating system and installing dependencies..."
run_on_vps "
    apt update -y &&
    apt upgrade -y &&
    apt install -y python3 python3-pip python3-venv git curl wget ufw &&
    apt install -y python3-dev build-essential libssl-dev libffi-dev
"

# 2. Configure firewall
print_status "Configuring firewall..."
run_on_vps "
    ufw --force enable &&
    ufw allow ssh &&
    ufw allow 8501 &&
    ufw status
"

# 3. Create application directory
print_status "Creating application directory..."
run_on_vps "
    mkdir -p /home/app/sree &&
    cd /home/app/sree
"

# 4. Clone or update repository
print_status "Updating SREE repository..."
run_on_vps "
    cd /opt/sree &&
    if [ -d .git ]; then
        git pull origin main
    else
        git clone https://github.com/centerserv/Sree.git .
    fi
"

# 5. Set up Python virtual environment
print_status "Setting up Python virtual environment..."
run_on_vps "
    cd /home/app/sree &&
    python3 -m venv venv &&
    source venv/bin/activate &&
    pip install --upgrade pip &&
    pip install -r requirements.txt
"

# 6. Create systemd service for Streamlit
print_status "Creating systemd service..."
run_on_vps "
    cat > /etc/systemd/system/sree-dashboard.service << 'EOF'
[Unit]
Description=SREE Dashboard
After=network.target

[Service]
Type=simple
User=app
Group=app
WorkingDirectory=/home/app/sree
Environment=PATH=/home/app/sree/venv/bin
ExecStart=/home/app/sree/venv/bin/streamlit run dashboard.py --server.port 8501 --server.address 0.0.0.0 --server.headless true
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
"

# 7. Enable and start the service
print_status "Starting SREE dashboard service..."
run_on_vps "
    systemctl daemon-reload &&
    systemctl enable sree-dashboard &&
    systemctl start sree-dashboard &&
    systemctl status sree-dashboard
"

# 8. Create Nginx configuration (optional, for better performance)
print_status "Setting up Nginx reverse proxy..."
run_on_vps "
    apt install -y nginx &&
    cat > /etc/nginx/sites-available/sree << 'EOF'
server {
    listen 80;
    server_name $VPS_IP;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_cache_bypass \$http_upgrade;
    }
}
EOF
"

run_on_vps "
    ln -sf /etc/nginx/sites-available/sree /etc/nginx/sites-enabled/ &&
    rm -f /etc/nginx/sites-enabled/default &&
    nginx -t &&
    systemctl restart nginx &&
    ufw allow 'Nginx Full'
"

# 9. Create health check script
print_status "Creating health check script..."
run_on_vps "
    cat > /home/app/sree/health_check.sh << 'EOF'
#!/bin/bash
# Health check script for SREE dashboard

SERVICE_STATUS=\$(systemctl is-active sree-dashboard)
HTTP_STATUS=\$(curl -s -o /dev/null -w \"%{http_code}\" http://localhost:8501)

if [ \"\$SERVICE_STATUS\" = \"active\" ] && [ \"\$HTTP_STATUS\" = \"200\" ]; then
    echo \"✅ SREE Dashboard is running properly\"
    exit 0
else
    echo \"❌ SREE Dashboard is not running properly\"
    echo \"Service status: \$SERVICE_STATUS\"
    echo \"HTTP status: \$HTTP_STATUS\"
    exit 1
fi
EOF
"

run_on_vps "
    chmod +x /opt/sree/health_check.sh
"

# 10. Create monitoring script
print_status "Creating monitoring script..."
run_on_vps "
    cat > /home/app/sree/monitor.sh << 'EOF'
#!/bin/bash
# Monitoring script for SREE dashboard

echo \"=== SREE Dashboard Status ===\"
echo \"Timestamp: \$(date)\"
echo \"\"

echo \"Service Status:\"
systemctl status sree-dashboard --no-pager -l
echo \"\"

echo \"Port Status:\"
netstat -tlnp | grep :8501
echo \"\"

echo \"Memory Usage:\"
ps aux | grep streamlit | grep -v grep
echo \"\"

echo \"Logs (last 10 lines):\"
journalctl -u sree-dashboard -n 10 --no-pager
EOF
"

run_on_vps "
    chmod +x /opt/sree/monitor.sh
"

# 11. Test the deployment
print_status "Testing deployment..."
sleep 10  # Wait for service to start

# Check if service is running
SERVICE_STATUS=$(run_on_vps "systemctl is-active sree-dashboard")
if [ "$SERVICE_STATUS" = "active" ]; then
    print_status "✅ SREE Dashboard service is running"
else
    print_error "❌ SREE Dashboard service is not running"
    run_on_vps "journalctl -u sree-dashboard -n 20 --no-pager"
    exit 1
fi

# Check if port is accessible
HTTP_STATUS=$(run_on_vps "curl -s -o /dev/null -w '%{http_code}' http://localhost:8501")
if [ "$HTTP_STATUS" = "200" ]; then
    print_status "✅ SREE Dashboard is accessible on port 8501"
else
    print_error "❌ SREE Dashboard is not accessible on port 8501 (HTTP status: $HTTP_STATUS)"
    exit 1
fi

# 12. Display final information
print_status "🎉 SREE Dashboard deployment completed successfully!"
echo ""
echo "=== Deployment Summary ==="
echo "VPS IP: $VPS_IP"
echo "Dashboard URL: http://$VPS_IP:8501"
echo "Nginx URL: http://$VPS_IP (port 80)"
echo ""
echo "=== Useful Commands ==="
echo "Check service status: ssh $VPS_USER@$VPS_IP 'systemctl status sree-dashboard'"
echo "View logs: ssh $VPS_USER@$VPS_IP 'journalctl -u sree-dashboard -f'"
echo "Monitor system: ssh $VPS_USER@$VPS_IP '/opt/sree/monitor.sh'"
echo "Health check: ssh $VPS_USER@$VPS_IP '/opt/sree/health_check.sh'"
echo ""
echo "=== Next Steps ==="
echo "1. Visit http://$VPS_IP:8501 to access the dashboard"
echo "2. Upload your CSV datasets for analysis"
echo "3. Monitor the system using the provided scripts"
echo ""
print_status "Deployment completed! 🚀" 