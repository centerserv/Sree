#!/bin/bash

# SREE Master Script
# Unified script for all SREE operations: testing, deployment, and server management

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_header() {
    echo -e "${PURPLE}$1${NC}"
}

print_subheader() {
    echo -e "${CYAN}$1${NC}"
}

# Load environment variables from .env file
if [ -f .env ]; then
    echo "📁 Loading configuration from .env file..."
    export $(cat .env | grep -v '^#' | xargs)
fi

# VPS Configuration
VPS_IP="${VPS_IP:-92.243.64.55}"
VPS_USER="${VPS_USER:-root}"
VPS_PASS="${VPS_PASSWORD:-}"
SSH_PORT="${SSH_PORT:-22}"

# Function to run commands on VPS
run_on_vps() {
    if [ -n "$VPS_PASS" ]; then
        sshpass -p "$VPS_PASS" ssh -o StrictHostKeyChecking=no -p $SSH_PORT $VPS_USER@$VPS_IP "$1"
    else
        ssh -o StrictHostKeyChecking=no -p $SSH_PORT $VPS_USER@$VPS_IP "$1"
    fi
}

# Function to copy files to VPS
copy_to_vps() {
    if [ -n "$VPS_PASS" ]; then
        sshpass -p "$VPS_PASS" scp -o StrictHostKeyChecking=no -P $SSH_PORT -r "$1" $VPS_USER@$VPS_IP:"$2"
    else
        scp -o StrictHostKeyChecking=no -P $SSH_PORT -r "$1" $VPS_USER@$VPS_IP:"$2"
    fi
}

# Function to show usage
show_usage() {
    echo "🚀 SREE Master Script"
    echo "===================="
    echo ""
    echo "Usage: ./sree_master.sh [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  test              - Run quick tests (< 30 seconds)"
    echo "  test-full         - Run all tests in parallel"
    echo "  deploy            - Deploy with tests (recommended)"
    echo "  deploy-vps        - Deploy to VPS server"
    echo "  start             - Start dashboard locally"
    echo "  update            - Update server with latest code"
    echo "  update-remote     - Update remote server (SSH only)"
    echo "  status            - Check server status"
    echo "  logs              - View server logs"
    echo "  monitor           - Monitor server performance"
    echo "  health            - Run health check"
    echo "  help              - Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./sree_master.sh test        # Quick validation"
    echo "  ./sree_master.sh deploy      # Deploy with tests"
    echo "  ./sree_master.sh start       # Start locally"
    echo "  ./sree_master.sh status      # Check server"
    echo ""
}

# Function to run quick tests
run_quick_tests() {
    print_header "🧪 Running Quick Tests..."
    print_subheader "Fast validation for deployment (target: <30 seconds)"
    echo ""
    
    if python3 quick_test.py; then
        print_status "Quick tests passed!"
        return 0
    else
        print_error "Quick tests failed! Fix issues before deploying."
        return 1
    fi
}

# Function to run full tests
run_full_tests() {
    print_header "⚡ Running Full Tests in Parallel..."
    print_subheader "Complete validation (2-5 minutes)"
    echo ""
    
    if python3 run_parallel_tests.py --workers 8; then
        print_status "All tests passed!"
        return 0
    else
        print_error "Some tests failed!"
        return 1
    fi
}

# Function to deploy with tests
deploy_with_tests() {
    print_header "🚀 Deploy with Tests"
    print_subheader "Safe deployment with validation"
    echo ""
    
    # Step 1: Run quick tests
    print_info "Step 1: Running Quick Tests..."
    if ! run_quick_tests; then
        exit 1
    fi
    
    echo ""
    
    # Step 2: Check git status
    print_info "Step 2: Checking Git Status..."
    if [[ -n $(git status --porcelain) ]]; then
        print_warning "You have uncommitted changes:"
        git status --short
        
        echo ""
        read -p "Do you want to commit these changes? (y/n): " -n 1 -r
        echo
        
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "📝 Committing changes..."
            git add .
            git commit -m "Auto-commit before deployment $(date '+%Y-%m-%d %H:%M:%S')"
            print_status "Changes committed!"
        else
            print_warning "Skipping commit. Make sure to commit manually."
        fi
    else
        print_status "No uncommitted changes."
    fi
    
    echo ""
    
    # Step 3: Check branch
    print_info "Step 3: Checking Branch..."
    current_branch=$(git branch --show-current)
    if [[ "$current_branch" == "main" ]]; then
        print_status "On main branch."
    else
        print_warning "Not on main branch (currently on: $current_branch)"
        read -p "Do you want to continue anyway? (y/n): " -n 1 -r
        echo
        
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "Deployment cancelled."
            exit 0
        fi
    fi
    
    echo ""
    
    # Step 4: Push to remote
    print_info "Step 4: Pushing to Remote..."
    if git push origin $current_branch; then
        print_status "Successfully pushed to remote!"
    else
        print_error "Failed to push to remote!"
        exit 1
    fi
    
    echo ""
    print_status "🎉 Deployment completed successfully!"
    print_info "Code pushed successfully"
    print_info "Quick tests passed"
    print_info "Ready for production!"
}

# Function to deploy to VPS
deploy_to_vps() {
    print_header "🚀 Deploying to VPS Server..."
    
    # Security check
    if [ -z "$VPS_PASS" ]; then
        print_error "VPS_PASSWORD environment variable not set!"
        print_error "Please set it with: export VPS_PASSWORD='your_password'"
        exit 1
    fi
    
    print_info "Connecting to VPS at $VPS_IP..."
    
    # 1. Update system and install dependencies
    print_info "Updating system and installing dependencies..."
    run_on_vps "
        apt update -y &&
        apt upgrade -y &&
        apt install -y python3 python3-pip python3-venv git curl wget ufw &&
        apt install -y python3-dev build-essential libssl-dev libffi-dev
    "
    
    # 2. Configure firewall
    print_info "Configuring firewall..."
    run_on_vps "
        ufw --force enable &&
        ufw allow ssh &&
        ufw allow 8501 &&
        ufw status
    "
    
    # 3. Create application directory
    print_info "Creating application directory..."
    run_on_vps "
        mkdir -p /home/app/sree &&
        cd /home/app/sree
    "
    
    # 4. Clone or update repository
    print_info "Updating SREE repository..."
    run_on_vps "
        cd /home/app/sree &&
        if [ -d .git ]; then
            git pull origin main
        else
            git clone https://github.com/centerserv/Sree.git .
        fi
    "
    
    # 5. Set up Python virtual environment
    print_info "Setting up Python virtual environment..."
    run_on_vps "
        cd /home/app/sree &&
        python3 -m venv venv &&
        source venv/bin/activate &&
        pip install --upgrade pip &&
        pip install -r requirements.txt
    "
    
    # 6. Create systemd service
    print_info "Creating systemd service..."
    run_on_vps "
        cat > /etc/systemd/system/sree-dashboard.service << 'EOF'
[Unit]
Description=SREE Dashboard
After=network.target

[Service]
Type=simple
User=root
Group=root
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
    print_info "Starting SREE dashboard service..."
    run_on_vps "
        systemctl daemon-reload &&
        systemctl enable sree-dashboard &&
        systemctl start sree-dashboard &&
        systemctl status sree-dashboard
    "
    
    # 8. Test the deployment
    print_info "Testing deployment..."
    sleep 10
    
    SERVICE_STATUS=$(run_on_vps "systemctl is-active sree-dashboard")
    if [ "$SERVICE_STATUS" = "active" ]; then
        print_status "SREE Dashboard service is running"
    else
        print_error "SREE Dashboard service is not running"
        run_on_vps "journalctl -u sree-dashboard -n 20 --no-pager"
        exit 1
    fi
    
    HTTP_STATUS=$(run_on_vps "curl -s -o /dev/null -w '%{http_code}' http://localhost:8501")
    if [ "$HTTP_STATUS" = "200" ]; then
        print_status "SREE Dashboard is accessible on port 8501"
    else
        print_error "SREE Dashboard is not accessible on port 8501 (HTTP status: $HTTP_STATUS)"
        exit 1
    fi
    
    print_status "🎉 VPS deployment completed successfully!"
    echo ""
    echo "=== Deployment Summary ==="
    echo "VPS IP: $VPS_IP"
    echo "Dashboard URL: http://$VPS_IP:8501"
}

# Function to start dashboard locally
start_dashboard() {
    print_header "🚀 Starting SREE Dashboard Locally"
    print_subheader "Local development server with verbose logging"
    echo ""
    
    # Check if required files exist
    print_info "Checking required files..."
    if [ -f "dashboard.py" ]; then
        print_status "dashboard.py"
    else
        print_error "dashboard.py - NOT FOUND!"
        exit 1
    fi
    
    if [ -f "heart_disease_dataset_new.csv" ]; then
        print_status "heart_disease_dataset_new.csv"
    else
        print_warning "heart_disease_dataset_new.csv - NOT FOUND!"
    fi
    
    # Check Python environment
    print_info "Checking Python environment..."
    if command -v python3 &> /dev/null; then
        print_status "Python3 found: $(python3 --version)"
    else
        print_error "Python3 not found!"
        exit 1
    fi
    
    # Check Streamlit installation
    print_info "Checking Streamlit installation..."
    
    # Check if we have a virtual environment and use it
    if [ -d "venv" ]; then
        print_info "Virtual environment found. Activating..."
        source venv/bin/activate
        if python3 -c "import streamlit" 2>/dev/null; then
            print_status "Streamlit is installed in virtual environment"
        else
            print_warning "Streamlit not found in virtual environment. Installing..."
            pip install streamlit
        fi
    else
        # Fallback to system Python with virtual environment creation
        if python3 -c "import streamlit" 2>/dev/null; then
            print_status "Streamlit is installed"
        else
            print_warning "Streamlit not found. Creating virtual environment and installing..."
            python3 -m venv venv
            source venv/bin/activate
            pip install -r requirements.txt
        fi
    fi
    
    echo ""
    print_info "Dashboard will be available at: http://localhost:8501"
    echo ""
    print_info "Verbose logging enabled for better visibility:"
    echo "   📊 Real-time processing logs"
    echo "   🔍 Detailed analysis progress"
    echo "   ⚡ Performance metrics"
    echo "   🐛 Error tracking and debugging"
    echo ""
    print_info "Instructions:"
    echo "   1. The dashboard will open automatically in your browser"
    echo "   2. Upload your CSV dataset"
    echo "   3. Click 'Run SREE Analysis'"
    echo "   4. Watch the console for detailed processing logs"
    echo "   5. Press Ctrl+C to stop the dashboard"
    echo ""
    print_info "Starting dashboard with verbose logging..."
    
    # Set environment variables for verbose logging
    export STREAMLIT_SERVER_HEADLESS=false
    export STREAMLIT_SERVER_RUN_ON_SAVE=true
    export STREAMLIT_SERVER_FILE_WATCHER_TYPE=poll
    export STREAMLIT_LOGGER_LEVEL=debug
    export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
    
             # Start streamlit with verbose options (using virtual environment if available)
             if [ -d "venv" ]; then
                 venv/bin/streamlit run dashboard.py \
                     --server.port 8501 \
                     --server.address localhost \
                     --server.headless false \
                     --server.runOnSave true \
                     --server.fileWatcherType poll \
                     --logger.level debug \
                     --browser.gatherUsageStats false \
                     --global.showWarningOnDirectExecution true \
                     --theme.base light \
                     --theme.primaryColor "#FF4B4B" \
                     --theme.backgroundColor "#FFFFFF" \
                     --theme.secondaryBackgroundColor "#F0F2F6" \
                     --theme.textColor "#262730" \
                     --theme.font sans
             else
                 streamlit run dashboard.py \
                     --server.port 8501 \
                     --server.address localhost \
                     --server.headless false \
                     --server.runOnSave true \
                     --server.fileWatcherType poll \
                     --logger.level debug \
                     --browser.gatherUsageStats false \
                     --global.showWarningOnDirectExecution true \
                     --theme.base light \
                     --theme.primaryColor "#FF4B4B" \
                     --theme.backgroundColor "#FFFFFF" \
                     --theme.secondaryBackgroundColor "#F0F2F6" \
                     --theme.textColor "#262730" \
                     --theme.font sans
             fi
}

# Function to update server
update_server() {
    print_header "🔄 Updating Server..."
    
    if [ -z "$VPS_PASS" ]; then
        print_error "VPS_PASSWORD environment variable not set!"
        print_error "Please set it with: export VPS_PASSWORD='your_password'"
        exit 1
    fi
    
    print_info "Connecting to VPS at $VPS_IP..."
    
    # 1. Stop the current service
    print_info "Stopping current SREE dashboard service..."
    run_on_vps "systemctl stop sree-dashboard || true"
    
    # 2. Update the repository
    print_info "Updating SREE repository..."
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
    print_info "Updating Python dependencies..."
    run_on_vps "
        cd /home/app/sree &&
        if [ ! -d venv ]; then
            python3 -m venv venv
        fi &&
        source venv/bin/activate &&
        pip install --upgrade pip &&
        pip install -r requirements.txt
    "
    
    # 4. Restart the service
    print_info "Restarting SREE dashboard service..."
    run_on_vps "
        systemctl daemon-reload &&
        systemctl start sree-dashboard
    "
    
    # 5. Wait and check status
    print_info "Waiting for service to start..."
    sleep 10
    
    SERVICE_STATUS=$(run_on_vps "systemctl is-active sree-dashboard")
    if [ "$SERVICE_STATUS" = "active" ]; then
        print_status "SREE Dashboard service is running"
    else
        print_error "SREE Dashboard service is not running"
        run_on_vps "journalctl -u sree-dashboard -n 20 --no-pager"
        exit 1
    fi
    
    HTTP_STATUS=$(run_on_vps "curl -s -o /dev/null -w '%{http_code}' http://localhost:8501")
    if [ "$HTTP_STATUS" = "200" ]; then
        print_status "SREE Dashboard is accessible on port 8501"
    else
        print_error "SREE Dashboard is not accessible on port 8501 (HTTP status: $HTTP_STATUS)"
        exit 1
    fi
    
    print_status "🎉 Server update completed successfully!"
    echo ""
    echo "=== Server Information ==="
    echo "VPS IP: $VPS_IP"
    echo "Dashboard URL: http://$VPS_IP:8501"
}

# Function to update remote server (SSH only)
update_remote_server() {
    print_header "🔄 Updating Remote Server (SSH only)..."
    
    print_info "Connecting to remote server at $VPS_IP..."
    
    # 1. Update repository
    print_info "Updating SREE repository..."
    ssh -o StrictHostKeyChecking=no $VPS_USER@$VPS_IP << 'EOF'
        echo "=== Updating SREE Repository ==="
        cd /home/app/sree
        
        echo "Current git status:"
        git status --short
        
        echo "Pulling latest changes..."
        git pull origin main
        
        echo "Updated git status:"
        git status --short
        
        echo "Latest commit:"
        git log --oneline -1
        
        echo "=== Repository Update Complete ==="
EOF
    
    # 2. Restart service
    print_info "Restarting SREE dashboard service..."
    ssh -o StrictHostKeyChecking=no $VPS_USER@$VPS_IP << 'EOF'
        echo "=== Restarting SREE Dashboard Service ==="
        
        echo "Stopping service..."
        systemctl stop sree-dashboard
        
        sleep 2
        
        echo "Starting service..."
        systemctl start sree-dashboard
        
        echo "Service status:"
        systemctl status sree-dashboard --no-pager -l
        
        echo "=== Service Restart Complete ==="
EOF
    
    # 3. Verify deployment
    print_info "Verifying deployment..."
    ssh -o StrictHostKeyChecking=no $VPS_USER@$VPS_IP << 'EOF'
        echo "=== Verifying Deployment ==="
        
        SERVICE_STATUS=$(systemctl is-active sree-dashboard)
        echo "Service status: $SERVICE_STATUS"
        
        if [ "$SERVICE_STATUS" = "active" ]; then
            echo "✅ SREE Dashboard service is running"
        else
            echo "❌ SREE Dashboard service is not running"
            journalctl -u sree-dashboard -n 10 --no-pager
            exit 1
        fi
        
        sleep 5
        HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501)
        echo "HTTP status: $HTTP_STATUS"
        
        if [ "$HTTP_STATUS" = "200" ]; then
            echo "✅ SREE Dashboard is accessible on port 8501"
        else
            echo "❌ SREE Dashboard is not accessible on port 8501"
            journalctl -u sree-dashboard -n 10 --no-pager
            exit 1
        fi
        
        echo "=== Verification Complete ==="
EOF
    
    print_status "🎉 Remote server update completed successfully!"
    echo ""
    echo "=== Update Summary ==="
    echo "VPS IP: $VPS_IP"
    echo "Dashboard URL: http://$VPS_IP:8501"
}

# Function to check server status
check_status() {
    print_header "📊 Checking Server Status..."
    
    print_info "Connecting to server at $VPS_IP..."
    
    ssh -o StrictHostKeyChecking=no $VPS_USER@$VPS_IP << 'EOF'
        echo "=== SREE Dashboard Status ==="
        echo "Timestamp: $(date)"
        echo ""
        
        echo "Service Status:"
        systemctl status sree-dashboard --no-pager -l
        echo ""
        
        echo "Port Status:"
        netstat -tlnp | grep :8501 || echo "Port 8501 not found"
        echo ""
        
        echo "Memory Usage:"
        ps aux | grep streamlit | grep -v grep || echo "No streamlit processes found"
        echo ""
        
        echo "=== Status Check Complete ==="
EOF
}

# Function to view logs
view_logs() {
    print_header "📋 Viewing Server Logs..."
    
    print_info "Connecting to server at $VPS_IP..."
    
    ssh -o StrictHostKeyChecking=no $VPS_USER@$VPS_IP << 'EOF'
        echo "=== SREE Dashboard Logs ==="
        echo "Last 20 log entries:"
        journalctl -u sree-dashboard -n 20 --no-pager
        echo ""
        echo "=== Logs Complete ==="
EOF
}

# Function to monitor server
monitor_server() {
    print_header "📈 Monitoring Server Performance..."
    
    print_info "Connecting to server at $VPS_IP..."
    
    ssh -o StrictHostKeyChecking=no $VPS_USER@$VPS_IP << 'EOF'
        echo "=== SREE Dashboard Monitoring ==="
        echo "Timestamp: $(date)"
        echo ""
        
        echo "System Load:"
        uptime
        echo ""
        
        echo "Memory Usage:"
        free -h
        echo ""
        
        echo "Disk Usage:"
        df -h
        echo ""
        
        echo "SREE Process:"
        ps aux | grep streamlit | grep -v grep
        echo ""
        
        echo "Network Connections:"
        netstat -tlnp | grep :8501
        echo ""
        
        echo "=== Monitoring Complete ==="
EOF
}

# Function to run health check
health_check() {
    print_header "🏥 Running Health Check..."
    
    print_info "Connecting to server at $VPS_IP..."
    
    ssh -o StrictHostKeyChecking=no $VPS_USER@$VPS_IP << 'EOF'
        echo "=== SREE Dashboard Health Check ==="
        
        SERVICE_STATUS=$(systemctl is-active sree-dashboard)
        HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8501)
        
        echo "Service Status: $SERVICE_STATUS"
        echo "HTTP Status: $HTTP_STATUS"
        echo ""
        
        if [ "$SERVICE_STATUS" = "active" ] && [ "$HTTP_STATUS" = "200" ]; then
            echo "✅ SREE Dashboard is healthy"
            exit 0
        else
            echo "❌ SREE Dashboard has issues"
            echo "Service status: $SERVICE_STATUS"
            echo "HTTP status: $HTTP_STATUS"
            exit 1
        fi
EOF
    
    if [ $? -eq 0 ]; then
        print_status "Health check passed!"
    else
        print_error "Health check failed!"
    fi
}

# Main script logic
case "${1:-}" in
    "test")
        run_quick_tests
        ;;
    "test-full")
        run_full_tests
        ;;
    "deploy")
        deploy_with_tests
        ;;
    "deploy-vps")
        deploy_to_vps
        ;;
    "start")
        start_dashboard
        ;;
    "update")
        update_server
        ;;
    "update-remote")
        update_remote_server
        ;;
    "status")
        check_status
        ;;
    "logs")
        view_logs
        ;;
    "monitor")
        monitor_server
        ;;
    "health")
        health_check
        ;;
    "help"|"-h"|"--help"|"")
        show_usage
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac 