#!/bin/bash

# Deploy with Local Tests Script
# This script runs local tests before deploying to ensure everything works correctly

set -e  # Exit on any error

echo "🚀 Starting Deploy with Local Tests..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "${BLUE}[STEP]${NC} $1"
}

# Step 1: Run local tests
print_step "1. Running Local Tests"
print_status "Executing local deployment tests..."

if python3 test_local_deployment.py; then
    print_status "✅ All local tests passed!"
else
    print_error "❌ Local tests failed! Please fix issues before deployment."
    print_error "Run 'python3 test_local_deployment.py' for detailed error information."
    exit 1
fi

# Step 2: Run dashboard local test
print_step "2. Running Dashboard Local Test"
print_status "Testing dashboard locally before deployment..."

if python3 test_dashboard_local.py; then
    print_status "✅ Dashboard local test passed!"
else
    print_error "❌ Dashboard local test failed! Fix dashboard issues before deployment."
    exit 1
fi

# Step 3: Check git status
print_step "3. Checking Git Status"
print_status "Checking for uncommitted changes..."

if [ -n "$(git status --porcelain)" ]; then
    print_warning "⚠️  There are uncommitted changes:"
    git status --short
    
    read -p "Do you want to commit these changes before deployment? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Committing changes..."
        git add .
        git commit -m "Auto-commit before deployment - $(date)"
    else
        print_warning "Proceeding with uncommitted changes..."
    fi
else
    print_status "✅ No uncommitted changes found"
fi

# Step 4: Push to repository
print_step "4. Pushing to Repository"
print_status "Pushing changes to remote repository..."

if git push origin main; then
    print_status "✅ Successfully pushed to repository"
else
    print_error "❌ Failed to push to repository"
    exit 1
fi

# Step 5: Deploy to server
print_step "5. Deploying to Server"
print_status "Starting server deployment..."

if ./update_remote_server.sh; then
    print_status "✅ Server deployment completed successfully"
else
    print_error "❌ Server deployment failed"
    exit 1
fi

# Step 6: Final verification
print_step "6. Final Verification"
print_status "Verifying deployment..."

# Check if server is accessible
VPS_IP="92.243.64.55"
if curl -s -f "http://$VPS_IP:8501" > /dev/null; then
    print_status "✅ Dashboard is accessible at http://$VPS_IP:8501"
else
    print_warning "⚠️  Dashboard might not be accessible yet (checking too early)"
fi

echo ""
echo "🎉 DEPLOYMENT COMPLETED SUCCESSFULLY!"
echo "=" * 60
echo "📊 Dashboard URL: http://$VPS_IP:8501"
echo "📊 Nginx URL: http://$VPS_IP"
echo ""
echo "📋 Deployment Summary:"
echo "   ✅ Local tests passed"
echo "   ✅ Code pushed to repository"
echo "   ✅ Server updated"
echo "   ✅ Service restarted"
echo ""
echo "🔍 Useful Commands:"
echo "   Check service status: ssh root@$VPS_IP 'systemctl status sree-dashboard'"
echo "   View logs: ssh root@$VPS_IP 'journalctl -u sree-dashboard -f'"
echo "   Health check: ssh root@$VPS_IP '/opt/sree/health_check.sh'" 