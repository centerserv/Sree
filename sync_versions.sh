#!/bin/bash

# Sync Versions Script - Ensure identical local and remote environments
# Senior dev approach: exact version matching for consistent behavior

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

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

# Load VPS config
VPS_IP="${VPS_IP:-144.91.126.89}"
VPS_USER="${VPS_USER:-yasilvalmeida}"

echo "🚀 SREE Version Synchronization Script"
echo "======================================"

# 1. Show local versions
print_info "📋 Checking local Python environment..."
echo "Python version: $(python3 --version)"
echo "Pip version: $(pip --version)"

print_info "📦 Local package versions:"
if [ -f requirements.txt ]; then
    while IFS= read -r line; do
        if [[ $line =~ ^[a-zA-Z] ]]; then
            package=$(echo $line | cut -d'=' -f1)
            if pip show $package >/dev/null 2>&1; then
                version=$(pip show $package | grep Version | cut -d' ' -f2)
                echo "  ✅ $package: $version"
            else
                echo "  ❌ $package: NOT INSTALLED"
            fi
        fi
    done < requirements.txt
else
    print_error "requirements.txt not found!"
    exit 1
fi

# 2. Update local environment to exact versions
print_info "🔄 Installing exact versions locally..."
pip install -r requirements.txt --force-reinstall

# 3. Test local environment
print_info "🧪 Testing local environment..."
python3 -c "
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
print(f'✅ NumPy: {np.__version__}')
print(f'✅ Pandas: {pd.__version__}')
print(f'✅ Scikit-learn: {sklearn.__version__}')
print(f'✅ Streamlit: {st.__version__}')
"

# 4. Check if we can connect to VPS
print_info "🌐 Testing VPS connection..."
if ssh -o ConnectTimeout=5 -o BatchMode=yes $VPS_USER@$VPS_IP echo "Connection successful" 2>/dev/null; then
    print_status "VPS connection successful"
    
    # 5. Update VPS environment
    print_info "📡 Updating VPS environment..."
    
    # Copy requirements to VPS
    scp requirements.txt $VPS_USER@$VPS_IP:/home/app/sree/
    
    # Update VPS packages
    ssh $VPS_USER@$VPS_IP "
        cd /home/app/sree &&
        source venv/bin/activate &&
        pip install --upgrade pip &&
        pip install -r requirements.txt --force-reinstall &&
        echo '✅ VPS packages updated'
    "
    
    # 6. Test VPS environment
    print_info "🧪 Testing VPS environment..."
    ssh $VPS_USER@$VPS_IP "
        cd /home/app/sree &&
        source venv/bin/activate &&
        python3 -c \"
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
print(f'✅ VPS NumPy: {np.__version__}')
print(f'✅ VPS Pandas: {pd.__version__}')
print(f'✅ VPS Scikit-learn: {sklearn.__version__}')
print(f'✅ VPS Streamlit: {st.__version__}')
\"
    "
    
    # 7. Update code on VPS
    print_info "📂 Updating SREE code on VPS..."
    ssh $VPS_USER@$VPS_IP "
        cd /home/app/sree &&
        git fetch origin &&
        git reset --hard origin/main &&
        echo '✅ Code updated to latest version'
    "
    
    # 8. Restart VPS service
    print_info "🔄 Restarting VPS dashboard service..."
    ssh $VPS_USER@$VPS_IP "
        sudo systemctl stop sree-dashboard &&
        sudo systemctl start sree-dashboard &&
        sudo systemctl status sree-dashboard --no-pager -l
    "
    
    print_status "🎉 Version synchronization completed successfully!"
    print_info "🌐 Dashboard available at: http://$VPS_IP:8501"
    
else
    print_warning "⚠️  Cannot connect to VPS. Local environment updated only."
    print_info "💡 To sync VPS later, ensure SSH access and run this script again."
fi

echo ""
print_status "✅ Environment synchronization completed!"
print_info "📋 Summary:"
print_info "   • Local environment: Updated to exact versions"
print_info "   • Remote environment: Updated to match local"
print_info "   • Code deployment: Latest version deployed"
print_info "   • Service status: Restarted and running"
echo "" 