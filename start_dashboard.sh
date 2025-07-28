#!/bin/bash

# Start SREE Dashboard Locally
echo "🚀 Starting SREE Dashboard Locally"
echo "=================================================="

# Check if required files exist
echo "📋 Checking required files..."
if [ -f "dashboard.py" ]; then
    echo "   ✅ dashboard.py"
else
    echo "   ❌ dashboard.py - NOT FOUND!"
    exit 1
fi

if [ -f "block_creation_system.py" ]; then
    echo "   ✅ block_creation_system.py"
else
    echo "   ❌ block_creation_system.py - NOT FOUND!"
    exit 1
fi

if [ -f "heart_disease_dataset_new.csv" ]; then
    echo "   ✅ heart_disease_dataset_new.csv"
else
    echo "   ❌ heart_disease_dataset_new.csv - NOT FOUND!"
    exit 1
fi

echo ""
echo "🌐 Dashboard will be available at:"
echo "   http://localhost:8501"
echo ""
echo "📋 Instructions:"
echo "   1. The dashboard will open automatically in your browser"
echo "   2. Upload the 'heart_disease_dataset_new.csv' file"
echo "   3. Click 'Run SREE Analysis'"
echo "   4. Verify that Accuracy shows ≥ 0.95"
echo "   5. Press Ctrl+C to stop the dashboard"
echo ""
echo "🚀 Starting dashboard..."
echo "   (This may take a few seconds)"

# Start streamlit
streamlit run dashboard.py \
    --server.port 8501 \
    --server.address localhost \
    --server.headless false \
    --browser.gatherUsageStats false 