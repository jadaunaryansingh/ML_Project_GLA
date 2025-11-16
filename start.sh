#!/bin/bash
# Start script for Render deployment

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Run setup script if it exists
if [ -f setup.sh ]; then
    chmod +x setup.sh
    ./setup.sh
fi

# Start Streamlit with proper configuration
exec streamlit run project.py \
    --server.port=$PORT \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false \
    --browser.gatherUsageStats=false


