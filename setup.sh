#!/bin/bash
# Create Streamlit config directory
mkdir -p ~/.streamlit/

# Create Streamlit config file
cat > ~/.streamlit/config.toml <<EOF
[server]
headless = true
port = \$PORT
enableCORS = false
enableXsrfProtection = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
EOF


