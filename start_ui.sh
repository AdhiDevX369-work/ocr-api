#!/bin/bash
echo "🎨 Starting Streamlit Vision Studio UI on Port 8600..."
export PYTHONPATH=.
streamlit run ui/streamlit_app.py --server.port 8600 --server.address 0.0.0.0
