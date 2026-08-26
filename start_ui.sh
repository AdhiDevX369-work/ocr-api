#!/bin/bash
echo "Starting Streamlit Medical Vision Studio UI on port 8600..."
streamlit run ui/streamlit_app.py --server.port 8600 --server.address 0.0.0.0
