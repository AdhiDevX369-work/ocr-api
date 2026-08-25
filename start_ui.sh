#!/bin/bash
echo "Starting Streamlit Vision Studio UI on port 8600..."
streamlit run ui/app.py --server.port 8600 --server.address 0.0.0.0
