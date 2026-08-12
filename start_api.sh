#!/bin/bash
echo "🚀 Starting Production Image-Based Vision Chat API on Port 8200 (Conda Env: rag)..."
conda run -n rag python3 app/main.py
