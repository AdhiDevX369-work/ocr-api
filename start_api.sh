#!/bin/bash
echo "🚀 Starting Production Image-Based Vision Chat API on Port 8200 (Conda Env: stt)..."
conda run -n stt python3 app/main.py
