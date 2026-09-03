#!/usr/bin/env bash
# ==============================================================================
# Development Runner: Starts FastAPI Backend and Streamlit UI Concurrently
# ==============================================================================
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"

echo "========================================================================"
echo " 🚀 Launching Vision OCR & Medical Document Platform (Dev Mode)"
echo "========================================================================"
echo "  📡 API Server: http://localhost:8200 (Docs: http://localhost:8200/ocr/docs)"
echo "  🖥️ UI Dashboard: http://localhost:8501"
echo "========================================================================"

# Trap SIGINT and SIGTERM to kill child processes cleanly
trap 'kill $(jobs -p) 2>/dev/null || true' EXIT SIGINT SIGTERM

mkdir -p storage

# Start FastAPI in background
uvicorn app.main:app --host 0.0.0.0 --port 8200 --reload &
API_PID=$!

# Wait for API to be responsive
sleep 2

# Start Streamlit UI
streamlit run ui/streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &
UI_PID=$!

wait $API_PID $UI_PID
