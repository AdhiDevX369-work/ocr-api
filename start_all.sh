#!/bin/bash
echo "=========================================================="
echo "Starting OCR Vision Studio"
echo "=========================================================="

# Check if Ollama is running
if ! curl -s http://localhost:11434/ > /dev/null; then
    echo "Warning: Ollama is not running on http://localhost:11434"
    echo "Please start Ollama service first."
    exit 1
fi

echo "Ollama Server detected on port 11434."
echo "Starting FastAPI Vision API on port 8200..."
echo "Starting Streamlit Studio UI on port 8600..."
echo "=========================================================="

export PYTHONPATH=.

# Function to handle shutdown on CTRL+C
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $API_PID $UI_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start FastAPI API in background
python3 app/main.py &
API_PID=$!

# Wait for API to initialize
sleep 2

# Start Streamlit UI
streamlit run ui/app.py --server.port 8600 --server.address 0.0.0.0 2>/dev/null || streamlit run ui/streamlit_app.py --server.port 8600 --server.address 0.0.0.0 &
UI_PID=$!

echo "Services started successfully."
echo "   - API Health: http://localhost:8200/health (or http://localhost:8200/ocr/health)"
echo "   - Streamlit UI: http://localhost:8600"
echo "Press Ctrl+C to stop all services."

wait
