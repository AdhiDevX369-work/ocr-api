#!/bin/bash
echo "=========================================================="
echo "🚀 Starting OCR Vision Studio (Ollama + Gemma 4 Backend)"
echo "=========================================================="

# Check if Ollama is running
if ! curl -s http://localhost:11434/ > /dev/null; then
    echo "⚠️ Ollama does not seem to be running on http://localhost:11434!"
    echo "   Please start Ollama first using: ollama serve"
    exit 1
fi

echo "🟢 Local Ollama Server Detected on port 11434."
echo "🚀 Starting FastAPI Vision API on http://localhost:8200 ..."
echo "🎨 Starting Streamlit Studio UI on http://localhost:8600 ..."
echo "=========================================================="

export PYTHONPATH=.

# Function to handle shutdown on CTRL+C
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $API_PID $UI_PID 2>/dev/null
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start FastAPI API in background
conda run -n stt python3 app/main.py &
API_PID=$!

# Wait for API to initialize
sleep 2

# Start Streamlit UI
conda run -n stt streamlit run ui/streamlit_app.py --server.port 8600 --server.address 0.0.0.0 &
UI_PID=$!

echo "✅ App started!"
echo "   - API Health: http://localhost:8200/health"
echo "   - Streamlit UI: http://localhost:8600"
echo "Press Ctrl+C to stop all services."

wait
