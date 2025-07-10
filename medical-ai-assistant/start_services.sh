#!/bin/bash

# Medical AI Assistant - Service Startup Script
echo "🏥 Starting Medical AI Assistant Services..."
echo "============================================="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found. Please run 'python -m venv .venv' first."
    exit 1
fi

# Activate virtual environment
source .venv/bin/activate

# Check if dependencies are installed
echo "📦 Checking dependencies..."
python -c "import fastapi, streamlit, openai, chromadb, ragas" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Dependencies not installed. Please run 'pip install -r requirements.txt' first."
    exit 1
fi

# Create required directories
mkdir -p data/vector_db data/documents data/evaluation logs

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.example .env
    echo "📝 Please edit .env file with your OpenAI API key before starting services."
    echo "   Set OPENAI_API_KEY=your_actual_api_key_here"
    exit 1
fi

# Function to start FastAPI
start_fastapi() {
    echo "🚀 Starting FastAPI server on http://localhost:8000..."
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
    FASTAPI_PID=$!
    echo "   FastAPI PID: $FASTAPI_PID"
}

# Function to start Streamlit
start_streamlit() {
    echo "🎨 Starting Streamlit app on http://localhost:8501..."
    streamlit run ui/app.py --server.port 8501 --server.address 0.0.0.0 &
    STREAMLIT_PID=$!
    echo "   Streamlit PID: $STREAMLIT_PID"
}

# Function to cleanup processes
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    if [ ! -z "$FASTAPI_PID" ]; then
        kill $FASTAPI_PID 2>/dev/null
        echo "   FastAPI stopped"
    fi
    if [ ! -z "$STREAMLIT_PID" ]; then
        kill $STREAMLIT_PID 2>/dev/null
        echo "   Streamlit stopped"
    fi
    echo "✅ All services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Start services
start_fastapi
sleep 2
start_streamlit

echo ""
echo "🎉 Medical AI Assistant is running!"
echo "   - FastAPI API: http://localhost:8000"
echo "   - API Documentation: http://localhost:8000/docs"
echo "   - Streamlit UI: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for services
wait 