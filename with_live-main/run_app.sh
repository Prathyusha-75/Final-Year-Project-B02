#!/bin/bash
# Night Vision Seekers - Startup Script

echo "🌙 Night Vision Seekers - Gradio Frontend"
echo "=========================================="
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

echo "✅ Virtual environment activated"
echo ""

# Start the app
echo "🚀 Starting Gradio application..."
echo "📍 Open your browser to: http://localhost:7860"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 app.py
