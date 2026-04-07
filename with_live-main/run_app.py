#!/usr/bin/env python3
"""
Quick launcher for Night Vision Seekers Gradio App
"""

import subprocess
import sys
import os
import webbrowser
import time

def main():
    print("🌙 Night Vision Seekers - Gradio Frontend")
    print("=" * 50)
    print("")
    
    # Get the directory of this script
    app_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(app_dir)
    
    # Path to venv
    venv_python = os.path.join(app_dir, "venv", "Scripts", "python.exe")
    # venv_python = os.path.join(app_dir, "venv", "bin", "python3")
    
    if not os.path.exists(venv_python):
        print("❌ Virtual environment not found!")
        print("Please ensure venv is created and dependencies are installed.")
        sys.exit(1)
    
    print("✅ Virtual environment found")
    print("🚀 Starting Gradio application...")
    print("")
    print("📍 Open your browser to: http://localhost:7860")
    print("Press Ctrl+C to stop the server")
    print("")
    
    # Start the app
    try:
        # Open browser after a short delay
        time.sleep(2)
        webbrowser.open("http://localhost:7860")
    except:
        pass
    
    # Run the app
    subprocess.run([venv_python, "app.py"])

if __name__ == "__main__":
    main()
