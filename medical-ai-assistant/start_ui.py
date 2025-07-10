#!/usr/bin/env python3
"""
Startup script for Medical AI Assistant UI
Handles the Python path configuration for proper imports
"""

import os
import sys
import subprocess

def main():
    """Main startup function."""
    print("🚀 Starting Medical AI Assistant UI...")
    
    # Add the current directory to Python path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Change to the UI directory
    ui_dir = os.path.join(current_dir, "ui")
    os.chdir(ui_dir)
    
    # Start Streamlit
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.port", "8501",
        "--server.address", "localhost"
    ]
    
    print(f"📋 Running command: {' '.join(cmd)}")
    print(f"📂 Working directory: {ui_dir}")
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down Medical AI Assistant UI")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 