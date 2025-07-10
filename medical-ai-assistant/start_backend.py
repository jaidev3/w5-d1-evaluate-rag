#!/usr/bin/env python3
"""
Startup script for Medical AI Assistant Backend
Handles the asyncio event loop configuration to avoid conflicts with RAGAS/nest_asyncio
"""

import os
import sys
import asyncio
import uvicorn

# Add the app directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Main startup function."""
    print("🚀 Starting Medical AI Assistant Backend...")
    print("📋 Using standard asyncio event loop (avoiding uvloop conflict)")
    
    # Configure asyncio policy for compatibility
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    # Start the server with standard asyncio
    uvicorn.run(
        "app.main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True,
        loop="asyncio"  # Use standard asyncio instead of uvloop
    )

if __name__ == "__main__":
    main() 