# Vercel Serverless Entry Point
import sys
import os

# Add parent directory to path so we can import from backend
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from main import app

# Export the FastAPI app for Vercel
handler = app