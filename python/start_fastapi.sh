#!/bin/bash
# FastAPI Startup Script for Linux/Mac
cd "$(dirname "$0")"
python3 -m uvicorn api:app --host 0.0.0.0 --port 8000 --reload

