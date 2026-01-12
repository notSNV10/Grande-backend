"""Test if all imports work"""
import sys

print("Testing imports...")
try:
    from fastapi import FastAPI
    print("✓ FastAPI imported")
except Exception as e:
    print(f"✗ FastAPI import failed: {e}")
    sys.exit(1)

try:
    from ai_agent import AIAgent
    print("✓ AIAgent imported")
except Exception as e:
    print(f"✗ AIAgent import failed: {e}")
    sys.exit(1)

try:
    from forecast import create_lstm_model
    print("✓ forecast imported")
except Exception as e:
    print(f"✗ forecast import failed: {e}")
    sys.exit(1)

try:
    import mysql.connector
    print("✓ mysql.connector imported")
except Exception as e:
    print(f"✗ mysql.connector import failed: {e}")
    sys.exit(1)

print("\nAll imports successful!")
