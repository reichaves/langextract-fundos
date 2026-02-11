#!/usr/bin/env python3
"""Quick test to verify API key configuration."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import load_config

load_config()

key = os.getenv("LANGEXTRACT_API_KEY") or os.getenv("GOOGLE_API_KEY")
if key:
    masked = key[:8] + "..." + key[-4:]
    print(f"✅ API key found: {masked}")
else:
    print("❌ No API key found. Create a .env file with:")
    print("   GOOGLE_API_KEY=your-key-here")
    print("   Get a key at: https://aistudio.google.com/app/apikey")
