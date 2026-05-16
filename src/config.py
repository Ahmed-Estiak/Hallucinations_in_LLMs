import os
from dotenv import load_dotenv

# Load local environment variables before reading provider credentials.
# This lets developers keep API keys in a .env file during local runs.
load_dotenv()

# API keys are read once at import time and reused by modules that import config.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Keep startup non-blocking when keys are missing; downstream code can decide
# whether a missing provider credential should fail a specific operation.
if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found")

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found")
