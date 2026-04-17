import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found")

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not found")
