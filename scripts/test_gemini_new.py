"""
Smoke test for the newer Google Gemini Python SDK client.

The script loads the Gemini API key from the local environment, sends a minimal
prompt, and prints the model response so the SDK setup can be verified quickly.
"""
import os
from dotenv import load_dotenv
from google import genai


# Load environment variables from a .env file, including GEMINI_API_KEY.
load_dotenv()

# Create a Gemini client using the API key from the environment.
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Send a tiny prompt so the test verifies authentication, model access, and the
# generate_content call without producing a long response.
resp = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="Say hello in one word."
)

# Print the model text response returned by Gemini.
print(resp.text)
