"""
Smoke test for checking OpenAI API access from the local environment.

The script loads an API key from .env, sends a tiny prompt through the OpenAI
Responses API, and prints the returned text.
"""
from openai import OpenAI
import os
from dotenv import load_dotenv


# Load environment variables from .env, including OPENAI_API_KEY.
load_dotenv()

# Create an OpenAI client using the API key from the local environment.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Send a minimal prompt to verify authentication, model access, and the
# Responses API call without producing a long response.
response = client.responses.create(
    model="gpt-5-mini",
    input="Say hello in one word.",
)

# Print the text content returned by the model.
print(response.output_text)
