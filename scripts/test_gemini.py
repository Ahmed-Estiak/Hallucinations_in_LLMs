"""
Smoke test for checking Gemini API access from the local environment.

The top block keeps an older SDK example commented out for reference. The active
code below uses the newer `google.genai` client to send a minimal prompt and
print the response.
"""

# Older Gemini SDK example kept for reference. It uses the
# `google.generativeai` package style instead of the newer `google.genai`
# client used below.
# import os
# from dotenv import load_dotenv
# import google.generativeai as genai

# load_dotenv()

# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# model = genai.GenerativeModel("gemini-1.5-flash")

# response = model.generate_content("Say hello in one word.")

# print(response.text)

import os
from dotenv import load_dotenv
from google import genai


# Load environment variables from .env, including GEMINI_API_KEY.
load_dotenv()

# Create a Gemini client with the API key from the local environment.
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Send a short prompt to verify authentication, model access, and basic content
# generation without producing a long response.
resp = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello in one word."
)

# Print the text returned by the Gemini model.
print(resp.text)
