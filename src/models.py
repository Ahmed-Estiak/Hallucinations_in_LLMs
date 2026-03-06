from openai import OpenAI
from google import genai
from src.config import OPENAI_API_KEY, GEMINI_API_KEY

openai_client = OpenAI(api_key=OPENAI_API_KEY)

gemini_client = genai.Client(api_key=GEMINI_API_KEY)


PROMPT_TEMPLATE = """
You are answering questions for an astronomy benchmark.

Rules:
- Give the shortest possible answer.
- Do not explain.
- Do not add extra words.

Output format rules:
- number → only the number
- boolean → yes or no
- entity → only the name
- list → comma separated
- ordered list → comma separated in correct order
- multi-field → comma separated

Question:
{question}

Answer:
"""


def ask_openai(question):

    prompt = PROMPT_TEMPLATE.format(question=question)

    resp = openai_client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return resp.output_text.strip()


def ask_gemini(question):

    prompt = PROMPT_TEMPLATE.format(question=question)

    resp = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt
    )

    return resp.text.strip()