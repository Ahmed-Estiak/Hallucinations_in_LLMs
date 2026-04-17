from openai import OpenAI
import warnings

try:
    from google import genai as google_genai
    _GEMINI_SDK = "google-genai"
except ImportError:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        import google.generativeai as legacy_genai
    google_genai = None
    _GEMINI_SDK = "google-generativeai"

try:
    from src.config import OPENAI_API_KEY, GEMINI_API_KEY
except ModuleNotFoundError:
    from config import OPENAI_API_KEY, GEMINI_API_KEY


openai_client = OpenAI(api_key=OPENAI_API_KEY)
if _GEMINI_SDK == "google-genai":
    gemini_client = google_genai.Client(api_key=GEMINI_API_KEY)
else:
    legacy_genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = legacy_genai.GenerativeModel("gemini-2.5-flash")


PROMPT_TEMPLATE = """
Answer the astronomy question using the most appropriate format from the list below.
Rules:
- Output only the final answer.
- Do not explain.
- Do not add extra words.
- Do not add labels.
- Do not add quotation marks.
- Do not add bullet points.
- Choose the format that best matches the question.

Format options:
- single_number: only the number
- boolean: yes or no
- entity: only the entity name
- entity_list: comma-separated names
- ordered_list: comma-separated names in the correct order
- multi_field: comma-separated values in the most natural required order

If the question does not fit any format exactly, return the shortest answer that best matches the closest format above.

Question:
{question}

Answer:
"""


def ask_openai(question):
    """Ask OpenAI with vanilla prompt (no KG facts)."""
    prompt = PROMPT_TEMPLATE.format(question=question)

    resp = openai_client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )

    return resp.output_text.strip()


def ask_gemini(question):
    """Ask Gemini with vanilla prompt (no KG facts)."""
    prompt = PROMPT_TEMPLATE.format(question=question)
    if _GEMINI_SDK == "google-genai":
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
    else:
        resp = gemini_model.generate_content(prompt)
    return resp.text.strip()


if __name__ == "__main__":
    print(f"Loaded src.models successfully using {_GEMINI_SDK}. Import this module and call ask_openai() or ask_gemini().")
