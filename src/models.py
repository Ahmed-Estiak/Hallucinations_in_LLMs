import re

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


MULTI_FIELD_SPLIT_PROMPT = """
You are splitting one astronomy question into exactly {expected_parts} standalone sub-questions.

Rules:
- Preserve the original meaning exactly.
- Resolve pronouns and shared context explicitly.
- Keep the original entity names.
- Keep any time constraint in every sub-question where it applies.
- Do not answer the question.
- Do not add explanations.
- Output exactly this format:
Question 1: ...
Question 2: ...

Original question:
{question}
"""


def _parse_split_questions(text: str, expected_parts: int) -> list[str]:
    """Parse strict Question 1 / Question 2 splitter output."""
    matches = re.findall(r"Question\s+(\d+):\s*(.+?)(?=(?:\nQuestion\s+\d+:)|\Z)", text, flags=re.IGNORECASE | re.DOTALL)
    parsed = {}
    for number, content in matches:
        try:
            index = int(number)
        except ValueError:
            continue
        parsed[index] = " ".join(content.strip().split())

    questions = [parsed[i] for i in range(1, expected_parts + 1) if i in parsed]
    if len(questions) == expected_parts:
        return questions
    return []


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


def split_multifield_question(question: str, expected_parts: int = 2) -> list[str]:
    """Ask OpenAI to split one multi-field question into standalone sub-questions."""
    prompt = MULTI_FIELD_SPLIT_PROMPT.format(
        question=question,
        expected_parts=expected_parts,
    )
    resp = openai_client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )
    text = resp.output_text.strip()
    return _parse_split_questions(text, expected_parts)


if __name__ == "__main__":
    print(f"Loaded src.models successfully using {_GEMINI_SDK}. Import this module and call ask_openai() or ask_gemini().")
