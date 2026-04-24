"""
KG Models: LLM calls with KG grounding
"""
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


# KG-grounded prompt aligned with deterministic KG selection
PROMPT_TEMPLATE_WITH_KG = """
Answer the astronomy question using the provided knowledge graph context.

Rules:
- Use only the context that is directly relevant to the question.
- Ignore extra information that does not help answer the question.
- If the question has a time condition, prefer the fact that matches that condition:
  - as of / by: use the latest valid fact at or before that time
  - before: use the latest fact before that time
  - after: use the earliest fact after that time
- If the context is insufficient, use your best knowledge but do not invent extra unsupported details.
- Output only the final answer.
- Do not explain.
- Do not add labels, quotation marks, or extra words.

Knowledge Graph Context:
{kg_facts}
{time_block}

Question: {question}

Answer:
"""


def _build_kg_prompt(question: str, kg_facts_text: str, time_constraint: str = "") -> str:
    """Build the KG-grounded prompt, omitting the time block when absent."""
    time_block = ""
    if time_constraint:
        time_block = f"\nTime Constraint from Question:\n{time_constraint}\n"
    return PROMPT_TEMPLATE_WITH_KG.format(
        kg_facts=kg_facts_text,
        question=question,
        time_block=time_block,
    )


def ask_openai_with_kg(question: str, kg_facts_text: str, time_constraint: str = "") -> str:
    """
    Call OpenAI with KG-grounded prompt and sophisticated conflict resolution.
    
    Args:
        question: The question to answer
        kg_facts_text: Formatted KG facts string (from kg_retriever.format_facts_for_prompt)
        time_constraint: Time constraint from parsed question
    
    Returns:
        answer string
    """
    prompt = _build_kg_prompt(question, kg_facts_text, time_constraint)
    resp = openai_client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )
    return resp.output_text.strip()


def ask_gemini_with_kg(question: str, kg_facts_text: str, time_constraint: str = "") -> str:
    """
    Call Gemini with KG-grounded prompt and sophisticated conflict resolution.
    
    Args:
        question: The question to answer
        kg_facts_text: Formatted KG facts string (from kg_retriever.format_facts_for_prompt)
        time_constraint: Time constraint from parsed question
    
    Returns:
        answer string
    """
    prompt = _build_kg_prompt(question, kg_facts_text, time_constraint)
    if _GEMINI_SDK == "google-genai":
        resp = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
    else:
        resp = gemini_model.generate_content(prompt)
    return resp.text.strip()
