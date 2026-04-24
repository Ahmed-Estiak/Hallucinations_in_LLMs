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

_openai_client = None
_gemini_handle = None


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
- Prefer the provided knowledge graph context when it is directly relevant and temporally appropriate for the question.
- If the provided context appears insufficient, not truly relevant, or temporally too distant for the question, use your best knowledge instead.
- If the context and your background knowledge appear to conflict, prefer the answer that is better supported by question-relevant evidence.
- Do not invent unsupported facts or add extra details beyond the answer.
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


def _load_api_keys() -> tuple[str, str]:
    """Load API keys lazily so imports do not instantiate clients."""
    return OPENAI_API_KEY, GEMINI_API_KEY


def _get_openai_client():
    """Get a lazily initialized OpenAI client."""
    global _openai_client
    if _openai_client is None:
        openai_api_key, _ = _load_api_keys()
        _openai_client = OpenAI(api_key=openai_api_key)
    return _openai_client


def _get_gemini_handle():
    """Get a lazily initialized Gemini SDK handle."""
    global _gemini_handle
    if _gemini_handle is not None:
        return _gemini_handle

    _, gemini_api_key = _load_api_keys()
    if _GEMINI_SDK == "google-genai":
        _gemini_handle = google_genai.Client(api_key=gemini_api_key)
    else:
        legacy_genai.configure(api_key=gemini_api_key)
        _gemini_handle = legacy_genai.GenerativeModel("gemini-2.5-flash")
    return _gemini_handle


def _generate_with_gemini(prompt: str) -> str:
    """Generate text through Gemini behind one unified wrapper."""
    gemini_handle = _get_gemini_handle()
    if _GEMINI_SDK == "google-genai":
        resp = gemini_handle.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
    else:
        resp = gemini_handle.generate_content(prompt)
    return resp.text.strip()


def _ask_with_kg(model_name: str, question: str, kg_facts_text: str, time_constraint: str = "") -> str:
    """Shared KG-grounded prompt flow for one model family."""
    prompt = _build_kg_prompt(question, kg_facts_text, time_constraint)
    if model_name == "openai":
        resp = _get_openai_client().responses.create(
            model="gpt-5-mini",
            input=prompt
        )
        return resp.output_text.strip()
    return _generate_with_gemini(prompt)


def ask_openai_with_kg(question: str, kg_facts_text: str, time_constraint: str = "") -> str:
    """
    Call OpenAI with a KG-grounded prompt.
    
    Args:
        question: The question to answer
        kg_facts_text: Formatted KG facts string (from kg_retriever.format_facts_for_prompt)
        time_constraint: Time constraint from parsed question
    
    Returns:
        answer string
    """
    return _ask_with_kg("openai", question, kg_facts_text, time_constraint)


def ask_gemini_with_kg(question: str, kg_facts_text: str, time_constraint: str = "") -> str:
    """
    Call Gemini with a KG-grounded prompt.
    
    Args:
        question: The question to answer
        kg_facts_text: Formatted KG facts string (from kg_retriever.format_facts_for_prompt)
        time_constraint: Time constraint from parsed question
    
    Returns:
        answer string
    """
    return _ask_with_kg("gemini", question, kg_facts_text, time_constraint)
