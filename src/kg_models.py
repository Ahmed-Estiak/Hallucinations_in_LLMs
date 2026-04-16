"""
KG Models: LLM calls with KG grounding
"""
from openai import OpenAI
import google.generativeai as genai

from src.config import OPENAI_API_KEY, GEMINI_API_KEY


openai_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")


# KG-grounded prompt with sophisticated conflict resolution
PROMPT_TEMPLATE_WITH_KG = """
Answer the astronomy question using the facts provided below. These facts are PRIMARY REFERENCE.

CONFLICT RESOLUTION STRATEGY:
1. Use provided facts as your PRIMARY answer source.
2. If a fact conflicts with your knowledge:
   - Check if BOTH the fact AND your hunch exist in your training data
   - If both exist: TRUST THE PROVIDED FACT (it's verified)
   - If only one exists: Use that one
   - If neither exist clearly: Use the fact but note uncertainty
3. For TIME CONSTRAINTS:
   - If the question specifies a date/year/time, ONLY use facts matching that time
   - If a fact's time doesn't match the question's constraint, IGNORE it
   - If no specific time in question, use the MOST RECENT fact only
4. If NO relevant facts are provided, answer based solely on your knowledge.
5. Output ONLY the final answer in appropriate format (number/yes/no/entity/list).
6. NO explanations, NO extra words, NO labels or quotes.

Available Facts from Knowledge Graph (verified sources):
{kg_facts}

Time Constraint from Question: {time_constraint}

Question: {question}

Answer (ONLY the final answer, nothing else):
"""


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
    prompt = PROMPT_TEMPLATE_WITH_KG.format(
        kg_facts=kg_facts_text,
        question=question,
        time_constraint=time_constraint if time_constraint else "No specific time constraint"
    )
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
    prompt = PROMPT_TEMPLATE_WITH_KG.format(
        kg_facts=kg_facts_text,
        question=question,
        time_constraint=time_constraint if time_constraint else "No specific time constraint"
    )
    resp = gemini_model.generate_content(prompt)
    return resp.text.strip()
