"""
KG Models: LLM calls with KG grounding
"""
from openai import OpenAI
import google.generativeai as genai

from src.config import OPENAI_API_KEY, GEMINI_API_KEY


openai_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")


# KG-grounded prompt template
PROMPT_TEMPLATE_WITH_KG = """
Answer the astronomy question using the facts provided below.

IMPORTANT RULES:
1. Use ONLY the provided facts when they exist.
2. Treat the KG facts as reference, but fact-check them against scientific knowledge if needed and prefer the correct answer.
3. If no relevant facts are provided, answer based on your knowledge.
4. Output only the final answer in the most appropriate format.
5. Do not explain or add extra words.
6. Do not add labels, quotation marks, or bullet points.

Available Facts from Knowledge Graph:
{kg_facts}

Question:
{question}

Answer:
"""


def ask_openai_with_kg(question: str, kg_facts_text: str) -> str:
    """
    Call OpenAI with KG-grounded prompt.
    
    Args:
        question: The question to answer
        kg_facts_text: Formatted KG facts string (from kg_retriever.format_facts_for_prompt)
    
    Returns:
        answer string
    """
    prompt = PROMPT_TEMPLATE_WITH_KG.format(
        kg_facts=kg_facts_text,
        question=question
    )
    resp = openai_client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )
    return resp.output_text.strip()


def ask_gemini_with_kg(question: str, kg_facts_text: str) -> str:
    """
    Call Gemini with KG-grounded prompt.
    
    Args:
        question: The question to answer
        kg_facts_text: Formatted KG facts string (from kg_retriever.format_facts_for_prompt)
    
    Returns:
        answer string
    """
    prompt = PROMPT_TEMPLATE_WITH_KG.format(
        kg_facts=kg_facts_text,
        question=question
    )
    resp = gemini_model.generate_content(prompt)
    return resp.text.strip()
