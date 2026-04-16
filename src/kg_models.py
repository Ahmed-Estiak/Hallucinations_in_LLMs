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
1. Use the provided facts as reference information.
2. Prioritize accurate scientific knowledge over the facts if they conflict or appear incorrect/outdated.
3. Fact-check the facts against your training knowledge and provide the most correct answer.
4. If no relevant facts are provided, answer based on your knowledge.
5. Output only the final answer in the most appropriate format (e.g., for yes/no questions, answer 'yes' or 'no').
6. Do not explain, add extra words, or include labels/quotation marks.

Available Facts from Knowledge Graph:
{kg_facts}

Question: {question}
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
