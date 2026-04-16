"""
KG Models: LLM calls with KG grounding
"""
from openai import OpenAI
import google.generativeai as genai

from src.config import OPENAI_API_KEY, GEMINI_API_KEY


openai_client = OpenAI(api_key=OPENAI_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")


# Original prompt template (without KG)
PROMPT_TEMPLATE_VANILLA = """
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


# KG-grounded prompt template
PROMPT_TEMPLATE_WITH_KG = """
Answer the astronomy question using the facts provided below.

IMPORTANT RULES:
1. Use ONLY the provided facts to answer.
2. If a fact appears incorrect, you may fact-check it against scientific knowledge and provide the correct answer.
3. If no relevant facts are provided, answer based on your knowledge but acknowledge the lack of supporting evidence.
4. Output only the final answer in the most appropriate format.
5. Do not explain or add extra words.
6. Do not add labels, quotation marks, or bullet points.

Available Facts from Knowledge Graph:
{kg_facts}

Question:
{question}

Answer:
"""


def ask_openai_vanilla(question: str) -> str:
    """
    Call OpenAI with vanilla prompt (no KG).
    Returns: answer string
    """
    prompt = PROMPT_TEMPLATE_VANILLA.format(question=question)
    resp = openai_client.responses.create(
        model="gpt-5-mini",
        input=prompt
    )
    return resp.output_text.strip()


def ask_gemini_vanilla(question: str) -> str:
    """
    Call Gemini with vanilla prompt (no KG).
    Returns: answer string
    """
    prompt = PROMPT_TEMPLATE_VANILLA.format(question=question)
    resp = gemini_model.generate_content(prompt)
    return resp.text.strip()


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
