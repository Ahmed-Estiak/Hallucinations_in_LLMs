"""LLM calls with retrieved RAG context."""

from __future__ import annotations

from src.models import _generate_with_gemini, _get_openai_client


PROMPT_TEMPLATE_WITH_RAG = """
Answer the astronomy question using the retrieved source context.

Rules:
- Use the retrieved context as the primary evidence.
- Combine evidence across sources when the answer requires filtering, comparison, or ordering.
- If the question asks for the earliest, first, largest, smallest, greater, fewer, or similar relation, reason over the relevant entities in the context before answering.
- For date-sensitive count questions, use a count only if it directly matches the requested date
  or the retrieved context explicitly provides a resolved validity interval for that date.
- Do not create your own validity intervals from dates in the context.
- Do not treat current counts or later dated counts as support for an earlier requested date.
- If temporal evidence is labelled conflicting or insufficient, answer exactly: insufficient context
- For a present moon-count question, use the resolved direct-current or validated
  current-table fact supplied in context; do not substitute other planets' counts.
- If the retrieved context is insufficient, answer exactly: insufficient context
- Do not invent unsupported facts.
- Output only the final answer.
- Do not explain.
- Do not add labels, quotation marks, bullets, or extra words.

Retrieved Source Context:
{rag_context}

Question: {question}

Answer:
"""


def build_rag_prompt(question: str, rag_context: str) -> str:
    return PROMPT_TEMPLATE_WITH_RAG.format(
        question=question,
        rag_context=rag_context,
    )


def ask_openai_with_rag(question: str, rag_context: str) -> str:
    prompt = build_rag_prompt(question, rag_context)
    resp = _get_openai_client().responses.create(
        model="gpt-5-mini",
        input=prompt,
    )
    return resp.output_text.strip()


def ask_gemini_with_rag(question: str, rag_context: str) -> str:
    prompt = build_rag_prompt(question, rag_context)
    return _generate_with_gemini(prompt)
