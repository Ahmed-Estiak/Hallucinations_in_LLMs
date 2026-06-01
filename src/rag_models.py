"""LLM calls with retrieved RAG context."""

from __future__ import annotations

from src.models import _generate_with_gemini, _get_openai_client


PROMPT_TEMPLATE_WITH_RAG = """
Answer the astronomy question using the retrieved source context.

Rules:
- Use the retrieved context as the primary evidence.
- Combine evidence across sources when the answer requires filtering, comparison, or ordering.
- Use generated structured fact/table entries together with the raw retrieved chunks. If they refer to the same claim, treat the raw chunk as supporting evidence for the structured entry.
- If the question asks for the earliest, first, largest, smallest, greater, fewer, or similar relation, reason over the relevant entities in the context before answering.
- For date-sensitive count questions, build a timeline only from dated count
  claims present in the retrieved context, including structured/resolved facts
  and raw chunks. If raw chunks provide an additional dated count anchor between
  two structured anchors, insert it into the timeline. Use the latest anchor on
  or before the requested date only when a later dated anchor for the same body
  and count type is also present in the context. If no later dated anchor is
  present, do not infer that the older count remained valid.
- If multiple count claims exist for the same body at the same date/time level,
  prefer the highest count unless the context clearly marks the higher value as
  speculative, pending, unconfirmed, or a different count type.
- Do not treat current counts or later dated counts as support for an earlier requested date.
- For a present moon-count question, use the resolved direct-current or validated
  current-table fact supplied in context; do not substitute other planets' counts.
- For a non-temporal moon-count comparison/list question, treat a "Highest structured
  moon-count claims" table as the highest structured count found in the provided
  sources. If a retrieved chunk states a higher count for the same body, prefer
  the higher chunk count. Do not replace a table count with a lower chunk count
  unless the question asks about a specific past time.
- If the retrieved context is insufficient, answer exactly: insufficient context
- Do not invent unsupported facts.
- Output only the final answer.
- For list/entity-list answers, output only comma-separated entity names. Do not
  include counts, numbers, parenthetical details, labels, or explanations.
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
