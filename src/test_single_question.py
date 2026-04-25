"""
Run one benchmark question through both vanilla and KG-grounded paths.

Default question ID is 5 to make targeted debugging cheap in time and tokens.
"""
import argparse
import json
import sys
from pathlib import Path

# Add workspace root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ask_openai, ask_gemini
from src.kg_runner import (
    _prepare_question_context,
    _has_comprehensive_kg_context,
    _should_use_kg_context,
)
from src.kg_retriever import KGRetriever
from src.kg_reasoning_engine import KGReasoningEngine
from src.question_classifier import QuestionClassifier
from src.kg_models import ask_openai_with_kg, ask_gemini_with_kg


def _load_question(question_id: int) -> dict:
    with open("data/qa_92.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    for question in questions:
        if question.get("id") == question_id:
            return question

    raise ValueError(f"Question ID {question_id} not found in data/qa_92.json")
def main() -> None:
    parser = argparse.ArgumentParser(description="Run one question through vanilla and KG paths.")
    parser.add_argument("--id", type=int, default=5, help="Question ID from data/qa_92.json (default: 5)")
    args = parser.parse_args()

    question_row = _load_question(args.id)
    question = question_row["question"]

    print(f"Question ID: {args.id}")
    print(f"Question: {question}")
    print("=" * 80)

    question_classifier = QuestionClassifier()
    kg_retriever = KGRetriever()
    kg_reasoning_engine = KGReasoningEngine()

    context = _prepare_question_context(question, question_classifier, kg_retriever, kg_reasoning_engine)
    classified_q = context["classified_q"]
    should_use_kg = _should_use_kg_context(classified_q)
    kg_prompt_used = should_use_kg and context["kg_found"] and _has_comprehensive_kg_context(context)

    print("Parsed entities:", context["entities"])
    print("Parsed predicates:", context["predicates"])
    print("Time constraint:", context["time_constraint"])
    print("Primary type:", classified_q.primary_type.name)
    print("Logical modifiers:", [modifier.name for modifier in classified_q.logical_modifiers])
    print("Reasoning strategy:", context["reasoning_strategy"])
    print("KG found:", context["kg_found"])
    print("KG prompt used:", kg_prompt_used)
    print("-" * 80)
    print(context["kg_facts_text"])
    print("-" * 80)

    print("Running vanilla answers...")
    openai_vanilla = ask_openai(question)
    gemini_vanilla = ask_gemini(question)

    print("Running KG-grounded answers...")
    if kg_prompt_used:
        openai_kg = ask_openai_with_kg(question, context["kg_facts_text"], context["time_constraint"])
        gemini_kg = ask_gemini_with_kg(question, context["kg_facts_text"], context["time_constraint"])
    else:
        openai_kg = openai_vanilla
        gemini_kg = gemini_vanilla

    print("Vanilla OpenAI:", openai_vanilla)
    print("Vanilla Gemini:", gemini_vanilla)
    print("KG OpenAI:", openai_kg)
    print("KG Gemini:", gemini_kg)


if __name__ == "__main__":
    main()
