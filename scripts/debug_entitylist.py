"""
Debug ENTITY_LIST questions to see what KG facts are returned.

This script checks selected benchmark questions and prints parser output,
classifier output, raw KG facts, and the formatted KG prompt context.
"""
import json
import sys
from pathlib import Path


# Make direct execution work from the repository root with:
# `python scripts/debug_entitylist.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.question_classifier import QuestionClassifier
from src.kg_retriever import KGRetriever
from src.question_parser import parse_question


# Load all benchmark questions from the QA dataset.
with open("data/qa_92.json") as f:
    questions = json.load(f)

# Check Q11 and Q12, the ENTITY_LIST/LIST-style questions being debugged here.
for qid in [11, 12]:
    # Find the full question record so we can compare parser/retrieval output
    # against the expected answer.
    q = next(q for q in questions if q["id"] == qid)
    question = q["question"]
    ground_truth = q["answer_spec"]["value"]

    print("\n" + "=" * 80)
    print(f"Q{qid}: {question}")
    print(f"Ground Truth: {ground_truth}")
    print("=" * 80)

    # Parse the question into the signals used for KG retrieval.
    parsed = parse_question(question)
    entities = parsed["entities"]
    predicates = parsed["predicates"]
    time_constraint = parsed["time_constraint"]

    print(f"Parsed entities: {entities}")
    print(f"Parsed predicates: {predicates}")
    print(f"Time constraint: {time_constraint}")

    # Classify the question to confirm which high-level question type the
    # pipeline thinks it is handling.
    classifier = QuestionClassifier()
    classified = classifier.classify(question)
    print(f"\nClassified as: {classified.primary_type.name}")

    # Retrieve matching KG facts using the parsed entities, predicates, and
    # time constraint.
    kg_retriever = KGRetriever()
    facts = kg_retriever.retrieve(entities, predicates, time_constraint, limit=3)

    print(f"\nRetrieved {len(facts)} facts:")
    for f in facts:
        print(
            f"  - {f.get('subject')} | {f.get('predicate')} | "
            f"{f.get('object')} ({f.get('time', 'unknown')})"
        )

    # Show the exact KG text that would be inserted into a model prompt.
    formatted = kg_retriever.format_facts_for_prompt(facts)
    print(f"\nFormatted for prompt:\n{formatted}")
