"""
Debug the reasoning path for benchmark question Q11.

This script prints parser output, classifier output, reasoning-engine output,
and a small raw-KG inspection for the entities expected in the ground truth.
"""
import json
import sys
from pathlib import Path


# Add the repository root to Python's import path so this file can be run
# directly with `python scripts/debug_q11.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.kg_reasoning_engine import KGReasoningEngine
from src.question_classifier import QuestionClassifier
from src.question_parser import parse_question


QUESTIONS_PATH = "data/qa_92.json"
KG_PATH = "data/astronomy_kg1.json"
QUESTION_ID = 11
GROUND_TRUTH_ENTITIES = ["Mars", "Uranus", "Neptune"]
SEPARATOR_WIDTH = 80
RAW_FACT_PREVIEW_LIMIT = 3


def load_json(path):
    """
    Load and return JSON data from the given file path.

    Both the benchmark question file and raw KG file are JSON, so this helper
    keeps file loading consistent across the script.
    """
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_question_by_id(questions, qid):
    """
    Return the benchmark question record for the requested id.

    The debug script is focused on one question, but a helper makes it clear
    that we are selecting a single record from the full benchmark list.
    """
    return next(q for q in questions if q["id"] == qid)


def print_question_header(question_record):
    """
    Print the selected question and its expected answer.

    This gives context before showing parser, classifier, and reasoning details.
    """
    question = question_record["question"]
    ground_truth = question_record["answer_spec"]["value"]

    print("=" * SEPARATOR_WIDTH)
    print(f"Q{question_record['id']}: {question}")
    print(f"Ground Truth: {ground_truth}")
    print("=" * SEPARATOR_WIDTH)


def parse_and_print_question(question):
    """
    Parse the question and print the extracted entities and predicates.

    These extracted signals are passed into the reasoning engine, so printing
    them helps diagnose whether problems start in parsing or later reasoning.
    """
    parsed = parse_question(question)
    entities = parsed["entities"]
    predicates = parsed["predicates"]

    print(f"\nParsed entities: {entities}")
    print(f"Parsed predicates: {predicates}")

    return entities, predicates


def classify_and_print_question(classifier, question):
    """
    Classify the question and print both primary and secondary type labels.

    The reasoning engine depends on this classification, so this output confirms
    whether Q11 is being routed through the expected reasoning strategy.
    """
    classified = classifier.classify(question)

    print(f"\nClassified as: {classified.primary_type.name}")
    print(f"Secondary types: {[t.name for t in classified.secondary_types]}")

    return classified


def reason_and_print_facts(reasoning_engine, classified, entities, predicates):
    """
    Run the KG reasoning engine and print the selected strategy and facts.

    The returned facts show what evidence the engine believes is relevant for
    answering Q11 after parsing and classification are complete.
    """
    reasoned_facts, strategy = reasoning_engine.reason(
        classified,
        entities,
        predicates,
    )

    print(f"\nReasoning strategy: {strategy}")
    print(f"Reasoned facts returned: {len(reasoned_facts)}")

    if reasoned_facts:
        print("\nReturned Facts:")
        for index, fact in enumerate(reasoned_facts, 1):
            print(f"  {index}. {fact}")


def print_raw_kg_header():
    """
    Print a section header before inspecting raw KG records.

    This separates reasoning-engine output from the lower-level KG fact check.
    """
    print("\n" + "=" * SEPARATOR_WIDTH)
    print("Checking raw KG for entity filtering details")
    print("=" * SEPARATOR_WIDTH)


def print_raw_facts_for_entities(kg, entity_names):
    """
    Print a short preview of raw KG facts for the expected answer entities.

    This confirms whether Mars, Uranus, and Neptune have relevant facts in the
    KG even if the reasoning engine does not return them for Q11.
    """
    print("\nFacts in KG for 'Mars', 'Uranus', 'Neptune':")

    for entity_name in entity_names:
        matching = [
            fact
            for fact in kg
            if fact.get("subject", "").lower() == entity_name.lower()
        ]

        print(f"\n{entity_name}: {len(matching)} facts")
        for fact in matching[:RAW_FACT_PREVIEW_LIMIT]:
            print(
                "  - "
                f"{fact.get('subject')} | "
                f"{fact.get('predicate')} | "
                f"{fact.get('object')}"
            )


def debug_question_reasoning(question_record, classifier, reasoning_engine):
    """
    Run parser, classifier, and reasoning-engine diagnostics for one question.

    Keeping this flow in one function makes the debug order explicit and keeps
    top-level script execution limited to loading data and calling helpers.
    """
    question = question_record["question"]

    print_question_header(question_record)
    entities, predicates = parse_and_print_question(question)
    classified = classify_and_print_question(classifier, question)
    reason_and_print_facts(reasoning_engine, classified, entities, predicates)


def main():
    """
    Load required data and run the full Q11 debugging report.

    Execution lives in main() so this module can be imported without immediately
    printing debug output or reading benchmark files.
    """
    classifier = QuestionClassifier()
    reasoning_engine = KGReasoningEngine()

    questions = load_json(QUESTIONS_PATH)
    question_record = find_question_by_id(questions, QUESTION_ID)
    debug_question_reasoning(question_record, classifier, reasoning_engine)

    kg = load_json(KG_PATH)
    print_raw_kg_header()
    print_raw_facts_for_entities(kg, GROUND_TRUTH_ENTITIES)


if __name__ == "__main__":
    main()
