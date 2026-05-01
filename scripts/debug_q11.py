"""
Debug specific question reasoning.

This script focuses on Q11 and prints parser output, classifier output,
reasoning-engine output, and a small direct KG inspection.
"""
import json
import sys
from pathlib import Path


# Make direct execution work from the repository root with:
# `python scripts/debug_q11.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.question_classifier import QuestionClassifier
from src.kg_reasoning_engine import KGReasoningEngine
from src.question_parser import parse_question


# Initialize the classifier and reasoning engine used by the debug run.
classifier = QuestionClassifier()
reasoning_engine = KGReasoningEngine()

# Load all benchmark questions so Q11 can be selected by id.
with open("data/qa_92.json", encoding="utf-8") as f:
    questions = json.load(f)

# Check Q11, the question this debug script is designed to inspect.
q11 = next(q for q in questions if q["id"] == 11)
question = q11["question"]
ground_truth = q11["answer_spec"]["value"]

print("=" * 80)
print(f"Q11: {question}")
print(f"Ground Truth: {ground_truth}")
print("=" * 80)

# Parse the natural-language question into entities and predicates. These values
# are passed into the reasoning engine.
parsed = parse_question(question)
entities = parsed["entities"]
predicates = parsed["predicates"]
print(f"\nParsed entities: {entities}")
print(f"Parsed predicates: {predicates}")

# Classify the question to see which reasoning route the pipeline will use.
classified = classifier.classify(question)
print(f"\nClassified as: {classified.primary_type.name}")
print(f"Secondary types: {[t.name for t in classified.secondary_types]}")

# Run the KG reasoning engine and print the chosen strategy plus returned facts.
reasoned_facts, strategy = reasoning_engine.reason(classified, entities, predicates)
print(f"\nReasoning strategy: {strategy}")
print(f"Reasoned facts returned: {len(reasoned_facts)}")

# Show the full reasoning facts so missing or irrelevant supporting evidence is
# easy to inspect.
if reasoned_facts:
    print("\nReturned Facts:")
    for i, fact in enumerate(reasoned_facts, 1):
        print(f"  {i}. {fact}")

# Check raw KG records directly for the expected answer entities.
print("\n" + "=" * 80)
print("Checking raw KG for entity filtering details")
print("=" * 80)

# Load the KG directly to confirm whether relevant facts exist before any
# reasoning-engine filtering is applied.
with open("data/astronomy_kg1.json", encoding="utf-8") as f:
    kg = json.load(f)

# Look for facts attached to the expected ground-truth entities.
print("\nFacts in KG for 'Mars', 'Uranus', 'Neptune':")
for entity_name in ["Mars", "Uranus", "Neptune"]:
    matching = [
        f for f in kg if f.get("subject", "").lower() == entity_name.lower()
    ]
    print(f"\n{entity_name}: {len(matching)} facts")
    for fact in matching[:3]:
        print(
            f"  - {fact.get('subject')} | {fact.get('predicate')} | "
            f"{fact.get('object')}"
        )
