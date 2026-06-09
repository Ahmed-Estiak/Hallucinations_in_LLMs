"""
Inspect raw KG retrieval for selected list-answer benchmark questions.

For the hard-coded Q11/Q12 cases, the script prints:
    1. The benchmark question and expected answer.
    2. Entities, predicates, and time constraint produced by question parsing.
    3. The answer-shape classification produced by QuestionClassifier.
    4. Raw facts returned directly by KGRetriever with a per-query limit of 3.
    5. The exact formatted fact text suitable for insertion into a KG prompt.

This deliberately stops before KGReasoningEngine and before any LLM call. It is
therefore useful for deciding whether a list-answer failure originates in
question parsing/raw retrieval or in a later reasoning/prompt/model stage.

The script is read-only: it loads ``data/qa_92.json`` and the retriever's KG
data, prints diagnostics, and does not write reports or call provider APIs.
"""
import json
import sys
from pathlib import Path


# Add the repository root so ``src`` imports work when this file is executed as
# a script instead of imported as a package module.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.question_classifier import QuestionClassifier
from src.kg_retriever import KGRetriever
from src.question_parser import parse_question


# Load the benchmark definitions once. Each selected row provides both the
# natural-language question and its expected entity-list answer for comparison.
with open("data/qa_92.json") as f:
    questions = json.load(f)

# These IDs are intentionally explicit because this is a focused diagnostic,
# not a general benchmark runner. Q11 and Q12 exercise LIST-style answers and
# expose whether raw KG retrieval covers all expected entities.
for qid in [11, 12]:
    # Select the complete benchmark row so retrieved facts can be inspected
    # against the declared ground truth.
    q = next(q for q in questions if q["id"] == qid)
    question = q["question"]
    ground_truth = q["answer_spec"]["value"]

    print("\n" + "=" * 80)
    print(f"Q{qid}: {question}")
    print(f"Ground Truth: {ground_truth}")
    print("=" * 80)

    # parse_question supplies the exact low-level arguments passed to
    # KGRetriever below. Missing entities/predicates at this stage explain why
    # relevant facts may never reach later reasoning.
    parsed = parse_question(question)
    entities = parsed["entities"]
    predicates = parsed["predicates"]
    time_constraint = parsed["time_constraint"]

    print(f"Parsed entities: {entities}")
    print(f"Parsed predicates: {predicates}")
    print(f"Time constraint: {time_constraint}")

    # Classification is displayed for diagnosis only. This script does not use
    # the classification to invoke KGReasoningEngine or alter raw retrieval.
    classifier = QuestionClassifier()
    classified = classifier.classify(question)
    print(f"\nClassified as: {classified.primary_type.name}")

    # Retrieve raw KG facts directly. The limit of 3 is intentionally small so
    # missing list members caused by retrieval ordering/capping are visible.
    # Full KG benchmark behavior may additionally apply KGReasoningEngine.
    kg_retriever = KGRetriever()
    facts = kg_retriever.retrieve(entities, predicates, time_constraint, limit=3)

    print(f"\nRetrieved {len(facts)} facts:")
    for f in facts:
        print(
            f"  - {f.get('subject')} | {f.get('predicate')} | "
            f"{f.get('object')} ({f.get('time', 'unknown')})"
        )

    # Format only the retrieved raw facts. This shows the prompt-ready evidence
    # boundary, but no prompt is built and no LLM/provider is called.
    formatted = kg_retriever.format_facts_for_prompt(facts)
    print(f"\nFormatted for prompt:\n{formatted}")
