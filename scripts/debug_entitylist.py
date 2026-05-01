"""
Debug ENTITY_LIST-style questions to inspect retrieved KG facts.

This script focuses on selected benchmark questions, then prints:
1. The original question and ground-truth answer.
2. Parsed entities, predicates, and time constraints.
3. The classifier's predicted question type.
4. Raw KG facts and the formatted prompt context built from them.
"""
import json
import sys
from pathlib import Path


# Add the repository root to Python's import path so this script works when it
# is run directly as `python scripts/debug_entitylist.py`.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.kg_retriever import KGRetriever
from src.question_classifier import QuestionClassifier
from src.question_parser import parse_question


QUESTIONS_PATH = "data/qa_92.json"
QUESTION_IDS_TO_DEBUG = [11, 12]
SEPARATOR_WIDTH = 80
RETRIEVAL_LIMIT = 3


def load_questions(path):
    """
    Load the benchmark questions from a JSON file.

    The file is expected to contain a list of question records, where each
    record includes an id, question text, and answer specification.
    """
    with open(path) as f:
        return json.load(f)


def find_question_by_id(questions, qid):
    """
    Return the question record that matches the requested question id.

    Using a dedicated helper makes the intent clearer than keeping the lookup
    inline inside the debugging loop.
    """
    return next(q for q in questions if q["id"] == qid)


def print_question_header(qid, question, ground_truth):
    """
    Print the question being inspected and its expected answer.

    This header separates each debug case so the parser, classifier, and KG
    retrieval output can be read without mixing it with the previous question.
    """
    print("\n" + "=" * SEPARATOR_WIDTH)
    print(f"Q{qid}: {question}")
    print(f"Ground Truth: {ground_truth}")
    print("=" * SEPARATOR_WIDTH)


def print_parsed_question(question):
    """
    Parse a natural-language question and print the extracted query signals.

    The parser output drives KG retrieval, so exposing entities, predicates, and
    time constraints helps explain why specific KG facts are returned.
    """
    parsed = parse_question(question)
    entities = parsed["entities"]
    predicates = parsed["predicates"]
    time_constraint = parsed["time_constraint"]

    print(f"Parsed entities: {entities}")
    print(f"Parsed predicates: {predicates}")
    print(f"Time constraint: {time_constraint}")

    return entities, predicates, time_constraint


def print_classification(classifier, question):
    """
    Classify the question and print the primary type selected by the classifier.

    This confirms whether the question is being routed as the expected
    high-level task type before KG facts are used.
    """
    classified = classifier.classify(question)
    print(f"\nClassified as: {classified.primary_type.name}")


def print_retrieved_facts(facts):
    """
    Print raw KG facts returned by the retriever in a compact row format.

    Each fact shows subject, predicate, object, and optional time metadata so
    incorrect or irrelevant retrievals are easy to spot.
    """
    print(f"\nRetrieved {len(facts)} facts:")
    for fact in facts:
        print(
            "  - "
            f"{fact.get('subject')} | "
            f"{fact.get('predicate')} | "
            f"{fact.get('object')} "
            f"({fact.get('time', 'unknown')})"
        )


def print_formatted_prompt_context(kg_retriever, facts):
    """
    Print the final KG context string that would be inserted into the prompt.

    Comparing raw facts against this formatted text helps identify formatting or
    summarization issues between retrieval and model prompting.
    """
    formatted = kg_retriever.format_facts_for_prompt(facts)
    print(f"\nFormatted for prompt:\n{formatted}")


def debug_question(question_record, classifier, kg_retriever):
    """
    Run the full debug flow for a single benchmark question.

    The function keeps all inspection steps together: display the question,
    parse it, classify it, retrieve KG facts, and show the prompt-ready context.
    """
    qid = question_record["id"]
    question = question_record["question"]
    ground_truth = question_record["answer_spec"]["value"]

    print_question_header(qid, question, ground_truth)
    entities, predicates, time_constraint = print_parsed_question(question)
    print_classification(classifier, question)

    facts = kg_retriever.retrieve(
        entities,
        predicates,
        time_constraint,
        limit=RETRIEVAL_LIMIT,
    )

    print_retrieved_facts(facts)
    print_formatted_prompt_context(kg_retriever, facts)


def main():
    """
    Load questions and debug the selected question ids.

    Keeping execution inside main() makes this module import-safe, so tests or
    other scripts can import its helpers without immediately running the debug
    report.
    """
    questions = load_questions(QUESTIONS_PATH)
    classifier = QuestionClassifier()
    kg_retriever = KGRetriever()

    for qid in QUESTION_IDS_TO_DEBUG:
        question_record = find_question_by_id(questions, qid)
        debug_question(question_record, classifier, kg_retriever)


if __name__ == "__main__":
    main()
