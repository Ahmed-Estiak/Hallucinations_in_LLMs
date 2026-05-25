"""Dispatch model answers to evaluators for each ground-truth answer shape.

Question records store evaluation expectations in ``answer_spec``. This module
keeps the benchmark runner independent of the answer format by routing each
answer to the evaluator that knows how to compare that format.
"""

from typing import Any, Dict

from src.boolean_evaluator import evaluate_boolean
from src.entity_evaluator import evaluate_entity
from src.entity_list_evaluator import evaluate_entity_list
from src.multi_field import evaluate_multi_field
from src.ordered_list_evaluator import evaluate_ordered_list
from src.single_number_evaluator import evaluate_single_number


def evaluate_answer(question: Dict[str, Any], answer: Any) -> Dict[str, Any]:
    """Evaluate one model answer against the question's declared answer schema.

    Each specialized evaluator is responsible for format-specific parsing and
    matching. This function only selects the appropriate evaluator and returns
    its standard evaluation result dictionary.
    """
    # ``kind`` determines both the expected answer shape and comparison logic.
    kind = question["answer_spec"]["kind"]
    answer_spec = question["answer_spec"]

    # Scalar results are evaluated using their single declared ground-truth value.
    if kind == "single_number":
        return evaluate_single_number(answer, answer_spec["value"])

    if kind == "boolean":
        return evaluate_boolean(answer, answer_spec["value"])

    if kind == "entity":
        return evaluate_entity(answer, answer_spec["value"])

    # List evaluators differ because one ignores ordering while the other requires it.
    if kind == "entity_list":
        return evaluate_entity_list(answer, answer_spec["value"])

    if kind == "ordered_list":
        return evaluate_ordered_list(answer, answer_spec["value"])

    # Multi-field answers carry separate expected values rather than one value.
    if kind == "multi_field":
        return evaluate_multi_field(answer, answer_spec["fields"])

    # Preserve unexpected schemas for inspection instead of silently scoring them.
    return {
        "is_correct": False,
        "manual_check": True,
        "reason": f"unsupported_answer_kind:{kind}",
    }
