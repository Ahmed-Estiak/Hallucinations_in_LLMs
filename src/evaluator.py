from typing import Any, Dict

from src.boolean_evaluator import evaluate_boolean
from src.entity_evaluator import evaluate_entity
from src.entity_list_evaluator import evaluate_entity_list
from src.multi_field import evaluate_multi_field
from src.ordered_list_evaluator import evaluate_ordered_list
from src.single_number_evaluator import evaluate_single_number


def evaluate_answer(question: Dict[str, Any], answer: Any) -> Dict[str, Any]:
    kind = question["answer_spec"]["kind"]
    answer_spec = question["answer_spec"]

    if kind == "single_number":
        return evaluate_single_number(answer, answer_spec["value"])

    if kind == "boolean":
        return evaluate_boolean(answer, answer_spec["value"])

    if kind == "entity":
        return evaluate_entity(answer, answer_spec["value"])

    if kind == "entity_list":
        return evaluate_entity_list(answer, answer_spec["value"])

    if kind == "ordered_list":
        return evaluate_ordered_list(answer, answer_spec["value"])

    if kind == "multi_field":
        return evaluate_multi_field(answer, answer_spec["fields"])

    return {
        "is_correct": False,
        "manual_check": True,
        "reason": f"unsupported_answer_kind:{kind}",
    }
