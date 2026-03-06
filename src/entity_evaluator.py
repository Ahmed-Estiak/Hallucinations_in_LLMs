from typing import Any, Dict


def normalize_text(text: Any) -> str:
    """
    Basic normalization for single-entity answers:
    - convert to string
    - remove leading/trailing spaces
    - lowercase everything
    """
    if text is None:
        return ""

    return str(text).strip().lower()


def evaluate_entity(answer: Any, truth_value: str) -> Dict[str, Any]:
    """
    entity evaluator logic:
    - normalize answer
    - normalize ground truth
    - exact string match only
    """
    normalized_answer = normalize_text(answer)
    normalized_truth = normalize_text(truth_value)

    is_correct = normalized_answer == normalized_truth

    return {
        "is_correct": is_correct,
        "manual_check": False,
        "normalized_answer": normalized_answer,
        "normalized_truth": normalized_truth,
        "reason": "matched" if is_correct else "entity_not_matched"
    }