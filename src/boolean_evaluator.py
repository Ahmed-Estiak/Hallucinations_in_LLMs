from typing import Any, Dict


def normalize_text(text: Any) -> str:
    """
    Basic normalization for boolean answers:
    - convert to string
    - remove leading/trailing spaces
    - lowercase everything
    """
    if text is None:
        return ""

    return str(text).strip().lower()


def evaluate_boolean(answer: Any, truth_value: bool) -> Dict[str, Any]:
    """
    boolean evaluator logic:
    - normalize answer
    - convert 'yes' -> True, 'no' -> False
    - compare predicted boolean with ground truth
    """
    normalized_answer = normalize_text(answer)

    if normalized_answer == "yes":
        predicted_value = True
    elif normalized_answer == "no":
        predicted_value = False
    else:
        return {
            "is_correct": False,
            "manual_check": False,
            "normalized_answer": normalized_answer,
            "predicted_value": None,
            "truth_value": truth_value,
            "reason": "invalid_boolean_format"
        }

    is_correct = predicted_value == truth_value

    return {
        "is_correct": is_correct,
        "manual_check": False,
        "normalized_answer": normalized_answer,
        "predicted_value": predicted_value,
        "truth_value": truth_value,
        "reason": "matched" if is_correct else "boolean_not_matched"
    }