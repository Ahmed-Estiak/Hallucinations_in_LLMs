import re
import math
from typing import Any, Dict, List


NUMBER_PATTERN = r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?"


def normalize_text(text: Any) -> str:
    """
    Basic normalization for model answers.
    """
    if text is None:
        return ""

    text = str(text).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def extract_numbers(text: Any) -> List[float]:
    """
    Extract all numbers from text.
    Example:
    'abc 45 yy 49 zz 59,39' -> [45.0, 49.0, 59.0, 39.0]
    """
    text = "" if text is None else str(text)
    matches = re.findall(NUMBER_PATTERN, text)

    if not matches:
        return []

    return [float(x) for x in matches]


def exact_or_approx_match(pred_num: float, truth_num: float, tol: float = 1e-9) -> bool:
    """
    Exact match first; otherwise allow a tiny tolerance.
    """
    if pred_num == truth_num:
        return True

    return math.isclose(pred_num, truth_num, rel_tol=tol, abs_tol=tol)


def evaluate_single_number(answer: Any, truth_value: float) -> Dict[str, Any]:
    """
    single_number evaluator logic:
    - normalize answer
    - extract all numbers
    - if no number -> incorrect
    - if any extracted number matches truth -> correct
    - if multiple numbers are present -> manual_check = True
    """
    normalized_answer = normalize_text(answer)
    predicted_numbers = extract_numbers(normalized_answer)

    if not predicted_numbers:
        return {
            "is_correct": False,
            "manual_check": False,
            "normalized_answer": normalized_answer,
            "predicted_numbers": predicted_numbers,
            "reason": "no_number_found"
        }

    matched = False
    for num in predicted_numbers:
        if exact_or_approx_match(num, truth_value):
            matched = True
            break

    manual_check = len(predicted_numbers) > 1

    return {
        "is_correct": matched,
        "manual_check": manual_check,
        "normalized_answer": normalized_answer,
        "predicted_numbers": predicted_numbers,
        "reason": "matched" if matched else "truth_not_found_in_predicted_numbers"
    }