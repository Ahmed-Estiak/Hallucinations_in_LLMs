import re
from typing import Any, Dict, List

from src.entity_evaluator import evaluate_entity


LEADING_NOISE_RE = re.compile(
    r"^(?:the\s+)?(?:final\s+answer|answer)\s*[:\-]\s*|^(?:the\s+answer\s+is|it\s+is|it's)\s+"
)
STRONG_LIST_SEPARATOR_RE = re.compile(r"\s*[,;|\n]\s*")
RELAXED_AND_SEPARATOR_RE = re.compile(r"\s+\band\b\s+")
ITEM_PREFIX_RE = re.compile(r"^(?:\d+[\).\:-]\s*|[-*]\s*)")


def normalize_text(text: Any) -> str:
    """
    Basic normalization for ordered-list answers.
    """
    if text is None:
        return ""

    text = str(text).strip().lower()
    return re.sub(r"\s+", " ", text)


def _clean_item(text: str) -> str:
    text = ITEM_PREFIX_RE.sub("", text.strip())
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;|")


def _parse_ordered_list(answer: str) -> Dict[str, Any]:
    normalized_answer = normalize_text(answer)
    stripped_answer = LEADING_NOISE_RE.sub("", normalized_answer)
    manual_check = stripped_answer != normalized_answer

    if not stripped_answer:
        return {
            "items": [],
            "manual_check": manual_check,
            "reason": "invalid_ordered_list_format",
        }

    if STRONG_LIST_SEPARATOR_RE.search(stripped_answer):
        raw_items = STRONG_LIST_SEPARATOR_RE.split(stripped_answer)
        items = [_clean_item(item) for item in raw_items if _clean_item(item)]
        return {
            "items": items,
            "manual_check": manual_check,
            "reason": "parsed_strong_separated_ordered_list",
        }

    if RELAXED_AND_SEPARATOR_RE.search(stripped_answer):
        raw_items = RELAXED_AND_SEPARATOR_RE.split(stripped_answer)
        items = [_clean_item(item) for item in raw_items if _clean_item(item)]
        return {
            "items": items,
            "manual_check": True,
            "reason": "parsed_relaxed_and_ordered_list",
        }

    item = _clean_item(stripped_answer)
    return {
        "items": [item] if item else [],
        "manual_check": manual_check,
        "reason": "parsed_single_ordered_item",
    }


def evaluate_ordered_list(answer: Any, truth_value: List[str]) -> Dict[str, Any]:
    """
    Evaluate an ordered entity list against the ground-truth ordered list.
    """
    normalized_answer = normalize_text(answer)
    normalized_truth = [normalize_text(item) for item in truth_value]
    parsed_result = _parse_ordered_list(normalized_answer)
    predicted_items = parsed_result["items"]
    manual_check = bool(parsed_result["manual_check"])

    if not predicted_items:
        return {
            "is_correct": False,
            "manual_check": manual_check,
            "normalized_answer": normalized_answer,
            "predicted_items": predicted_items,
            "normalized_truth": normalized_truth,
            "reason": parsed_result["reason"],
        }

    if len(predicted_items) != len(normalized_truth):
        return {
            "is_correct": False,
            "manual_check": True,
            "normalized_answer": normalized_answer,
            "predicted_items": predicted_items,
            "normalized_truth": normalized_truth,
            "reason": "ordered_list_length_not_matched",
        }

    item_results = [
        evaluate_entity(predicted_items[index], truth_value[index])
        for index in range(len(truth_value))
    ]

    is_correct = all(result["is_correct"] for result in item_results)
    combined_manual_check = manual_check or any(bool(result["manual_check"]) for result in item_results)

    if not is_correct:
        unordered_match = sorted(predicted_items) == sorted(normalized_truth)
        return {
            "is_correct": False,
            "manual_check": combined_manual_check or unordered_match,
            "normalized_answer": normalized_answer,
            "predicted_items": predicted_items,
            "normalized_truth": normalized_truth,
            "item_results": item_results,
            "reason": "ordered_list_order_not_matched" if unordered_match else "ordered_list_not_matched",
        }

    return {
        "is_correct": True,
        "manual_check": combined_manual_check,
        "normalized_answer": normalized_answer,
        "predicted_items": predicted_items,
        "normalized_truth": normalized_truth,
        "item_results": item_results,
        "reason": "matched",
    }
