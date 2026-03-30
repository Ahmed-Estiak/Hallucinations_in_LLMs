import re
from typing import Any, Dict, List, Optional, Set

from src.entity_evaluator import evaluate_entity


LEADING_NOISE_RE = re.compile(
    r"^(?:the\s+)?(?:final\s+answer|answer)\s*[:\-]\s*|^(?:the\s+answer\s+is|it\s+is|it's)\s+"
)
LIST_SEPARATOR_RE = re.compile(r"\s*,\s*|\s*;\s*|\s*\|\s*|\s*\n\s*")
ITEM_PREFIX_RE = re.compile(r"^(?:\d+[\).\:-]\s*|[-*]\s*)")


def normalize_text(text: Any) -> str:
    """
    Basic normalization for entity-list answers.
    """
    if text is None:
        return ""

    text = str(text).strip().lower()
    return re.sub(r"\s+", " ", text)


def _clean_item(text: str) -> str:
    text = ITEM_PREFIX_RE.sub("", text.strip())
    text = re.sub(r"^and\s+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" ,;|")


def _split_rightmost_and(text: str, splits_needed: int) -> Optional[List[str]]:
    parts = [text.strip()]

    for _ in range(splits_needed):
        current_text = parts[0]
        rightmost_match = None
        for match in re.finditer(r"\s+\band\b\s+", current_text):
            rightmost_match = match

        if rightmost_match is None:
            return None

        left = current_text[:rightmost_match.start()].strip()
        right = current_text[rightmost_match.end():].strip()
        parts = [left, right, *parts[1:]]

    return parts if all(parts) else None


def _normalize_list_items(text: str, expected_count: int) -> List[str]:
    text = re.sub(r"\s*;\s*", ", ", text)
    text = re.sub(r"\s*\|\s*", ", ", text)
    text = re.sub(r"\s*\n\s*", ", ", text)
    text = re.sub(r"\s*,\s*", ", ", text)
    text = re.sub(r"^\s*and\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" ,;|")

    if "," in text:
        text = re.sub(r",\s*and\s+", ", ", text)
        items = [_clean_item(item) for item in LIST_SEPARATOR_RE.split(text) if _clean_item(item)]

        if len(items) < expected_count and items:
            tail_splits_needed = expected_count - len(items)
            tail_parts = _split_rightmost_and(items[-1], tail_splits_needed)
            if tail_parts is not None:
                items = items[:-1] + [_clean_item(item) for item in tail_parts if _clean_item(item)]

        return items

    and_splits_needed = max(expected_count - 1, 0)
    and_items = _split_rightmost_and(text, and_splits_needed)
    if and_items is not None and len(and_items) == expected_count:
        return [_clean_item(item) for item in and_items if _clean_item(item)]

    item = _clean_item(text)
    return [item] if item else []


def _parse_entity_list(answer: str, expected_count: int) -> Dict[str, Any]:
    normalized_answer = normalize_text(answer)
    stripped_answer = LEADING_NOISE_RE.sub("", normalized_answer)
    manual_check = stripped_answer != normalized_answer

    if not stripped_answer:
        return {
            "items": [],
            "manual_check": manual_check,
            "reason": "invalid_entity_list_format",
        }

    items = _normalize_list_items(stripped_answer, expected_count)
    if len(items) > 1:
        return {
            "items": items,
            "manual_check": manual_check or normalize_text(", ".join(items)) != stripped_answer,
            "reason": "parsed_normalized_entity_list",
        }

    item = items[0] if items else ""
    return {
        "items": [item] if item else [],
        "manual_check": manual_check,
        "reason": "parsed_single_entity_item",
    }


def _match_unordered_items(
    predicted_items: List[str],
    truth_items: List[str],
) -> Optional[List[Dict[str, Any]]]:
    def backtrack(index: int, used_truth: Set[int], current_results: List[Dict[str, Any]]) -> Optional[List[Dict[str, Any]]]:
        if index == len(predicted_items):
            return current_results

        predicted_item = predicted_items[index]

        for truth_index, truth_item in enumerate(truth_items):
            if truth_index in used_truth:
                continue

            item_result = evaluate_entity(predicted_item, truth_item)
            if not item_result["is_correct"]:
                continue

            matched_result = backtrack(
                index + 1,
                used_truth | {truth_index},
                current_results + [item_result],
            )
            if matched_result is not None:
                return matched_result

        return None

    return backtrack(0, set(), [])


def evaluate_entity_list(answer: Any, truth_value: List[str]) -> Dict[str, Any]:
    """
    Evaluate an unordered entity list against the ground-truth entity list.
    """
    normalized_answer = normalize_text(answer)
    normalized_truth = [normalize_text(item) for item in truth_value]
    parsed_result = _parse_entity_list(normalized_answer, len(normalized_truth))
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
            "reason": "entity_list_length_not_matched",
        }

    item_results = _match_unordered_items(predicted_items, truth_value)
    if item_results is None:
        return {
            "is_correct": False,
            "manual_check": manual_check,
            "normalized_answer": normalized_answer,
            "predicted_items": predicted_items,
            "normalized_truth": normalized_truth,
            "reason": "entity_list_not_matched",
        }

    combined_manual_check = manual_check or any(bool(result["manual_check"]) for result in item_results)

    return {
        "is_correct": True,
        "manual_check": combined_manual_check,
        "normalized_answer": normalized_answer,
        "predicted_items": predicted_items,
        "normalized_truth": normalized_truth,
        "item_results": item_results,
        "reason": "matched",
    }
