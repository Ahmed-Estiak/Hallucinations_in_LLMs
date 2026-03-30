import re
from typing import Any, Dict, List, Optional

from src.boolean_evaluator import evaluate_boolean
from src.entity_evaluator import evaluate_entity
from src.single_number_evaluator import evaluate_single_number


def normalize_text(text: Any) -> str:
    """
    Basic normalization for multi-field answers.
    """
    if text is None:
        return ""

    text = str(text).strip().lower()
    return re.sub(r"\s+", " ", text)


def _normalize_field_name(field_name: str) -> str:
    return re.sub(r"\s+", " ", field_name.replace("_", " ").strip().lower())


def _field_label_variants(field_name: str) -> List[str]:
    normalized_name = _normalize_field_name(field_name)
    variants = {normalized_name}

    if " " in normalized_name:
        variants.add(normalized_name.replace(" ", "_"))

    return sorted(variants, key=len, reverse=True)


def _field_candidate_strong_fit(field_type: str, candidate: str) -> bool:
    candidate = normalize_text(candidate)

    if field_type in {"single_number", "number", "year"}:
        return bool(re.fullmatch(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?", candidate))

    if field_type == "boolean":
        return candidate in {"yes", "no"}

    if field_type == "entity":
        return bool(candidate)

    return bool(candidate)


def _extract_labeled_candidates(answer: str, fields: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    candidates: Dict[str, str] = {}
    label_matches = []

    for field in fields:
        field_name = field["name"]
        label_pattern = "|".join(re.escape(label) for label in _field_label_variants(field_name))
        for match in re.finditer(rf"\b(?:{label_pattern})\b(?:\s*[:\-]\s*|\s+)", answer):
            label_matches.append({
                "field_name": field_name,
                "start": match.start(),
                "end": match.end(),
            })

    if not label_matches:
        return None

    label_matches.sort(key=lambda item: item["start"])
    unique_label_matches = []
    seen_fields = set()

    for match in label_matches:
        if match["field_name"] in seen_fields:
            continue
        seen_fields.add(match["field_name"])
        unique_label_matches.append(match)

    if len(unique_label_matches) != len(fields):
        return None

    for index, match in enumerate(unique_label_matches):
        field_name = match["field_name"]
        start = match["end"]
        end = (
            unique_label_matches[index + 1]["start"]
            if index + 1 < len(unique_label_matches)
            else len(answer)
        )
        candidate = answer[start:end].strip(" ,;|")

        if candidate:
            candidates[field_name] = candidate

    if len(candidates) == len(fields):
        return candidates

    return None


def _split_ordered_candidates(answer: str, expected_count: int) -> Optional[List[str]]:
    parts = [part.strip() for part in re.split(r"\s*[,;|]\s*", answer) if part.strip()]
    if len(parts) == expected_count:
        return parts

    return None


def _split_relaxed_two_field_candidates(answer: str, fields: List[Dict[str, Any]]) -> Optional[List[str]]:
    if len(fields) != 2:
        return None

    if answer.count(" and ") != 1:
        return None

    left_candidate, right_candidate = [part.strip() for part in answer.split(" and ", 1)]
    if not left_candidate or not right_candidate:
        return None

    first_field_type = str(fields[0]["type"]).lower()
    second_field_type = str(fields[1]["type"]).lower()

    if not _field_candidate_strong_fit(first_field_type, left_candidate):
        return None

    if not _field_candidate_strong_fit(second_field_type, right_candidate):
        return None

    return [left_candidate, right_candidate]


def _evaluate_field(field_type: str, candidate: Any, truth_value: Any) -> Dict[str, Any]:
    if field_type in {"single_number", "number", "year"}:
        return evaluate_single_number(candidate, truth_value)

    if field_type == "entity":
        return evaluate_entity(candidate, truth_value)

    if field_type == "boolean":
        return evaluate_boolean(candidate, truth_value)

    normalized_candidate = normalize_text(candidate)
    normalized_truth = normalize_text(truth_value)

    return {
        "is_correct": normalized_candidate == normalized_truth,
        "manual_check": False,
        "normalized_answer": normalized_candidate,
        "normalized_truth": normalized_truth,
        "reason": "matched" if normalized_candidate == normalized_truth else "unsupported_field_type",
    }


def evaluate_multi_field(answer: Any, fields: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate a multi-field answer against a schema-driven field list.
    """
    normalized_answer = normalize_text(answer)
    candidate_map = _extract_labeled_candidates(normalized_answer, fields)
    mapping_strategy = "labeled" if candidate_map is not None else None

    if candidate_map is None:
        ordered_candidates = _split_ordered_candidates(normalized_answer, len(fields))
        if ordered_candidates is not None:
            candidate_map = {
                field["name"]: ordered_candidates[index]
                for index, field in enumerate(fields)
            }
            mapping_strategy = "ordered"
        else:
            relaxed_candidates = _split_relaxed_two_field_candidates(normalized_answer, fields)
            if relaxed_candidates is not None:
                candidate_map = {
                    field["name"]: relaxed_candidates[index]
                    for index, field in enumerate(fields)
                }
                mapping_strategy = "relaxed_ordered"
            else:
                candidate_map = {
                    field["name"]: normalized_answer
                    for field in fields
                }
                mapping_strategy = "fallback"

    field_results: Dict[str, Dict[str, Any]] = {}

    for field in fields:
        field_name = field["name"]
        field_type = str(field["type"]).lower()
        truth_value = field["value"]
        candidate = candidate_map.get(field_name, "")

        field_results[field_name] = _evaluate_field(field_type, candidate, truth_value)

    is_correct = all(result["is_correct"] for result in field_results.values())
    manual_check = any(bool(result["manual_check"]) for result in field_results.values())

    if mapping_strategy == "fallback" and len(fields) > 1:
        is_correct = False
        manual_check = True
    elif mapping_strategy == "relaxed_ordered":
        manual_check = True

    predicted_fields = {
        field["name"]: candidate_map.get(field["name"], "")
        for field in fields
    }

    if is_correct:
        reason = "matched"
    elif mapping_strategy == "fallback" and len(fields) > 1:
        reason = "unstructured_multi_field_answer"
    else:
        reason = "multi_field_not_matched"

    return {
        "is_correct": is_correct,
        "manual_check": manual_check,
        "normalized_answer": normalized_answer,
        "predicted_fields": predicted_fields,
        "field_results": field_results,
        "mapping_strategy": mapping_strategy,
        "reason": reason,
    }
