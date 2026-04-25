"""
Run the benchmark using only KG retrieval/reasoning outputs, with no LLM calls.
Saves results to results/results_kg_only.csv.
"""
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# Add workspace root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluator import evaluate_answer
from src.kg_reasoning_engine import KGReasoningEngine
from src.kg_retriever import KGRetriever
from src.kg_runner import _prepare_question_context
from src.question_classifier import LogicalModifier, QuestionClassifier, QuestionType
from src.time_utils import time_window


def _serialize_ground_truth(answer_spec: Dict[str, Any]) -> str:
    if "value" in answer_spec:
        return json.dumps(answer_spec["value"], ensure_ascii=False)
    if "fields" in answer_spec:
        return json.dumps(answer_spec["fields"], ensure_ascii=False)
    return ""


def _first_nonempty(values: List[Optional[str]]) -> str:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return ""


def _normalize_scalar_answer(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    return str(value).strip()


def _date_like_value_to_year(value: Any) -> str:
    text = _normalize_scalar_answer(value)
    if not text:
        return ""
    if len(text) >= 4 and text[:4].isdigit():
        return text[:4]
    return text


def _extract_numeric(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except Exception:
        return None


def _extract_time_order_value(fact: Dict[str, Any]) -> Optional[float]:
    for candidate in (fact.get("object"), fact.get("time")):
        window = time_window(candidate)
        if window is not None:
            return float(window[0])
    return None


def _facts_for_answer(context: Dict[str, Any]) -> List[Dict[str, Any]]:
    return context["reasoned_facts"] or context["kg_facts"]


def _derived_entities(context: Dict[str, Any]) -> List[str]:
    reasoned_facts = context["reasoned_facts"]
    if reasoned_facts and reasoned_facts[0].get("_derived_result"):
        return [str(entity) for entity in reasoned_facts[0].get("entities", [])]
    return []


def _answer_single_number(question_row: Dict[str, Any], context: Dict[str, Any]) -> str:
    facts = _facts_for_answer(context)
    if not facts:
        return ""
    fact = facts[0]
    predicate = str(fact.get("predicate", "")).lower()
    if predicate in {"discovered_on", "recognized_on", "first_observed_on"}:
        return _date_like_value_to_year(fact.get("object"))
    return _normalize_scalar_answer(fact.get("object"))


def _answer_multi_field(question_row: Dict[str, Any], context: Dict[str, Any], kg_reasoning_engine: KGReasoningEngine) -> str:
    classified_q = context["classified_q"]
    parts: List[str] = []

    for field_spec in classified_q.fields:
        if not field_spec.entity or not field_spec.predicate:
            parts.append("")
            continue
        fact = kg_reasoning_engine._latest_fact_for(field_spec.entity, field_spec.predicate, classified_q)
        if not fact:
            parts.append("")
            continue

        if field_spec.predicate in {"discovered_on", "recognized_on", "first_observed_on"}:
            parts.append(_date_like_value_to_year(fact.get("object")))
        else:
            parts.append(_normalize_scalar_answer(fact.get("object")))

    return ", ".join(part for part in parts if part)


def _answer_comparison_entity(context: Dict[str, Any]) -> str:
    facts = _facts_for_answer(context)
    if not facts:
        return ""

    classified_q = context["classified_q"]
    operator = classified_q.comparison_operator
    predicate = str(facts[0].get("predicate", "")).lower()

    best_fact = None
    best_value = None
    for fact in facts:
        if predicate in {"discovered_on", "recognized_on", "first_observed_on"}:
            value = _extract_time_order_value(fact)
        else:
            value = _extract_numeric(fact.get("object"))
        if value is None:
            continue
        if best_fact is None:
            best_fact = fact
            best_value = value
            continue

        if operator == "<" and value < best_value:
            best_fact, best_value = fact, value
        elif operator != "<" and value > best_value:
            best_fact, best_value = fact, value

    if best_fact is None:
        return ""
    return _normalize_scalar_answer(best_fact.get("subject"))


def _answer_boolean(question_row: Dict[str, Any], context: Dict[str, Any], kg_reasoning_engine: KGReasoningEngine) -> str:
    question = question_row["question"].lower()
    entity = _first_nonempty(context["entities"] or context["classified_q"].major_entities)
    if not entity:
        return ""

    classified_q = context["classified_q"]
    checks: List[bool] = []

    if "dwarf planet" in question:
        classification_fact = kg_reasoning_engine._latest_fact_for(entity, "classification", classified_q)
        checks.append(
            bool(classification_fact and str(classification_fact.get("object", "")).strip().lower() == "dwarf planet")
        )

    if "largest diameter" in question or "largest size" in question:
        size_flag = kg_reasoning_engine._latest_fact_for(entity, "is_largest_dwarf_planet_by_size", classified_q)
        if size_flag is not None:
            checks.append(bool(size_flag.get("object")))
        else:
            diameter_fact = kg_reasoning_engine._latest_fact_for(entity, "diameter", classified_q)
            if diameter_fact:
                entity_value = _extract_numeric(diameter_fact.get("object"))
                best_value = entity_value
                best_entity = entity
                for dwarf in kg_reasoning_engine._candidate_dwarf_planets():
                    fact = kg_reasoning_engine._latest_fact_for(dwarf, "diameter", classified_q)
                    value = _extract_numeric(fact.get("object")) if fact else None
                    if value is not None and (best_value is None or value > best_value):
                        best_value = value
                        best_entity = dwarf
                checks.append(best_entity.lower() == entity.lower())

    if "located in" in question or "found in" in question:
        if "kuiper belt" in question:
            location_fact = kg_reasoning_engine._latest_fact_for(entity, "location", classified_q)
            checks.append(bool(location_fact and str(location_fact.get("object", "")).strip().lower() == "kuiper belt"))
        if "asteroid belt" in question:
            location_fact = kg_reasoning_engine._latest_fact_for(entity, "location", classified_q)
            checks.append(bool(location_fact and str(location_fact.get("object", "")).strip().lower() == "asteroid belt"))

    if not checks:
        facts = _facts_for_answer(context)
        if facts:
            first_object = facts[0].get("object")
            if isinstance(first_object, bool):
                return "yes" if first_object else "no"
        return ""

    return "yes" if all(checks) else "no"


def _answer_entity(question_row: Dict[str, Any], context: Dict[str, Any], kg_reasoning_engine: KGReasoningEngine) -> str:
    classified_q = context["classified_q"]
    if classified_q.primary_type == QuestionType.BOOLEAN or question_row["answer_spec"]["kind"] == "boolean":
        return _answer_boolean(question_row, context, kg_reasoning_engine)

    derived_entities = _derived_entities(context)
    if derived_entities:
        return derived_entities[0]

    if LogicalModifier.COMPARISON in classified_q.logical_modifiers:
        return _answer_comparison_entity(context)

    facts = _facts_for_answer(context)
    if not facts:
        return ""

    fact = facts[0]
    if fact.get("predicate") in {"discovered_by"}:
        return _normalize_scalar_answer(fact.get("object"))
    if fact.get("predicate") in {"classification", "planet_type", "location"}:
        return _normalize_scalar_answer(fact.get("object"))
    return _normalize_scalar_answer(fact.get("subject"))


def _answer_entity_list(context: Dict[str, Any]) -> str:
    derived_entities = _derived_entities(context)
    if derived_entities:
        return ", ".join(derived_entities)

    facts = _facts_for_answer(context)
    subjects = []
    seen = set()
    for fact in facts:
        subject = _normalize_scalar_answer(fact.get("subject"))
        if subject and subject not in seen:
            seen.add(subject)
            subjects.append(subject)
    return ", ".join(subjects)


def _answer_ordered_list(context: Dict[str, Any]) -> str:
    return _answer_entity_list(context)


def kg_only_answer(question_row: Dict[str, Any], context: Dict[str, Any], kg_reasoning_engine: KGReasoningEngine) -> str:
    kind = question_row["answer_spec"]["kind"]

    if kind == "single_number":
        return _answer_single_number(question_row, context)
    if kind == "boolean":
        return _answer_boolean(question_row, context, kg_reasoning_engine)
    if kind == "entity":
        return _answer_entity(question_row, context, kg_reasoning_engine)
    if kind == "entity_list":
        return _answer_entity_list(context)
    if kind == "ordered_list":
        return _answer_ordered_list(context)
    if kind == "multi_field":
        return _answer_multi_field(question_row, context, kg_reasoning_engine)

    return ""


def run_kg_only_benchmark() -> None:
    start_time = time.time()

    with open("data/qa_92.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    question_classifier = QuestionClassifier()
    kg_retriever = KGRetriever()
    kg_reasoning_engine = KGReasoningEngine()
    results = []
    Path("results").mkdir(exist_ok=True)

    total_questions = len(questions)
    kg_found_count = 0
    kg_only_correct = 0

    print(f"Starting KG-only benchmark ({total_questions} questions)")
    print("=" * 80)

    for index, question_row in enumerate(questions, start=1):
        question_start = time.time()
        qid = question_row["id"]
        question = question_row["question"]
        print(f"[{index}/{total_questions}] Q{qid}: {question[:60]}...")

        context = _prepare_question_context(
            question,
            question_classifier,
            kg_retriever,
            kg_reasoning_engine,
        )
        if context["kg_found"]:
            kg_found_count += 1

        answer = kg_only_answer(question_row, context, kg_reasoning_engine)
        evaluation = evaluate_answer(question_row, answer)
        kg_only_correct += int(bool(evaluation["is_correct"]))

        results.append({
            "id": qid,
            "question": question,
            "kind": question_row["answer_spec"]["kind"],
            "type": question_row.get("type", ""),
            "ground_truth": _serialize_ground_truth(question_row["answer_spec"]),
            "primary_type": context["classified_q"].primary_type.name if context["classified_q"].primary_type else "UNKNOWN",
            "time_semantic": context["classified_q"].time_semantic.name if context["classified_q"].time_semantic else "NONE",
            "logical_modifiers": json.dumps([modifier.name for modifier in context["classified_q"].logical_modifiers], ensure_ascii=False),
            "reasoning_strategy": context["reasoning_strategy"],
            "kg_found": context["kg_found"],
            "kg_facts_count": len(context["kg_facts"]),
            "parsed_entities": json.dumps(context["entities"], ensure_ascii=False),
            "parsed_predicates": json.dumps(context["predicates"], ensure_ascii=False),
            "time_constraint": context["time_constraint"],
            "kg_facts_text": context["kg_facts_text"],
            "kg_only_answer": answer,
            "kg_only_is_correct": evaluation["is_correct"],
            "kg_only_reason": evaluation["reason"],
        })

        question_elapsed = time.time() - question_start
        print(f"  KG-only answer: {answer}")
        print(f"  Timing -> Question total: {question_elapsed:.2f}s")

    df = pd.DataFrame(results)
    df.to_csv("results/results_kg_only.csv", index=False)

    elapsed_seconds = time.time() - start_time
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY - KG Only")
    print("=" * 80)
    print(f"Total questions: {total_questions}")
    print(f"KG facts found for: {kg_found_count}/{total_questions} questions ({(kg_found_count/total_questions)*100:.1f}%)")
    print(f"KG-only correct: {kg_only_correct}/{total_questions} ({(kg_only_correct/total_questions)*100:.2f}%)")
    print("\nResults saved to: results/results_kg_only.csv")
    print("=" * 80)
    print(f"Total runtime: {elapsed_seconds:.2f} seconds ({elapsed_seconds / 60:.2f} minutes)")


if __name__ == "__main__":
    run_kg_only_benchmark()
