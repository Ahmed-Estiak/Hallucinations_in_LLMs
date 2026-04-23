"""
KG Runner: Benchmark with KG-grounded LLM answers
Outputs results to results_with_kg.csv
Integrates Advanced KG Reasoning System with question classification
"""
import json
import time
import sys
from pathlib import Path

# Add workspace root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.models import ask_openai, ask_gemini, split_multifield_question
from src.question_parser import parse_question
from src.kg_retriever import KGRetriever
from src.kg_models import ask_openai_with_kg, ask_gemini_with_kg
from src.evaluator import evaluate_answer
from src.question_classifier import QuestionClassifier, QuestionType, LogicalModifier
from src.kg_reasoning_engine import KGReasoningEngine, format_reasoned_facts


def _serialize_ground_truth(answer_spec):
    """Serialize ground truth from answer spec (same as original runner)."""
    if "value" in answer_spec:
        return json.dumps(answer_spec["value"], ensure_ascii=False)

    if "fields" in answer_spec:
        return json.dumps(answer_spec["fields"], ensure_ascii=False)

    return ""


def _load_vanilla_results():
    """Load vanilla results from results.csv keyed by question ID."""
    path = Path("results/results.csv")
    if not path.exists():
        return {}

    df = pd.read_csv(path)
    if "id" not in df.columns:
        return {}

    vanilla_data = {}
    for _, row in df.iterrows():
        try:
            qid = int(row["id"])
            vanilla_data[qid] = {
                "openai_answer": row.get("openai_answer", None),
                "gemini_answer": row.get("gemini_answer", None),
                "openai_is_correct": row.get("openai_is_correct", None),
                "gemini_is_correct": row.get("gemini_is_correct", None),
                "openai_reason": row.get("openai_reason", None),
                "gemini_reason": row.get("gemini_reason", None),
            }
        except Exception:
            continue  # Skip invalid rows

    return vanilla_data


def _prepare_question_context(question, question_classifier, kg_retriever, kg_reasoning_engine,
                              time_semantic_override=None, time_constraint_override=None):
    """Parse/classify one question and prepare KG context."""
    parsed = parse_question(question)
    entities = parsed["entities"]
    predicates = parsed["predicates"]
    time_constraint = parsed["time_constraint"]

    classified_q = question_classifier.classify(question)
    if time_semantic_override is not None and time_constraint_override is not None:
        classified_q.has_time_constraint = True
        classified_q.time_semantic = time_semantic_override
        classified_q.time_value = time_constraint_override
        if LogicalModifier.TIME_LOOKUP not in classified_q.logical_modifiers:
            classified_q.logical_modifiers.append(LogicalModifier.TIME_LOOKUP)
        time_constraint = time_constraint_override

    use_kg_reasoning = (
        classified_q.primary_type == QuestionType.MULTI_FIELD or
        LogicalModifier.FILTER in classified_q.logical_modifiers or
        LogicalModifier.ORDERING in classified_q.logical_modifiers or
        LogicalModifier.COMPARISON in classified_q.logical_modifiers or
        classified_q.has_time_constraint
    )

    if use_kg_reasoning:
        reasoned_facts, reasoning_strategy = kg_reasoning_engine.reason(
            classified_q,
            entities,
            predicates
        )
    else:
        reasoned_facts = []
        reasoning_strategy = "vanilla_retrieval"

    retrieval_limit = _determine_retrieval_limit(classified_q)
    derived_result_available = bool(reasoned_facts and reasoned_facts[0].get("_derived_result"))
    should_fetch_raw_kg = not (use_kg_reasoning and derived_result_available)
    if should_fetch_raw_kg:
        kg_facts = kg_retriever.retrieve(
            entities=entities,
            predicates=predicates,
            time_constraint=time_constraint,
            time_semantic=classified_q.time_semantic.name if classified_q.time_semantic else None,
            limit=retrieval_limit
        )
    else:
        kg_facts = []

    kg_found = len(kg_facts) > 0 or len(reasoned_facts) > 0
    if reasoned_facts:
        kg_facts_text = format_reasoned_facts(reasoned_facts, reasoning_strategy)
    else:
        kg_facts_text = kg_retriever.format_facts_for_prompt(kg_facts)

    return {
        "parsed": parsed,
        "entities": entities,
        "predicates": predicates,
        "time_constraint": time_constraint,
        "classified_q": classified_q,
        "reasoned_facts": reasoned_facts,
        "reasoning_strategy": reasoning_strategy,
        "kg_facts": kg_facts,
        "derived_result_available": derived_result_available,
        "kg_found": kg_found,
        "kg_facts_text": kg_facts_text,
        "retrieval_limit": retrieval_limit,
    }


def _has_comprehensive_kg_context(context):
    """Decide whether KG context is strong enough to use."""
    classified_q = context["classified_q"]
    entities = context["entities"]
    kg_facts = context["kg_facts"]
    reasoned_facts = context["reasoned_facts"]

    if reasoned_facts and reasoned_facts[0].get("_derived_result"):
        derived_entities = reasoned_facts[0].get("entities", [])
        return len(derived_entities) > 0

    if reasoned_facts:
        effective_facts = reasoned_facts
    else:
        effective_facts = kg_facts

    relevant_entities_in_facts = set()
    relevant_subjects_in_facts = set()
    for fact in effective_facts:
        relevant_subjects_in_facts.add(fact.get("subject", "").lower())

    for entity in entities:
        if any(entity.lower() in subject for subject in relevant_subjects_in_facts):
            relevant_entities_in_facts.add(entity)

    if classified_q.primary_type in {QuestionType.COUNT, QuestionType.ENTITY, QuestionType.BOOLEAN}:
        return len(effective_facts) >= 1

    if LogicalModifier.COMPARISON in classified_q.logical_modifiers:
        return len(effective_facts) >= 2 and len(relevant_entities_in_facts) >= 2

    return (len(effective_facts) >= 2 and len(relevant_entities_in_facts) >= 2) or (len(effective_facts) >= 3)


def _determine_retrieval_limit(classified_q):
    """Choose a compact but sufficient raw KG retrieval limit by question shape."""
    if LogicalModifier.FILTER in classified_q.logical_modifiers:
        return 4
    if LogicalModifier.ORDERING in classified_q.logical_modifiers:
        return 4
    if LogicalModifier.COMPARISON in classified_q.logical_modifiers:
        return 3
    if classified_q.primary_type == QuestionType.BOOLEAN:
        return 3
    if classified_q.primary_type in {QuestionType.COUNT, QuestionType.ENTITY}:
        return 1
    if classified_q.primary_type == QuestionType.MULTI_FIELD:
        return 4
    return 4


def _should_use_kg_context(classified_q):
    """Whether this question shape should use KG context when available."""
    return classified_q.primary_type != QuestionType.LIST or bool(classified_q.logical_modifiers)


def _query_models(question, kg_facts_text=None, time_constraint=None):
    """Query both models either with KG context or without it."""
    if kg_facts_text is not None:
        openai_answer = ask_openai_with_kg(question, kg_facts_text, time_constraint)
        gemini_answer = ask_gemini_with_kg(question, kg_facts_text, time_constraint)
    else:
        openai_answer = ask_openai(question)
        gemini_answer = ask_gemini(question)
    return openai_answer, gemini_answer


def _answer_single_question(question, question_classifier, kg_retriever, kg_reasoning_engine,
                            time_semantic_override=None, time_constraint_override=None):
    """Answer one question using the same KG-or-fallback flow as the main benchmark."""
    context = _prepare_question_context(
        question,
        question_classifier,
        kg_retriever,
        kg_reasoning_engine,
        time_semantic_override=time_semantic_override,
        time_constraint_override=time_constraint_override,
    )
    classified_q = context["classified_q"]
    should_use_kg = _should_use_kg_context(classified_q)

    if should_use_kg and context["kg_found"] and _has_comprehensive_kg_context(context):
        openai_answer, gemini_answer = _query_models(
            question,
            kg_facts_text=context["kg_facts_text"],
            time_constraint=context["time_constraint"],
        )
    else:
        openai_answer, gemini_answer = _query_models(question)

    return context, openai_answer, gemini_answer


def _resolve_multifield_subquestions(question, classified_q):
    """Prefer LLM-generated split questions, but fall back to deterministic field templates."""
    fallback_questions = [field.sub_question for field in classified_q.fields if field.sub_question]
    if not fallback_questions:
        return [], "none"

    try:
        split_questions = split_multifield_question(question, expected_parts=len(fallback_questions))
    except Exception:
        split_questions = []

    if len(split_questions) == len(fallback_questions):
        return split_questions, "llm_splitter"
    return fallback_questions, "template_fallback"


def _get_vanilla_row(vanilla_data, qid):
    """Return a normalized vanilla-results row with None defaults."""
    return vanilla_data.get(qid) or {
        "openai_answer": None,
        "gemini_answer": None,
        "openai_is_correct": None,
        "gemini_is_correct": None,
        "openai_reason": None,
        "gemini_reason": None,
    }


def _answer_multifield_question(question, classified_q, time_constraint, question_classifier,
                                kg_retriever, kg_reasoning_engine):
    """Resolve, answer, and merge a multi-field question through sub-questions."""
    openai_parts = []
    gemini_parts = []
    field_strategies = []
    sub_kg_found = False
    resolved_sub_questions, multifield_split_source = _resolve_multifield_subquestions(question, classified_q)

    for field_index, field_spec in enumerate(classified_q.fields):
        sub_question = resolved_sub_questions[field_index] if field_index < len(resolved_sub_questions) else field_spec.sub_question
        sub_context, openai_part, gemini_part = _answer_single_question(
            sub_question,
            question_classifier,
            kg_retriever,
            kg_reasoning_engine,
            time_semantic_override=classified_q.time_semantic if field_spec.time_aware else None,
            time_constraint_override=time_constraint if field_spec.time_aware else None,
        )
        openai_parts.append(openai_part)
        gemini_parts.append(gemini_part)
        sub_kg_found = sub_kg_found or sub_context["kg_found"]
        field_strategies.append({
            "field": field_spec.name,
            "sub_question": sub_question,
            "strategy": sub_context["reasoning_strategy"],
            "primary_type": sub_context["classified_q"].primary_type.name,
        })

    return {
        "openai_answer": ", ".join(openai_parts),
        "gemini_answer": ", ".join(gemini_parts),
        "reasoning_strategy": json.dumps(field_strategies, ensure_ascii=False),
        "multifield_split_source": multifield_split_source,
        "kg_found": sub_kg_found,
        "kg_facts_text": "MULTI_FIELD_SPLIT",
        "gemini_calls": len(classified_q.fields),
    }


def _build_result_row(q, ground_truth, context, vanilla_row, openai_kg_ans, gemini_kg_ans,
                      openai_kg_eval, gemini_kg_eval, reasoning_strategy, multifield_split_source, kg_found):
    """Build one benchmark result row."""
    classified_q = context["classified_q"]
    primary_type = classified_q.primary_type.name if classified_q.primary_type else "UNKNOWN"
    time_semantic = classified_q.time_semantic.name if classified_q.time_semantic else "NONE"
    logical_modifiers = [modifier.name for modifier in classified_q.logical_modifiers]
    openai_vanilla_is_correct = vanilla_row["openai_is_correct"]
    gemini_vanilla_is_correct = vanilla_row["gemini_is_correct"]

    openai_improved = None if openai_vanilla_is_correct is None else int(openai_kg_eval["is_correct"]) - int(bool(openai_vanilla_is_correct))
    gemini_improved = None if gemini_vanilla_is_correct is None else int(gemini_kg_eval["is_correct"]) - int(bool(gemini_vanilla_is_correct))

    return {
        "id": q["id"],
        "question": q["question"],
        "kind": q["answer_spec"]["kind"],
        "type": q.get("type", ""),
        "ground_truth": ground_truth,
        "primary_type": primary_type,
        "time_semantic": time_semantic,
        "logical_modifiers": json.dumps(logical_modifiers, ensure_ascii=False),
        "reasoning_strategy": reasoning_strategy,
        "multifield_split_source": multifield_split_source,
        "kg_found": kg_found,
        "kg_facts_count": len(context["kg_facts"]),
        "retrieval_limit": context["retrieval_limit"],
        "parsed_entities": json.dumps(context["entities"], ensure_ascii=False),
        "parsed_predicates": json.dumps(context["predicates"], ensure_ascii=False),
        "time_constraint": context["time_constraint"],
        "openai_vanilla_answer": vanilla_row["openai_answer"],
        "gemini_vanilla_answer": vanilla_row["gemini_answer"],
        "openai_vanilla_is_correct": openai_vanilla_is_correct,
        "gemini_vanilla_is_correct": gemini_vanilla_is_correct,
        "openai_kg_answer": openai_kg_ans,
        "gemini_kg_answer": gemini_kg_ans,
        "openai_kg_is_correct": openai_kg_eval["is_correct"],
        "gemini_kg_is_correct": gemini_kg_eval["is_correct"],
        "openai_vanilla_vs_kg": "same" if vanilla_row["openai_answer"] == openai_kg_ans else "different",
        "gemini_vanilla_vs_kg": "same" if vanilla_row["gemini_answer"] == gemini_kg_ans else "different",
        "openai_improved": openai_improved,
        "gemini_improved": gemini_improved,
        "openai_vanilla_reason": vanilla_row["openai_reason"],
        "gemini_vanilla_reason": vanilla_row["gemini_reason"],
        "openai_kg_reason": openai_kg_eval["reason"],
        "gemini_kg_reason": gemini_kg_eval["reason"],
    }


def _print_benchmark_summary(total_questions, kg_found_count, vanilla_reused_count,
                             openai_vanilla_correct, gemini_vanilla_correct,
                             openai_kg_correct, gemini_kg_correct, results):
    """Print benchmark summary with exact improvement aggregation."""
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY - KG-Integrated Results")
    print("=" * 80)
    print(f"\nTotal questions: {total_questions}")
    print(f"KG facts found for: {kg_found_count}/{total_questions} questions ({(kg_found_count/total_questions)*100:.1f}%)")

    print("\n--- VANILLA LLM (No KG) ---")
    if vanilla_reused_count > 0:
        print(f"Vanilla results reused for {vanilla_reused_count} questions from results/results.csv.")
        print(f"OpenAI  -> Correct: {openai_vanilla_correct}/{vanilla_reused_count} ({(openai_vanilla_correct/vanilla_reused_count)*100:.2f}%)")
        print(f"Gemini  -> Correct: {gemini_vanilla_correct}/{vanilla_reused_count} ({(gemini_vanilla_correct/vanilla_reused_count)*100:.2f}%)")
    else:
        print("No vanilla results were reused.")

    print("\n--- KG-GROUNDED LLM ---")
    print(f"OpenAI  -> Correct: {openai_kg_correct}/{total_questions} ({(openai_kg_correct/total_questions)*100:.2f}%)")
    print(f"Gemini  -> Correct: {gemini_kg_correct}/{total_questions} ({(gemini_kg_correct/total_questions)*100:.2f}%)")

    print("\n--- IMPROVEMENT ANALYSIS ---")
    if vanilla_reused_count > 0:
        openai_improvement = sum((row["openai_improved"] or 0) for row in results if row["openai_improved"] is not None)
        gemini_improvement = sum((row["gemini_improved"] or 0) for row in results if row["gemini_improved"] is not None)
        print(f"OpenAI exact improvement: {openai_improvement:+d} ({(openai_improvement/vanilla_reused_count)*100:+.2f}%)")
        print(f"Gemini exact improvement: {gemini_improvement:+d} ({(gemini_improvement/vanilla_reused_count)*100:+.2f}%)")
    else:
        print("No vanilla results available for improvement analysis.")

    print("\nResults saved to: results/results_with_kg.csv")
    print("=" * 80)


def run_kg_benchmark():
    """Run benchmark with KG integration and advanced question reasoning."""
    with open("data/qa_92.json") as f:
        questions = json.load(f)

    total_questions = len(questions)
    vanilla_data = _load_vanilla_results()
    print(f"Loaded {len(vanilla_data)} vanilla results from results/results.csv for reuse.")

    kg_retriever = KGRetriever()
    question_classifier = QuestionClassifier()
    kg_reasoning_engine = KGReasoningEngine()
    results = []
    Path("results").mkdir(exist_ok=True)

    gemini_call_counter = 0
    openai_vanilla_correct = 0
    gemini_vanilla_correct = 0
    vanilla_reused_count = 0
    openai_kg_correct = 0
    gemini_kg_correct = 0
    kg_found_count = 0

    print(f"Starting KG-integrated benchmark ({total_questions} questions)")
    print("=" * 80)

    for index, q in enumerate(questions, start=1):
        qid = q["id"]
        question = q["question"]
        ground_truth = _serialize_ground_truth(q["answer_spec"])

        print(f"[{index}/{total_questions}] Q{qid}: {question[:60]}...")

        context = _prepare_question_context(question, question_classifier, kg_retriever, kg_reasoning_engine)
        classified_q = context["classified_q"]
        reasoning_strategy = context["reasoning_strategy"]
        kg_found = context["kg_found"]
        if kg_found:
            kg_found_count += 1

        vanilla_row = _get_vanilla_row(vanilla_data, qid)

        if classified_q.primary_type == QuestionType.MULTI_FIELD and classified_q.fields:
            multifield_result = _answer_multifield_question(
                question,
                classified_q,
                context["time_constraint"],
                question_classifier,
                kg_retriever,
                kg_reasoning_engine,
            )
            openai_kg_ans = multifield_result["openai_answer"]
            gemini_kg_ans = multifield_result["gemini_answer"]
            reasoning_strategy = multifield_result["reasoning_strategy"]
            multifield_split_source = multifield_result["multifield_split_source"]
            kg_found = multifield_result["kg_found"]
            if kg_found and not context["kg_found"]:
                kg_found_count += 1
            gemini_call_counter += multifield_result["gemini_calls"]
        else:
            multifield_split_source = "not_multifield"
            should_use_kg = _should_use_kg_context(classified_q)
            if should_use_kg and kg_found and _has_comprehensive_kg_context(context):
                openai_kg_ans, gemini_kg_ans = _query_models(
                    question,
                    kg_facts_text=context["kg_facts_text"],
                    time_constraint=context["time_constraint"],
                )
            else:
                openai_kg_ans, gemini_kg_ans = _query_models(question)
            gemini_call_counter += 1

        openai_kg_eval = evaluate_answer(q, openai_kg_ans)
        gemini_kg_eval = evaluate_answer(q, gemini_kg_ans)

        openai_kg_correct += int(bool(openai_kg_eval["is_correct"]))
        gemini_kg_correct += int(bool(gemini_kg_eval["is_correct"]))
        if qid in vanilla_data:
            vanilla_reused_count += 1
            openai_vanilla_correct += int(bool(vanilla_row["openai_is_correct"]))
            gemini_vanilla_correct += int(bool(vanilla_row["gemini_is_correct"]))

        results.append(
            _build_result_row(
                q,
                ground_truth,
                context,
                vanilla_row,
                openai_kg_ans,
                gemini_kg_ans,
                openai_kg_eval,
                gemini_kg_eval,
                reasoning_strategy,
                multifield_split_source,
                kg_found,
            )
        )

        if gemini_call_counter > 0 and gemini_call_counter % 4 == 0:
            print("  [Waiting 60s for Gemini rate limit]")
            time.sleep(60)

    df = pd.DataFrame(results)
    df.to_csv("results/results_with_kg.csv", index=False)

    _print_benchmark_summary(
        total_questions,
        kg_found_count,
        vanilla_reused_count,
        openai_vanilla_correct,
        gemini_vanilla_correct,
        openai_kg_correct,
        gemini_kg_correct,
        results,
    )


if __name__ == "__main__":
    run_kg_benchmark()
