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


def _is_time_sensitive_factual_query(classified_q):
    """Whether the answer should come from the latest temporal snapshot even without an explicit time."""
    time_sensitive_predicates = {
        "moon_count",
        "discovered_on",
        "discovered_by",
    }
    return any(predicate in time_sensitive_predicates for predicate in classified_q.major_predicates)


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
    should_use_kg = classified_q.primary_type != QuestionType.LIST or bool(classified_q.logical_modifiers)

    if should_use_kg and context["kg_found"] and _has_comprehensive_kg_context(context):
        openai_answer = ask_openai_with_kg(question, context["kg_facts_text"], context["time_constraint"])
        gemini_answer = ask_gemini_with_kg(question, context["kg_facts_text"], context["time_constraint"])
    else:
        openai_answer = ask_openai(question)
        gemini_answer = ask_gemini(question)

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


def run_kg_benchmark():
    """Run benchmark with KG integration and advanced question reasoning."""
    
    # Initialize
    with open("data/qa_92.json") as f:
        questions = json.load(f)

    total_questions = len(questions)
    vanilla_data = _load_vanilla_results()
    loaded_vanilla_count = len(vanilla_data)
    print(f"Loaded {loaded_vanilla_count} vanilla results from results/results.csv for reuse.")

    kg_retriever = KGRetriever()
    question_classifier = QuestionClassifier()
    kg_reasoning_engine = KGReasoningEngine()
    results = []
    Path("results").mkdir(exist_ok=True)

    gemini_counter = 0
    
    # Counters for vanilla LLM stored values
    openai_vanilla_correct = 0
    gemini_vanilla_correct = 0
    vanilla_reused_count = 0
    
    # Counters for KG-grounded LLM
    openai_kg_correct = 0
    gemini_kg_correct = 0
    
    # Counters for KG retrieval stats
    kg_found_count = 0

    print(f"Starting KG-integrated benchmark ({total_questions} questions)")
    print("=" * 80)

    for index, q in enumerate(questions, start=1):
        qid = q["id"]
        question = q["question"]
        kind = q["answer_spec"]["kind"]
        question_type = q.get("type", "")
        ground_truth = _serialize_ground_truth(q["answer_spec"])

        print(f"[{index}/{total_questions}] Q{qid}: {question[:60]}...")

        # Step 1: Parse/classify question and prepare KG context
        context = _prepare_question_context(question, question_classifier, kg_retriever, kg_reasoning_engine)
        parsed = context["parsed"]
        entities = context["entities"]
        predicates = context["predicates"]
        time_constraint = context["time_constraint"]
        classified_q = context["classified_q"]
        primary_type = classified_q.primary_type.name if classified_q.primary_type else "UNKNOWN"
        time_semantic = classified_q.time_semantic.name if classified_q.time_semantic else "NONE"
        logical_modifiers = [modifier.name for modifier in classified_q.logical_modifiers]
        reasoned_facts = context["reasoned_facts"]
        reasoning_strategy = context["reasoning_strategy"]
        kg_facts = context["kg_facts"]
        derived_result_available = context["derived_result_available"]
        kg_found = context["kg_found"]
        if kg_found:
            kg_found_count += 1

        kg_facts_text = context["kg_facts_text"]

        # Step 3: Retrieve vanilla LLM answers from previous results.csv if valid
        vanilla_row = vanilla_data.get(qid, None)
        if vanilla_row is not None:
            openai_vanilla_ans = vanilla_row["openai_answer"]
            gemini_vanilla_ans = vanilla_row["gemini_answer"]
            openai_vanilla_is_correct = vanilla_row["openai_is_correct"]
            gemini_vanilla_is_correct = vanilla_row["gemini_is_correct"]
            openai_vanilla_reason = vanilla_row["openai_reason"]
            gemini_vanilla_reason = vanilla_row["gemini_reason"]
        else:
            openai_vanilla_ans = pd.NA
            gemini_vanilla_ans = pd.NA
            openai_vanilla_is_correct = pd.NA
            gemini_vanilla_is_correct = pd.NA
            openai_vanilla_reason = pd.NA
            gemini_vanilla_reason = pd.NA

        # Step 4: Get LLM answers
        if classified_q.primary_type == QuestionType.MULTI_FIELD and classified_q.fields:
            openai_parts = []
            gemini_parts = []
            field_strategies = []
            total_sub_kg_facts = 0
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
                total_sub_kg_facts += len(sub_context["kg_facts"])
                sub_kg_found = sub_kg_found or sub_context["kg_found"]
                field_strategies.append({
                    "field": field_spec.name,
                    "sub_question": sub_question,
                    "strategy": sub_context["reasoning_strategy"],
                    "primary_type": sub_context["classified_q"].primary_type.name,
                })

            openai_kg_ans = ", ".join(openai_parts)
            gemini_kg_ans = ", ".join(gemini_parts)
            reasoning_strategy = json.dumps(field_strategies, ensure_ascii=False)
            kg_found = sub_kg_found
            kg_facts_text = "MULTI_FIELD_SPLIT"
            if kg_found and not context["kg_found"]:
                kg_found_count += 1
            kg_facts = kg_facts[:]
            if total_sub_kg_facts > len(kg_facts):
                pass
        else:
            multifield_split_source = "not_multifield"
            should_use_kg = classified_q.primary_type != QuestionType.LIST or bool(classified_q.logical_modifiers)

            if should_use_kg and kg_found and _has_comprehensive_kg_context(context):
                openai_kg_ans = ask_openai_with_kg(question, kg_facts_text, time_constraint)
                gemini_kg_ans = ask_gemini_with_kg(question, kg_facts_text, time_constraint)
            else:
                openai_kg_ans = ask_openai(question)
                gemini_kg_ans = ask_gemini(question)

        # Step 5: Evaluate KG-grounded answers only
        openai_kg_eval = evaluate_answer(q, openai_kg_ans)
        gemini_kg_eval = evaluate_answer(q, gemini_kg_ans)

        # Update counters
        openai_kg_correct += int(bool(openai_kg_eval["is_correct"]))
        gemini_kg_correct += int(bool(gemini_kg_eval["is_correct"]))
        if vanilla_row is not None:
            vanilla_reused_count += 1
            openai_vanilla_correct += int(bool(openai_vanilla_is_correct))
            gemini_vanilla_correct += int(bool(gemini_vanilla_is_correct))

        # Build result row
        result_row = {
            "id": qid,
            "question": question,
            "kind": kind,
            "type": question_type,
            "ground_truth": ground_truth,
            
            # Question classification
            "primary_type": primary_type,
            "time_semantic": time_semantic,
            "logical_modifiers": json.dumps(logical_modifiers, ensure_ascii=False),
            "reasoning_strategy": reasoning_strategy,
            "multifield_split_source": multifield_split_source,
            
            # KG retrieval info
            "kg_found": kg_found,
            "kg_facts_count": len(kg_facts),
            "retrieval_limit": context["retrieval_limit"],
            "parsed_entities": json.dumps(entities, ensure_ascii=False),
            "parsed_predicates": json.dumps(predicates, ensure_ascii=False),
            "time_constraint": time_constraint,
            
            # Vanilla LLM answers from previous results.csv
            "openai_vanilla_answer": openai_vanilla_ans,
            "gemini_vanilla_answer": gemini_vanilla_ans,
            "openai_vanilla_is_correct": openai_vanilla_is_correct,
            "gemini_vanilla_is_correct": gemini_vanilla_is_correct,
            
            # KG-grounded LLM answers
            "openai_kg_answer": openai_kg_ans,
            "gemini_kg_answer": gemini_kg_ans,
            "openai_kg_is_correct": openai_kg_eval["is_correct"],
            "gemini_kg_is_correct": gemini_kg_eval["is_correct"],
            
            # Comparison: did KG help?
            "openai_vanilla_vs_kg": "same" if openai_vanilla_ans == openai_kg_ans else "different",
            "gemini_vanilla_vs_kg": "same" if gemini_vanilla_ans == gemini_kg_ans else "different",
            "openai_improved": (
                int(openai_kg_eval["is_correct"]) - int(bool(openai_vanilla_is_correct))
                if not pd.isna(openai_vanilla_is_correct) else pd.NA
            ),
            "gemini_improved": (
                int(gemini_kg_eval["is_correct"]) - int(bool(gemini_vanilla_is_correct))
                if not pd.isna(gemini_vanilla_is_correct) else pd.NA
            ),
            
            # Evaluation reasons
            "openai_vanilla_reason": openai_vanilla_reason,
            "gemini_vanilla_reason": gemini_vanilla_reason,
            "openai_kg_reason": openai_kg_eval["reason"],
            "gemini_kg_reason": gemini_kg_eval["reason"],
        }

        results.append(result_row)
        
        gemini_counter += 1
        
        # Rate limit handling
        if gemini_counter % 4 == 0:
            print("  [Waiting 60s for Gemini rate limit]")
            time.sleep(60)

    # Save results
    df = pd.DataFrame(results)
    df.to_csv("results/results_with_kg.csv", index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY - KG-Integrated Results")
    print("=" * 80)
    print(f"\nTotal questions: {total_questions}")
    print(f"KG facts found for: {kg_found_count}/{total_questions} questions ({(kg_found_count/total_questions)*100:.1f}%)")
    
    print("\n--- VANILLA LLM (No KG) ---")
    if vanilla_reused_count > 0:
        print(f"Vanilla results reused for {vanilla_reused_count} questions from results/results.csv.")
        print(
            f"OpenAI  → Correct: {openai_vanilla_correct}/{vanilla_reused_count} "
            f"({(openai_vanilla_correct/vanilla_reused_count)*100:.2f}%)"
        )
        print(
            f"Gemini  → Correct: {gemini_vanilla_correct}/{vanilla_reused_count} "
            f"({(gemini_vanilla_correct/vanilla_reused_count)*100:.2f}%)"
        )
    else:
        print("No vanilla results were reused.")
    
    print("\n--- KG-GROUNDED LLM ---")
    print(
        f"OpenAI  → Correct: {openai_kg_correct}/{total_questions} "
        f"({(openai_kg_correct/total_questions)*100:.2f}%)"
    )
    print(
        f"Gemini  → Correct: {gemini_kg_correct}/{total_questions} "
        f"({(gemini_kg_correct/total_questions)*100:.2f}%)"
    )
    
    print("\n--- IMPROVEMENT ANALYSIS ---")
    if vanilla_reused_count > 0:
        # Improvement calculated only for questions where vanilla results were reused
        openai_improvement = openai_kg_correct - openai_vanilla_correct  # Note: kg_correct is total, but improvement is approximate
        gemini_improvement = gemini_kg_correct - gemini_vanilla_correct
        print(f"OpenAI improvement (approx): {openai_improvement:+d} ({(openai_improvement/vanilla_reused_count)*100:+.2f}%)")
        print(f"Gemini improvement (approx): {gemini_improvement:+d} ({(gemini_improvement/vanilla_reused_count)*100:+.2f}%)")
    else:
        print("No vanilla results available for improvement analysis.")
    
    print("\nResults saved to: results/results_with_kg.csv")
    print("=" * 80)


if __name__ == "__main__":
    run_kg_benchmark()
