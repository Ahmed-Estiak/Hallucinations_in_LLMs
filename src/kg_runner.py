"""
KG Runner: Benchmark with KG-grounded LLM answers
Runs in parallel with vanilla LLM benchmark (does not modify existing code)
Outputs results to results/results_with_kg.csv
Integrates Advanced KG Reasoning System with question classification
"""
import json
import time
import sys
from pathlib import Path

# Add workspace root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

from src.question_parser import parse_question
from src.kg_retriever import KGRetriever
from src.kg_models import ask_openai_with_kg, ask_gemini_with_kg
from src.evaluator import evaluate_answer
from src.question_classifier import QuestionClassifier
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
                "openai_answer": row.get("openai_answer", pd.NA),
                "gemini_answer": row.get("gemini_answer", pd.NA),
                "openai_is_correct": row.get("openai_is_correct", pd.NA),
                "gemini_is_correct": row.get("gemini_is_correct", pd.NA),
                "openai_reason": row.get("openai_reason", pd.NA),
                "gemini_reason": row.get("gemini_reason", pd.NA),
            }
        except Exception:
            continue  # Skip invalid rows

    return vanilla_data


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

        # Step 1: Parse question
        parsed = parse_question(question)
        entities = parsed["entities"]
        predicates = parsed["predicates"]
        time_constraint = parsed["time_constraint"]

        # Step 1.5: Classify question for advanced reasoning
        classified_q = question_classifier.classify(question)
        primary_type = classified_q.primary_type.name if classified_q.primary_type else "UNKNOWN"
        time_semantic = classified_q.time_semantic.name if classified_q.time_semantic else "NONE"
        
        # Step 2: Retrieve KG facts via advanced reasoning engine
        # Apply reasoning only for question types where it's proven to help
        use_kg_reasoning = classified_q.primary_type.name in ["MULTI_FIELD"]
        
        if use_kg_reasoning:
            reasoned_facts, reasoning_strategy = kg_reasoning_engine.reason(
                classified_q, 
                entities, 
                predicates
            )
        else:
            # For other types, use standard retrieval
            reasoned_facts = []
            reasoning_strategy = "vanilla_retrieval"
        
        # Also retrieve raw facts for comparison
        kg_facts = kg_retriever.retrieve(
            entities=entities,
            predicates=predicates,
            time_constraint=time_constraint,
            limit=3
        )
        
        kg_found = len(kg_facts) > 0
        if kg_found:
            kg_found_count += 1

        # Use reasoned facts if available (from advanced reasoning), otherwise use raw retrieval
        if reasoned_facts:
            kg_facts_text = format_reasoned_facts(reasoned_facts, reasoning_strategy)
        else:
            kg_facts_text = kg_retriever.format_facts_for_prompt(kg_facts)

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
        # Strategy: Selective KG usage based on question type and fact comprehensiveness
        # ENTITY_LIST questions are excluded because KG retrieval can't handle category queries
        # (e.g., "planets" isn't present as entity - needs expansion logic)
        # Use ask_openai_with_kg for consistency, even with empty_facts for non-KG cases
        
        should_use_kg = not (classified_q.primary_type.name == "ENTITY_LIST")
        
        if should_use_kg and kg_found:
            # Check if facts are comprehensive enough
            relevant_entities_in_facts = set()
            relevant_subjects_in_facts = set()
            for f in kg_facts:
                relevant_subjects_in_facts.add(f.get('subject', '').lower())
            
            for entity in entities:
                if any(entity.lower() in subject for subject in relevant_subjects_in_facts):
                    relevant_entities_in_facts.add(entity)
            
            # Use KG only if retrieval found multiple relevant facts
            has_comprehensive_facts = (len(kg_facts) >= 2 and len(relevant_entities_in_facts) >= 2) or \
                                     (len(kg_facts) >= 3)
            
            if has_comprehensive_facts:
                # We have relevant KG facts
                openai_kg_ans = ask_openai_with_kg(question, kg_facts_text, time_constraint)
                gemini_kg_ans = ask_gemini_with_kg(question, kg_facts_text, time_constraint)
            else:
                # Facts are incomplete or irrelevant - use without KG facts
                openai_kg_ans = ask_openai_with_kg(question, "No relevant KG facts available.", "")
                gemini_kg_ans = ask_gemini_with_kg(question, "No relevant KG facts available.", "")
        else:
            # ENTITY_LIST or no facts found - use without KG facts (vanilla prompt)
            openai_kg_ans = ask_openai_with_kg(question, "No relevant KG facts available.", "")
            gemini_kg_ans = ask_gemini_with_kg(question, "No relevant KG facts available.", "")

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
            "reasoning_strategy": reasoning_strategy,
            
            # KG retrieval info
            "kg_found": kg_found,
            "kg_facts_count": len(kg_facts),
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
