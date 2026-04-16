"""
KG Runner: Benchmark with KG-grounded LLM answers
Runs in parallel with vanilla LLM benchmark (does not modify existing code)
Outputs results to results/results_with_kg.csv
"""
import json
import time
from pathlib import Path

import pandas as pd

from src.question_parser import parse_question
from src.kg_retriever import KGRetriever
from src.kg_models import ask_openai_with_kg, ask_gemini_with_kg, ask_openai_vanilla, ask_gemini_vanilla
from src.evaluator import evaluate_answer


def _serialize_ground_truth(answer_spec):
    """Serialize ground truth from answer spec (same as original runner)."""
    if "value" in answer_spec:
        return json.dumps(answer_spec["value"], ensure_ascii=False)

    if "fields" in answer_spec:
        return json.dumps(answer_spec["fields"], ensure_ascii=False)

    return ""


def run_kg_benchmark():
    """Run benchmark with KG integration."""
    
    # Initialize
    with open("data/qa_92.json") as f:
        questions = json.load(f)

    kg_retriever = KGRetriever()
    results = []
    Path("results").mkdir(exist_ok=True)

    gemini_counter = 0
    total_questions = len(questions)
    
    # Counters for vanilla LLM
    openai_vanilla_correct = 0
    gemini_vanilla_correct = 0
    
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

        # Step 2: Retrieve KG facts
        kg_facts = kg_retriever.retrieve(
            entities=entities,
            predicates=predicates,
            time_constraint=time_constraint,
            limit=5
        )
        
        kg_found = len(kg_facts) > 0
        if kg_found:
            kg_found_count += 1

        kg_facts_text = kg_retriever.format_facts_for_prompt(kg_facts)

        # Step 3: Get vanilla LLM answers (for comparison)
        openai_vanilla_ans = ask_openai_vanilla(question)
        gemini_vanilla_ans = ask_gemini_vanilla(question)

        # Step 4: Get KG-grounded LLM answers
        if kg_found:
            openai_kg_ans = ask_openai_with_kg(question, kg_facts_text)
            gemini_kg_ans = ask_gemini_with_kg(question, kg_facts_text)
        else:
            # If no KG found, use vanilla answer (same as LLM-only)
            openai_kg_ans = openai_vanilla_ans
            gemini_kg_ans = gemini_vanilla_ans

        # Step 5: Evaluate all answers
        openai_vanilla_eval = evaluate_answer(q, openai_vanilla_ans)
        gemini_vanilla_eval = evaluate_answer(q, gemini_vanilla_ans)
        openai_kg_eval = evaluate_answer(q, openai_kg_ans)
        gemini_kg_eval = evaluate_answer(q, gemini_kg_ans)

        # Update counters
        openai_vanilla_correct += int(bool(openai_vanilla_eval["is_correct"]))
        gemini_vanilla_correct += int(bool(gemini_vanilla_eval["is_correct"]))
        openai_kg_correct += int(bool(openai_kg_eval["is_correct"]))
        gemini_kg_correct += int(bool(gemini_kg_eval["is_correct"]))

        # Build result row
        result_row = {
            "id": qid,
            "question": question,
            "kind": kind,
            "type": question_type,
            "ground_truth": ground_truth,
            
            # KG retrieval info
            "kg_found": kg_found,
            "kg_facts_count": len(kg_facts),
            "parsed_entities": json.dumps(entities, ensure_ascii=False),
            "parsed_predicates": json.dumps(predicates, ensure_ascii=False),
            "time_constraint": time_constraint,
            
            # Vanilla LLM answers
            "openai_vanilla_answer": openai_vanilla_ans,
            "gemini_vanilla_answer": gemini_vanilla_ans,
            "openai_vanilla_is_correct": openai_vanilla_eval["is_correct"],
            "gemini_vanilla_is_correct": gemini_vanilla_eval["is_correct"],
            
            # KG-grounded LLM answers
            "openai_kg_answer": openai_kg_ans,
            "gemini_kg_answer": gemini_kg_ans,
            "openai_kg_is_correct": openai_kg_eval["is_correct"],
            "gemini_kg_is_correct": gemini_kg_eval["is_correct"],
            
            # Comparison: did KG help?
            "openai_vanilla_vs_kg": "same" if openai_vanilla_ans == openai_kg_ans else "different",
            "gemini_vanilla_vs_kg": "same" if gemini_vanilla_ans == gemini_kg_ans else "different",
            "openai_improved": int(openai_kg_eval["is_correct"]) - int(openai_vanilla_eval["is_correct"]),
            "gemini_improved": int(gemini_kg_eval["is_correct"]) - int(gemini_vanilla_eval["is_correct"]),
            
            # Evaluation reasons
            "openai_vanilla_reason": openai_vanilla_eval["reason"],
            "gemini_vanilla_reason": gemini_vanilla_eval["reason"],
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
    print(
        f"OpenAI  → Correct: {openai_vanilla_correct}/{total_questions} "
        f"({(openai_vanilla_correct/total_questions)*100:.2f}%)"
    )
    print(
        f"Gemini  → Correct: {gemini_vanilla_correct}/{total_questions} "
        f"({(gemini_vanilla_correct/total_questions)*100:.2f}%)"
    )
    
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
    openai_improvement = openai_kg_correct - openai_vanilla_correct
    gemini_improvement = gemini_kg_correct - gemini_vanilla_correct
    
    print(f"OpenAI improvement: {openai_improvement:+d} ({(openai_improvement/total_questions)*100:+.2f}%)")
    print(f"Gemini improvement: {gemini_improvement:+d} ({(gemini_improvement/total_questions)*100:+.2f}%)")
    
    print("\nResults saved to: results/results_with_kg.csv")
    print("=" * 80)


if __name__ == "__main__":
    run_kg_benchmark()
