import json
import time
from pathlib import Path
import sys

import pandas as pd

# Add workspace root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import ask_openai, ask_gemini
from src.evaluator import evaluate_answer


def _serialize_ground_truth(answer_spec):
    if "value" in answer_spec:
        return json.dumps(answer_spec["value"], ensure_ascii=False)

    if "fields" in answer_spec:
        return json.dumps(answer_spec["fields"], ensure_ascii=False)

    return ""


def run_benchmark():
    start_time = time.time()

    with open("data/qa_92.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    Path("results").mkdir(exist_ok=True)

    gemini_counter = 0
    total_questions = len(questions)
    openai_correct_count = 0
    gemini_correct_count = 0
    openai_manual_check_count = 0
    gemini_manual_check_count = 0

    for index, q in enumerate(questions, start=1):

        qid = q["id"]
        question = q["question"]
        kind = q["answer_spec"]["kind"]
        question_type = q.get("type", "")
        ground_truth = _serialize_ground_truth(q["answer_spec"])

        print(f"Running: {qid} ({index}/{total_questions})")

        openai_ans = ask_openai(question)

        gemini_ans = ask_gemini(question)

        openai_eval = evaluate_answer(q, openai_ans)
        gemini_eval = evaluate_answer(q, gemini_ans)

        openai_correct_count += int(bool(openai_eval["is_correct"]))
        gemini_correct_count += int(bool(gemini_eval["is_correct"]))
        openai_manual_check_count += int(bool(openai_eval["manual_check"]))
        gemini_manual_check_count += int(bool(gemini_eval["manual_check"]))

        results.append({
            "id": qid,
            "question": question,
            "kind": kind,
            "type": question_type,
            "ground_truth": ground_truth,
            "openai_answer": openai_ans,
            "gemini_answer": gemini_ans,
            "openai_is_correct": openai_eval["is_correct"],
            "gemini_is_correct": gemini_eval["is_correct"],
            "openai_manual_check": openai_eval["manual_check"],
            "gemini_manual_check": gemini_eval["manual_check"],
            "openai_reason": openai_eval["reason"],
            "gemini_reason": gemini_eval["reason"],
            "openai_eval": json.dumps(openai_eval, ensure_ascii=False),
            "gemini_eval": json.dumps(gemini_eval, ensure_ascii=False),
        })

        gemini_counter += 1

        if gemini_counter % 4 == 0 and index < total_questions:

            print("Waiting 60 seconds for Gemini rate limit")
            time.sleep(60)

    df = pd.DataFrame(results)

    df.to_csv("results/results.csv", index=False)

    print("Final summary")
    print(
        "OpenAI -> "
        f"correct: {openai_correct_count}/{total_questions} "
        f"({(openai_correct_count / total_questions) * 100:.2f}%), "
        f"manual_check: {openai_manual_check_count}"
    )
    print(
        "Gemini -> "
        f"correct: {gemini_correct_count}/{total_questions} "
        f"({(gemini_correct_count / total_questions) * 100:.2f}%), "
        f"manual_check: {gemini_manual_check_count}"
    )
    elapsed_seconds = time.time() - start_time
    print(f"Total runtime: {elapsed_seconds:.2f} seconds ({elapsed_seconds / 60:.2f} minutes)")
    print("Benchmark finished")


if __name__ == "__main__":
    run_benchmark()
