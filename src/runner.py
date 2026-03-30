import json
import time
from pathlib import Path

import pandas as pd

from src.models import ask_openai, ask_gemini
from src.evaluator import evaluate_answer


def run_benchmark():

    with open("data/qa_92.json") as f:
        questions = json.load(f)

    results = []
    Path("results").mkdir(exist_ok=True)

    gemini_counter = 0

    for q in questions:

        qid = q["id"]
        question = q["question"]
        kind = q["answer_spec"]["kind"]

        print("Running:", qid)

        openai_ans = ask_openai(question)

        gemini_ans = ask_gemini(question)

        openai_eval = evaluate_answer(q, openai_ans)
        gemini_eval = evaluate_answer(q, gemini_ans)

        results.append({
            "id": qid,
            "question": question,
            "kind": kind,
            "openai_answer": openai_ans,
            "gemini_answer": gemini_ans,
            "openai_is_correct": openai_eval["is_correct"],
            "openai_manual_check": openai_eval["manual_check"],
            "openai_reason": openai_eval["reason"],
            "openai_eval": json.dumps(openai_eval, ensure_ascii=False),
            "gemini_is_correct": gemini_eval["is_correct"],
            "gemini_manual_check": gemini_eval["manual_check"],
            "gemini_reason": gemini_eval["reason"],
            "gemini_eval": json.dumps(gemini_eval, ensure_ascii=False),
        })

        gemini_counter += 1

        if gemini_counter % 4 == 0:

            print("Waiting 60 seconds for Gemini rate limit")
            time.sleep(60)

    df = pd.DataFrame(results)

    df.to_csv("results/results.csv", index=False)

    print("Benchmark finished")
