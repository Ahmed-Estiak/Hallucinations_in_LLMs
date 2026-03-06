import json
import time
import pandas as pd

from src.models import ask_openai, ask_gemini
from src.evaluator import evaluate_answer


def run_benchmark():

    with open("data/qa_92.json") as f:
        questions = json.load(f)

    results = []

    gemini_counter = 0

    for q in questions[:15]:

        qid = q["id"]
        question = q["question"]

        print("Running:", qid)

        openai_ans = ask_openai(question)

        gemini_ans = ask_gemini(question)

        openai_correct = evaluate_answer(q, openai_ans)
        gemini_correct = evaluate_answer(q, gemini_ans)

        results.append({

            "id": qid,
            "question": question,
            "openai_answer": openai_ans,
            "gemini_answer": gemini_ans,
            "openai_correct": openai_correct,
            "gemini_correct": gemini_correct
        })

        gemini_counter += 1

        if gemini_counter % 4 == 0:

            print("Waiting 60 seconds for Gemini rate limit")
            time.sleep(60)

    df = pd.DataFrame(results)

    df.to_csv("results/results.csv", index=False)

    print("Benchmark finished")