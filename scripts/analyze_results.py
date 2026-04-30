"""
Analyze KG benchmark results to identify where KG context helps or hurts.

This script reads the benchmark CSV, then prints three views:
1. OpenAI answers that became worse after adding KG context.
2. OpenAI answers that became better after adding KG context.
3. Accuracy summary grouped by question type.
"""
import pandas as pd


RESULTS_PATH = "results/results_with_kg.csv"
SEPARATOR_WIDTH = 80


def print_section_header(title):
    """
    Print a consistent section header before each report block.

    The repeated separator makes terminal output easier to scan when the report
    contains many questions.
    """
    print("=" * SEPARATOR_WIDTH)
    print(title)
    print("=" * SEPARATOR_WIDTH)


def has_openai_correctness_labels(row):
    """
    Check whether both vanilla and KG correctness labels are available.

    Some rows may have missing evaluation values. Those rows are skipped because
    we cannot safely compare vanilla performance against KG performance.
    """
    return (
        not pd.isna(row["openai_vanilla_is_correct"])
        and not pd.isna(row["openai_kg_is_correct"])
    )


def print_answer_comparison(row, include_ground_truth=False):
    """
    Print the main metadata and answer snippets for one benchmark question.

    Long text fields are truncated so each question stays readable in the
    console while still showing enough context to understand the model behavior.
    """
    print(f"  Question: {row['question'][:70]}...")
    print(f"  Primary Type: {row['primary_type']}")
    print(f"  Time Semantic: {row['time_semantic']}")
    print(f"  KG Found: {row['kg_found']} ({int(row['kg_facts_count'])} facts)")
    print(f"  Vanilla Answer: {str(row['openai_vanilla_answer'])[:80]}")
    print(f"  KG Answer: {str(row['openai_kg_answer'])[:80]}")

    if include_ground_truth:
        print(f"  Ground Truth: {str(row['ground_truth'])[:80]}")


def print_openai_regressions(df):
    """
    Print questions where KG context changed a correct vanilla answer to wrong.

    These examples are useful for diagnosing whether retrieved KG facts are
    misleading, incomplete, stale, or being over-weighted by the model.
    """
    print_section_header("OPENAI PERFORMANCE DEGRADATION")

    for _, row in df.iterrows():
        if not has_openai_correctness_labels(row):
            continue

        vanilla = bool(row["openai_vanilla_is_correct"])
        kg = bool(row["openai_kg_is_correct"])

        if vanilla and not kg:
            print(f"\nQ{row['id']} - REGRESSED (was correct, now wrong)")
            print_answer_comparison(row, include_ground_truth=True)


def print_openai_improvements(df):
    """
    Print questions where KG context changed a wrong vanilla answer to correct.

    These examples show where the knowledge graph is helping the model recover
    facts or reasoning steps that the vanilla prompt missed.
    """
    print("\n" + "=" * SEPARATOR_WIDTH)
    print("OPENAI PERFORMANCE IMPROVEMENTS")
    print("=" * SEPARATOR_WIDTH)

    for _, row in df.iterrows():
        if not has_openai_correctness_labels(row):
            continue

        vanilla = bool(row["openai_vanilla_is_correct"])
        kg = bool(row["openai_kg_is_correct"])

        if not vanilla and kg:
            print(f"\nQ{row['id']} - IMPROVED (was wrong, now correct)")
            print(f"  Question: {row['question'][:70]}...")
            print(f"  Primary Type: {row['primary_type']}")
            print(f"  Time Semantic: {row['time_semantic']}")
            print(f"  Reasoning Strategy: {row['reasoning_strategy']}")
            print(f"  KG Found: {row['kg_found']} ({int(row['kg_facts_count'])} facts)")
            print(f"  Vanilla Answer: {str(row['openai_vanilla_answer'])[:80]}")
            print(f"  KG Answer: {str(row['openai_kg_answer'])[:80]}")


def print_summary_by_question_type(df):
    """
    Print vanilla-vs-KG accuracy for every primary question type.

    This grouped view helps reveal whether KG support is generally useful for a
    class of questions, even if individual examples vary.
    """
    print("\n" + "=" * SEPARATOR_WIDTH)
    print("SUMMARY STATS BY QUESTION TYPE")
    print("=" * SEPARATOR_WIDTH)

    for qtype in df["primary_type"].unique():
        subset = df[df["primary_type"] == qtype]
        count = len(subset)

        vanilla_correct = (subset["openai_vanilla_is_correct"] == True).sum()
        kg_correct = (subset["openai_kg_is_correct"] == True).sum()

        vanilla_pct = (vanilla_correct / count) * 100 if count > 0 else 0
        kg_pct = (kg_correct / count) * 100 if count > 0 else 0

        print(f"\n{qtype:15} ({count} questions)")
        print(f"  Vanilla: {vanilla_correct}/{count} ({vanilla_pct:.1f}%)")
        print(f"  KG:      {kg_correct}/{count} ({kg_pct:.1f}%)")
        print(f"  Change:  {kg_pct - vanilla_pct:+.1f}%")


def main():
    """
    Load benchmark results and print all analysis sections.

    Keeping this in main() makes the file import-safe: importing this module
    will not immediately read the CSV or print the report.
    """
    df = pd.read_csv(RESULTS_PATH)

    print_openai_regressions(df)
    print_openai_improvements(df)
    print_summary_by_question_type(df)


if __name__ == "__main__":
    main()
