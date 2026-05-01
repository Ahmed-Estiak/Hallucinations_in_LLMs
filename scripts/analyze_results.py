"""
Analyze KG benchmark results to identify issues.

The script compares vanilla OpenAI answers against KG-assisted OpenAI answers
and prints regressions, improvements, and grouped accuracy summaries.
"""
import pandas as pd


# Load the benchmark result table. Each row contains one question, the vanilla
# model answer, the KG-assisted answer, and correctness labels for both.
df = pd.read_csv("results/results_with_kg.csv")

# Find questions where OpenAI got worse after KG context was added.
print("=" * 80)
print("OPENAI PERFORMANCE DEGRADATION")
print("=" * 80)

for idx, row in df.iterrows():
    # Skip rows where either correctness label is missing, because those rows
    # cannot be compared safely.
    if not pd.isna(row["openai_vanilla_is_correct"]) and not pd.isna(
        row["openai_kg_is_correct"]
    ):
        vanilla = bool(row["openai_vanilla_is_correct"])
        kg = bool(row["openai_kg_is_correct"])

        # Regression means the vanilla answer was correct but the KG-assisted
        # answer became wrong.
        if vanilla and not kg:
            print(f"\nQ{row['id']} - REGRESSED (was correct, now wrong)")
            print(f"  Question: {row['question'][:70]}...")
            print(f"  Primary Type: {row['primary_type']}")
            print(f"  Time Semantic: {row['time_semantic']}")
            print(f"  KG Found: {row['kg_found']} ({int(row['kg_facts_count'])} facts)")
            print(f"  Vanilla Answer: {str(row['openai_vanilla_answer'])[:80]}")
            print(f"  KG Answer: {str(row['openai_kg_answer'])[:80]}")
            print(f"  Ground Truth: {str(row['ground_truth'])[:80]}")

print("\n" + "=" * 80)
print("OPENAI PERFORMANCE IMPROVEMENTS")
print("=" * 80)

for idx, row in df.iterrows():
    # Only compare rows where both correctness labels are present.
    if not pd.isna(row["openai_vanilla_is_correct"]) and not pd.isna(
        row["openai_kg_is_correct"]
    ):
        vanilla = bool(row["openai_vanilla_is_correct"])
        kg = bool(row["openai_kg_is_correct"])

        # Improvement means the vanilla answer was wrong but the KG-assisted
        # answer became correct.
        if not vanilla and kg:
            print(f"\nQ{row['id']} - IMPROVED (was wrong, now correct)")
            print(f"  Question: {row['question'][:70]}...")
            print(f"  Primary Type: {row['primary_type']}")
            print(f"  Time Semantic: {row['time_semantic']}")
            print(f"  Reasoning Strategy: {row['reasoning_strategy']}")
            print(f"  KG Found: {row['kg_found']} ({int(row['kg_facts_count'])} facts)")
            print(f"  Vanilla Answer: {str(row['openai_vanilla_answer'])[:80]}")
            print(f"  KG Answer: {str(row['openai_kg_answer'])[:80]}")

print("\n" + "=" * 80)
print("SUMMARY STATS BY QUESTION TYPE")
print("=" * 80)

for qtype in df["primary_type"].unique():
    # Group rows by the manually/automatically assigned primary question type
    # so accuracy changes can be compared by task category.
    subset = df[df["primary_type"] == qtype]
    vanilla_correct = (subset["openai_vanilla_is_correct"] == True).sum()
    kg_correct = (subset["openai_kg_is_correct"] == True).sum()
    count = len(subset)

    # Avoid division by zero even though every group from unique() should have
    # at least one row.
    vanilla_pct = (vanilla_correct / count) * 100 if count > 0 else 0
    kg_pct = (kg_correct / count) * 100 if count > 0 else 0

    print(f"\n{qtype:15} ({count} questions)")
    print(f"  Vanilla: {vanilla_correct}/{count} ({vanilla_pct:.1f}%)")
    print(f"  KG:      {kg_correct}/{count} ({kg_pct:.1f}%)")
    print(f"  Change:  {kg_pct - vanilla_pct:+.1f}%")
