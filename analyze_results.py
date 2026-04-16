"""
Analyze KG benchmark results to identify issues
"""
import pandas as pd
import json

df = pd.read_csv('results/results_with_kg.csv')

# Find questions where OpenAI got worse
print("=" * 80)
print("OPENAI PERFORMANCE DEGRADATION")
print("=" * 80)

for idx, row in df.iterrows():
    if not pd.isna(row['openai_vanilla_is_correct']) and not pd.isna(row['openai_kg_is_correct']):
        vanilla = bool(row['openai_vanilla_is_correct'])
        kg = bool(row['openai_kg_is_correct'])
        if vanilla and not kg:  # Was correct, now wrong
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
    if not pd.isna(row['openai_vanilla_is_correct']) and not pd.isna(row['openai_kg_is_correct']):
        vanilla = bool(row['openai_vanilla_is_correct'])
        kg = bool(row['openai_kg_is_correct'])
        if not vanilla and kg:  # Was wrong, now correct
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

for qtype in df['primary_type'].unique():
    subset = df[df['primary_type'] == qtype]
    vanilla_correct = (subset['openai_vanilla_is_correct'] == True).sum()
    kg_correct = (subset['openai_kg_is_correct'] == True).sum()
    count = len(subset)
    vanilla_pct = (vanilla_correct/count)*100 if count > 0 else 0
    kg_pct = (kg_correct/count)*100 if count > 0 else 0
    print(f"\n{qtype:15} ({count} questions)")
    print(f"  Vanilla: {vanilla_correct}/{count} ({vanilla_pct:.1f}%)")
    print(f"  KG:      {kg_correct}/{count} ({kg_pct:.1f}%)")
    print(f"  Change:  {kg_pct - vanilla_pct:+.1f}%")
