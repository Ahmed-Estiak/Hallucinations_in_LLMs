"""
Debug specific question reasoning
"""
import json
from src.question_classifier import QuestionClassifier
from src.kg_reasoning_engine import KGReasoningEngine
from src.question_parser import parse_question

# Initialize
classifier = QuestionClassifier()
reasoning_engine = KGReasoningEngine()

# Load questions
with open('data/qa_92.json') as f:
    questions = json.load(f)

# Check Q11
q11 = next(q for q in questions if q['id'] == 11)
question = q11['question']
ground_truth = q11['answer_spec']['value']

print("=" * 80)
print(f"Q11: {question}")
print(f"Ground Truth: {ground_truth}")
print("=" * 80)

# Parse
parsed = parse_question(question)
entities = parsed['entities']
predicates = parsed['predicates']
print(f"\nParsed entities: {entities}")
print(f"Parsed predicates: {predicates}")

# Classify
classified = classifier.classify(question)
print(f"\nClassified as: {classified.primary_type.name}")
print(f"Secondary types: {[t.name for t in classified.secondary_types]}")

# Reason
reasoned_facts, strategy = reasoning_engine.reason(classified, entities, predicates)
print(f"\nReasoning strategy: {strategy}")
print(f"Reasoned facts returned: {len(reasoned_facts)}")

# Show facts
if reasoned_facts:
    print("\nReturned Facts:")
    for i, fact in enumerate(reasoned_facts, 1):
        print(f"  {i}. {fact}")

# Check raw KG
print("\n" + "=" * 80)
print("Checking raw KG for entity filtering details")
print("=" * 80)

# Load and check KG directly
with open('data/astronomy_kg1.json') as f:
    kg = json.load(f)

# Look for relevant facts
print("\nFacts in KG for 'Mars', 'Uranus', 'Neptune':")
for entity_name in ['Mars', 'Uranus', 'Neptune']:
    matching = [f for f in kg if f.get('subject', '').lower() == entity_name.lower()]
    print(f"\n{entity_name}: {len(matching)} facts")
    for fact in matching[:3]:
        print(f"  - {fact.get('subject')} | {fact.get('predicate')} | {fact.get('object')}")
