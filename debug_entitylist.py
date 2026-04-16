"""
Debug ENTITY_LIST questions to see what KG facts are returned
"""
import json
from src.question_classifier import QuestionClassifier
from src.kg_retriever import KGRetriever
from src.question_parser import parse_question

# Load questions
with open('data/qa_92.json') as f:
    questions = json.load(f)

# Check Q11 and Q12
for qid in [11, 12]:
    q = next(q for q in questions if q['id'] == qid)
    question = q['question']
    ground_truth = q['answer_spec']['value']
    
    print("\n" + "=" * 80)
    print(f"Q{qid}: {question}")
    print(f"Ground Truth: {ground_truth}")
    print("=" * 80)
    
    # Parse
    parsed = parse_question(question)
    entities = parsed['entities']
    predicates = parsed['predicates']
    time_constraint = parsed['time_constraint']
    
    print(f"Parsed entities: {entities}")
    print(f"Parsed predicates: {predicates}")
    print(f"Time constraint: {time_constraint}")
    
    # Classify
    classifier = QuestionClassifier()
    classified = classifier.classify(question)
    print(f"\nClassified as: {classified.primary_type.name}")
    
    # Retrieve with vanilla retriever
    kg_retriever = KGRetriever()
    facts = kg_retriever.retrieve(entities, predicates, time_constraint, limit=3)
    
    print(f"\nRetrieved {len(facts)} facts:")
    for f in facts:
        print(f"  - {f.get('subject')} | {f.get('predicate')} | {f.get('object')} ({f.get('time', 'unknown')})")
    
    formatted = kg_retriever.format_facts_for_prompt(facts)
    print(f"\nFormatted for prompt:\n{formatted}")
