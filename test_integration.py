"""
Quick integration test to verify Advanced KG Reasoning System is properly integrated
"""
import json
from pathlib import Path
from src.question_parser import parse_question
from src.question_classifier import QuestionClassifier
from src.kg_reasoning_engine import KGReasoningEngine, format_reasoned_facts
from src.kg_retriever import KGRetriever

def test_integration():
    """Quick integration test with first 3 questions."""
    
    # Load questions
    with open("data/qa_92.json") as f:
        questions = json.load(f)[:3]  # Test with first 3 questions only
    
    # Initialize components
    classifier = QuestionClassifier()
    reasoning_engine = KGReasoningEngine()
    kg_retriever = KGRetriever()
    
    print("=" * 80)
    print("INTEGRATION TEST: Advanced KG Reasoning System")
    print("=" * 80)
    
    for i, q in enumerate(questions, start=1):
        qid = q["id"]
        question = q["question"]
        
        print(f"\n[{i}] Q{qid}: {question[:70]}...")
        
        # Step 1: Parse question
        parsed = parse_question(question)
        entities = parsed["entities"]
        predicates = parsed["predicates"]
        time_constraint = parsed["time_constraint"]
        print(f"   Parsed: entities={entities}, predicates={predicates}, time={time_constraint}")
        
        # Step 2: Classify question
        classified = classifier.classify(question)
        print(f"   Classification: primary_type={classified.primary_type.name if classified.primary_type else 'None'}")
        print(f"                   time_semantic={classified.time_semantic.name if classified.time_semantic else 'None'}")
        print(f"                   has_time={classified.has_time_constraint}")
        print(f"                   multi_field={classified.is_multi_field}")
        print(f"                   confidence={classified.confidence:.2f}")
        
        # Step 3: Apply reasoning
        reasoned_facts, strategy = reasoning_engine.reason(classified, entities, predicates)
        print(f"   Reasoning: strategy={strategy}, facts_count={len(reasoned_facts)}")
        
        # Step 4: Format facts for prompt
        if reasoned_facts:
            formatted = format_reasoned_facts(reasoned_facts, strategy)
            print(f"   Formatted facts:\n      {formatted[:150]}..." if len(formatted) > 150 else f"   Formatted facts:\n      {formatted}")
        
        # Step 5: Retrieve raw facts for comparison
        raw_facts = kg_retriever.retrieve(entities, predicates, time_constraint, limit=3)
        print(f"   Raw retrieval: {len(raw_facts)} facts")
    
    print("\n" + "=" * 80)
    print("INTEGRATION TEST: PASSED ✓")
    print("=" * 80)

if __name__ == "__main__":
    test_integration()
