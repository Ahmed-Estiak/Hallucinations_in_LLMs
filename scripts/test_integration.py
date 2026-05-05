"""
Quick integration test for the Advanced KG Reasoning System.

This script runs a small slice of the benchmark through parsing,
classification, KG reasoning, prompt formatting, and raw KG retrieval. It is
intended as a fast manual check that the main pipeline components still work
together.
"""
import json
from pathlib import Path
from src.question_parser import parse_question
from src.question_classifier import QuestionClassifier
from src.kg_reasoning_engine import KGReasoningEngine, format_reasoned_facts
from src.kg_retriever import KGRetriever


def test_integration():
    """
    Run the integration check against the first three benchmark questions.

    The test is intentionally small so it can be run quickly while still
    touching the parser, classifier, reasoning engine, formatter, and retriever.
    """
    
    # Load only the first three questions to keep this integration check fast
    # and readable in the terminal.
    with open("data/qa_92.json") as f:
        questions = json.load(f)[:3]  # Test with first 3 questions only
    
    # Initialize the pipeline components that should work together end-to-end.
    classifier = QuestionClassifier()
    reasoning_engine = KGReasoningEngine()
    kg_retriever = KGRetriever()
    
    print("=" * 80)
    print("INTEGRATION TEST: Advanced KG Reasoning System")
    print("=" * 80)
    
    for i, q in enumerate(questions, start=1):
        # Extract the question id and text for display and downstream pipeline
        # calls.
        qid = q["id"]
        question = q["question"]
        
        print(f"\n[{i}] Q{qid}: {question[:70]}...")
        
        # Step 1: parse the question into entities, predicates, and an optional
        # time constraint. These signals feed both reasoning and raw retrieval.
        parsed = parse_question(question)
        entities = parsed["entities"]
        predicates = parsed["predicates"]
        time_constraint = parsed["time_constraint"]
        print(f"   Parsed: entities={entities}, predicates={predicates}, time={time_constraint}")
        
        # Step 2: classify the question so the reasoning engine knows which
        # strategy and metadata apply.
        classified = classifier.classify(question)
        print(f"   Classification: primary_type={classified.primary_type.name if classified.primary_type else 'None'}")
        print(f"                   time_semantic={classified.time_semantic.name if classified.time_semantic else 'None'}")
        print(f"                   has_time={classified.has_time_constraint}")
        print(f"                   multi_field={classified.is_multi_field}")
        print(f"                   confidence={classified.confidence:.2f}")
        
        # Step 3: run KG reasoning using both the classification result and the
        # parser output.
        reasoned_facts, strategy = reasoning_engine.reason(classified, entities, predicates)
        print(f"   Reasoning: strategy={strategy}, facts_count={len(reasoned_facts)}")
        
        # Step 4: format reasoned facts exactly as they would be inserted into
        # a model prompt.
        if reasoned_facts:
            formatted = format_reasoned_facts(reasoned_facts, strategy)
            print(f"   Formatted facts:\n      {formatted[:150]}..." if len(formatted) > 150 else f"   Formatted facts:\n      {formatted}")
        
        # Step 5: retrieve raw KG facts for comparison with the reasoned facts.
        raw_facts = kg_retriever.retrieve(entities, predicates, time_constraint, limit=3)
        print(f"   Raw retrieval: {len(raw_facts)} facts")
    
    print("\n" + "=" * 80)
    print("INTEGRATION TEST: PASSED ✓")
    print("=" * 80)

if __name__ == "__main__":
    test_integration()
