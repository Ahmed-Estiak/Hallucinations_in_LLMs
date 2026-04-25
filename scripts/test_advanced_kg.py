"""
Advanced KG System Tester: Demonstrates question classification and reasoning
"""
from src.question_classifier import QuestionClassifier
from src.kg_reasoning_engine import KGReasoningEngine, format_reasoned_facts
from src.question_parser import parse_question


def test_advanced_kg_system():
    """Test the advanced KG classification and reasoning system."""
    
    classifier = QuestionClassifier()
    reasoning_engine = KGReasoningEngine()
    
    # Test questions covering all types and scenarios
    test_questions = [
        # Boolean questions
        "As of 2022, did the IAU recognize Pluto as the dwarf planet with the largest diameter?",
        
        # Entity questions
        "Which dwarf planet located in the Kuiper Belt was discovered first?",
        
        # Entity list with filtering
        "Which planets orbit beyond Earth yet have fewer moons than Jupiter?",
        
        # Ordered list
        "List the terrestrial planets in order of decreasing mass.",
        
        # Comparison
        "Between Neptune and Uranus, which planet has the greater mass?",
        
        # Count with time constraint
        "As of November 2022, how many confirmed moons orbit Jupiter?",
        
        # Time lookup with multi-field
        "In what year was Neptune discovered, and who discovered it?",
        
        # Before constraint (historical)
        "Before 1980, how many of Saturn's moons had been officially confirmed?",
    ]
    
    print("=" * 80)
    print("ADVANCED KG CLASSIFICATION AND REASONING SYSTEM TEST")
    print("=" * 80)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[{i}] Question: {question}")
        print("-" * 80)
        
        # Step 1: Classify question
        classified = classifier.classify(question)
        parsed = parse_question(question)
        
        print(f"Primary Type: {classified.primary_type.value}")
        if classified.secondary_types:
            print(f"Secondary Types: {[t.value for t in classified.secondary_types]}")
        print(f"Confidence: {classified.confidence:.2f}")
        
        # Time information
        if classified.has_time_constraint:
            print(f"Time Constraint: {classified.time_semantic.value} '{classified.time_value}'")
        
        # Multi-field information
        if classified.is_multi_field:
            print(f"Multi-field: YES (predicates: {classified.multi_field_predicates})")
        
        # Logic operators
        if classified.logic_operator.value != "none":
            print(f"Logic Operator: {classified.logic_operator.value}")
        
        # Special attributes
        if classified.ordering_attribute:
            print(f"Ordering: by '{classified.ordering_attribute}' ({classified.order_direction})")
        if classified.comparison_operator:
            print(f"Comparison: {classified.comparison_operator}")
        
        # Step 2: Apply reasoning
        print(f"\nParsed Entities: {parsed['entities']}")
        print(f"Parsed Predicates: {parsed['predicates']}")
        
        facts, strategy = reasoning_engine.reason(
            classified,
            parsed['entities'],
            parsed['predicates']
        )
        
        print(f"\nReasoning Strategy: {strategy}")
        print(f"Facts Retrieved: {len(facts)}")
        
        if facts:
            print("\nRetrieved Facts:")
            for j, fact in enumerate(facts[:3], 1):
                print(f"  {j}. {fact.get('subject')} | {fact.get('predicate')} = {fact.get('object')} (as of {fact.get('time')})")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    test_advanced_kg_system()
