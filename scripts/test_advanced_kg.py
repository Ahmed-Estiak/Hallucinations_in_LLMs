"""
Advanced KG System Tester: Demonstrates question classification and reasoning.

This script runs a small set of representative questions through the advanced
KG pipeline and prints how each question is classified, parsed, and reasoned
over.
"""
from src.question_classifier import QuestionClassifier
from src.kg_reasoning_engine import KGReasoningEngine, format_reasoned_facts
from src.question_parser import parse_question


def test_advanced_kg_system():
    """
    Test the advanced KG classification and reasoning system.

    The output is intentionally verbose because this script is used for manual
    inspection: it shows the classifier decision, parser output, reasoning
    strategy, and a preview of retrieved facts for each question.
    """
    
    # Create one classifier and one reasoning engine for the whole test run so
    # every sample question goes through the same pipeline components.
    classifier = QuestionClassifier()
    reasoning_engine = KGReasoningEngine()
    
    # Test questions covering multiple question types and KG reasoning scenarios.
    test_questions = [
        # Boolean question: checks yes/no classification and date-sensitive facts.
        "As of 2022, did the IAU recognize Pluto as the dwarf planet with the largest diameter?",
        
        # Entity question: asks for a single entity matching a factual condition.
        "Which dwarf planet located in the Kuiper Belt was discovered first?",
        
        # Entity list with filtering: requires applying multiple constraints.
        "Which planets orbit beyond Earth yet have fewer moons than Jupiter?",
        
        # Ordered list: tests whether the system can sort entities by an attribute.
        "List the terrestrial planets in order of decreasing mass.",
        
        # Comparison: compares two named entities using a numeric attribute.
        "Between Neptune and Uranus, which planet has the greater mass?",
        
        # Count with time constraint: requires the answer at a specific date.
        "As of November 2022, how many confirmed moons orbit Jupiter?",
        
        # Time lookup with multi-field answer: asks for both date and discoverer.
        "In what year was Neptune discovered, and who discovered it?",
        
        # Historical count: uses a "before" time constraint.
        "Before 1980, how many of Saturn's moons had been officially confirmed?",
    ]
    
    print("=" * 80)
    print("ADVANCED KG CLASSIFICATION AND REASONING SYSTEM TEST")
    print("=" * 80)
    
    for i, question in enumerate(test_questions, 1):
        # Print a clear boundary for each sample question so the debug output is
        # easy to scan in the terminal.
        print(f"\n[{i}] Question: {question}")
        print("-" * 80)
        
        # Step 1: classify the question and parse it into KG retrieval signals.
        classified = classifier.classify(question)
        parsed = parse_question(question)
        
        print(f"Primary Type: {classified.primary_type.value}")
        if classified.secondary_types:
            print(f"Secondary Types: {[t.value for t in classified.secondary_types]}")
        print(f"Confidence: {classified.confidence:.2f}")
        
        # Print time-related metadata only when the classifier found a time
        # constraint in the question.
        if classified.has_time_constraint:
            print(f"Time Constraint: {classified.time_semantic.value} '{classified.time_value}'")
        
        # Multi-field questions need more than one answer field, such as year
        # plus discoverer.
        if classified.is_multi_field:
            print(f"Multi-field: YES (predicates: {classified.multi_field_predicates})")
        
        # Logic operators identify questions that combine conditions, such as
        # "and", "or", or comparative filtering.
        if classified.logic_operator.value != "none":
            print(f"Logic Operator: {classified.logic_operator.value}")
        
        # Special attributes capture ordering and comparison requirements.
        if classified.ordering_attribute:
            print(f"Ordering: by '{classified.ordering_attribute}' ({classified.order_direction})")
        if classified.comparison_operator:
            print(f"Comparison: {classified.comparison_operator}")
        
        # Step 2: show the parser output that will be used by the KG reasoning
        # engine.
        print(f"\nParsed Entities: {parsed['entities']}")
        print(f"Parsed Predicates: {parsed['predicates']}")
        
        # Run the KG reasoning engine with the classifier result and parser
        # signals. The strategy explains which reasoning path was selected.
        facts, strategy = reasoning_engine.reason(
            classified,
            parsed['entities'],
            parsed['predicates']
        )
        
        print(f"\nReasoning Strategy: {strategy}")
        print(f"Facts Retrieved: {len(facts)}")
        
        # Print only the first few facts so the test remains readable while
        # still showing whether retrieval returned relevant evidence.
        if facts:
            print("\nRetrieved Facts:")
            for j, fact in enumerate(facts[:3], 1):
                print(f"  {j}. {fact.get('subject')} | {fact.get('predicate')} = {fact.get('object')} (as of {fact.get('time')})")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    test_advanced_kg_system()
