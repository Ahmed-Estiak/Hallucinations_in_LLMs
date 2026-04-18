"""
Advanced Question Classifier and Type Analyzer
Categorizes questions with semantic understanding of time, logic operators, and multi-field requirements
"""
import re
from typing import Dict, List, Tuple, Optional
from enum import Enum

from src.question_parser import TIME_TOKEN_PATTERN, TIME_RANGE_SEPARATOR, _normalize_time_token


class QuestionType(Enum):
    """Primary question types ordered by reasoning priority."""
    BOOLEAN = "boolean"           # Highest priority: Yes/No questions
    ENTITY = "entity"             # Direct entity lookup
    ENTITY_LIST = "entity_list"   # Multiple entities with filtering
    ORDERED_LIST = "ordered_list" # Sorted list by attribute
    COMPARISON = "comparison"      # Comparative (greater, less, etc)
    COUNT = "count"                # Numerical aggregation
    TIME_LOOKUP = "time_lookup"    # Historical fact lookup
    MULTI_FIELD = "multi_field"   # Multiple predicates combined


class TimeSemantic(Enum):
    """How to interpret time constraint."""
    EXACT = "exact"               # "As of 2022" - exact time snapshot
    BEFORE = "before"             # "Before 1980" - any time <= constraint
    AFTER = "after"               # "After 2000" - any time >= constraint
    BETWEEN = "between"           # Range
    NONE = "none"                 # No time constraint


class LogicOperator(Enum):
    """How to combine multiple conditions."""
    AND = "and"
    OR = "or"
    NONE = "none"


class ClassifiedQuestion:
    """Result of question classification."""
    
    def __init__(self):
        self.primary_type: QuestionType = QuestionType.ENTITY
        self.secondary_types: List[QuestionType] = []
        self.has_time_constraint: bool = False
        self.time_semantic: TimeSemantic = TimeSemantic.NONE
        self.time_value: Optional[str] = None
        self.is_multi_field: bool = False
        self.multi_field_predicates: List[str] = []
        self.logic_operator: LogicOperator = LogicOperator.NONE
        self.ordering_attribute: Optional[str] = None  # For ordered_list
        self.order_direction: str = "ascending"  # ascending or descending
        self.comparison_operator: Optional[str] = None  # >, <, ==, etc
        self.entity_filter_conditions: List[Dict] = []  # For entity_list
        self.boolean_keyword: Optional[str] = None  # yes/no/true/false etc
        self.confidence: float = 1.0
        self.type_scores: Dict[QuestionType, float] = {}


class QuestionClassifier:
    """
    Classifies questions into types with advanced semantic understanding.
    Handles time semantics, multi-field, logic operators, and priorities.
    """
    
    # Question type indicators (keyword patterns)
    TYPE_PATTERNS = {
        QuestionType.BOOLEAN: {
            "patterns": [
                r"^(?:is|are|was|were)\b",
                r"^(?:does|did)\b",
                r"\b(?:does|did)\s+.*\b(?:have|orbit|orbits|recognize|recognized|discover|discovered|contain)\b",
                r"\b(?:is|are|was|were)\s+.*\b(?:a|an|the)\b",
                r"^(is|are|was|were|does|did)",
            ],
            "keywords": ["recognize", "recognized"]
        },
        QuestionType.ENTITY: {
            "patterns": [
                r"which\s+(?:dwarf\s+)?planet",
                r"who\s+(?:discovered|found)",
                r"what\s+(?:planet|dwarf planet|object|body)",
                r"name\s+(?:the|a)\s+(?:planet|dwarf planet|object|body)",
            ],
            "keywords": ["what planet", "what dwarf", "who discovered", "who found"]
        },
        QuestionType.ENTITY_LIST: {
            "patterns": [
                r"which\s+planets?",
                r"list.*planets?",
                r"names?\s+.*planets?",
                r"(?:all|multiple)\s+(?:planets?|dwarfs?|moons?|satellites?)",
            ],
            "keywords": ["which planets", "all planets", "which moons", "which satellites"]
        },
        QuestionType.ORDERED_LIST: {
            "patterns": [
                r"(?:in\s+)?order\s+(?:of|by)",
                r"(?:decreasing|increasing)\s+",
                r"(?:smallest\s+to\s+largest|largest\s+to\s+smallest)",
                r"list\s+.*\s+(?:in\s+order|ranked)",
                r"(?:rank|sequence)\s+.*by",
            ],
            "keywords": ["order", "decreasing", "increasing", "ranked", "smallest to largest", "largest to smallest"]
        },
        QuestionType.COMPARISON: {
            "patterns": [
                r"which\s+(?:is\s+)?(?:greater|more|larger|heavier|farther|closer)",
                r"(?:between|of)\s+.*which\s+(?:is|has)",
                r"(?:greater|less|more|fewer)\s+than",
                r"compare(?:d)?\s+(?:to|with)?",
            ],
            "keywords": ["between", "compared to", "greater than", "less than", "more than", "fewer than"]
        },
        QuestionType.COUNT: {
            "patterns": [
                r"how\s+many",
                r"(?:count|number)\s+(?:of|to)",
                r"total\s+(?:number\s+)?(?:of)?",
            ],
            "keywords": ["how many", "count", "number", "total"]
        },
        QuestionType.TIME_LOOKUP: {
            "patterns": [
                r"when\s+(?:was|were|is|are|did)",
                r"what\s+year",
                r"in\s+what\s+year",
                r"what\s+date",
                r"what\s+month",
            ],
            "keywords": ["when", "what year", "in what year", "what date", "what month"]
        }
    }
    
    # Time semantic patterns
    TIME_PATTERNS = {
        TimeSemantic.EXACT: [
            rf"(?:as\s+of|by|on|during|in)\s+({TIME_TOKEN_PATTERN})",
        ],
        TimeSemantic.BEFORE: [
            rf"(?:before|prior\s+to|until|up\s+to)\s+({TIME_TOKEN_PATTERN})",
        ],
        TimeSemantic.AFTER: [
            rf"(?:after|since)\s+({TIME_TOKEN_PATTERN})",
        ],
        TimeSemantic.BETWEEN: [
            rf"(?:between|from)\s+({TIME_TOKEN_PATTERN})\s+(?:and|to)\s+({TIME_TOKEN_PATTERN})",
        ]
    }
    
    # Ordering keywords
    ORDERING_KEYWORDS = {
        "mass": r"(?:mass|massive|heaviest|lightest|weight)",
        "distance": r"(?:distance|farther|closer|away)",
        "size": r"(?:size|diameter|largest|smallest|big)",
        "discovered": r"(?:discovered|discovery)",
        "moons": r"(?:moon|moons|satellite)",
    }
    
    # Comparison operators
    COMPARISON_KEYWORDS = {
        ">": ["greater", "more", "larger", "heavier", "farther"],
        "<": ["less", "fewer", "smaller", "lighter", "closer"],
        "==": ["same", "equal", "equal to", "same as"],
    }
    
    # Boolean value keywords
    BOOLEAN_VALUES = ["yes", "no", "true", "false", "is", "was"]
    
    def classify(self, question: str) -> ClassifiedQuestion:
        """
        Classify a question into types with detailed semantic analysis.
        Returns ClassifiedQuestion with all metadata.
        """
        result = ClassifiedQuestion()
        question_lower = question.lower()
        
        # Step 1: Detect time constraint and semantics
        self._detect_time_constraint(question_lower, result)
        
        # Step 2: Detect logic operators (AND, OR)
        self._detect_logic_operators(question_lower, result)
        
        # Step 3: Detect multi-field nature
        self._detect_multi_field(question_lower, result)
        
        # Step 4: Classify primary and secondary types
        self._classify_types(question_lower, result)
        
        # Step 5: Set priority based on rules
        self._set_priority_and_operators(question_lower, result)
        
        # Step 6: Detect special attributes (ordering, comparison, filter)
        self._detect_special_attributes(question_lower, result)

        # Step 7: Normalize secondary types after promotions/overrides
        self._finalize_secondary_types(result)
        
        return result
    
    def _detect_time_constraint(self, question: str, result: ClassifiedQuestion) -> None:
        """Extract time constraint and semantic meaning."""
        for semantic, patterns in self.TIME_PATTERNS.items():
            for pattern in patterns:
                match = re.search(pattern, question)
                if match:
                    result.has_time_constraint = True
                    result.time_semantic = semantic
                    if semantic == TimeSemantic.BETWEEN:
                        start = _normalize_time_token(match.group(1))
                        end = _normalize_time_token(match.group(2))
                        if start and end:
                            result.time_value = f"{start}{TIME_RANGE_SEPARATOR}{end}"
                        else:
                            result.time_value = f"{match.group(1)}{TIME_RANGE_SEPARATOR}{match.group(2)}"
                    else:
                        result.time_value = _normalize_time_token(match.group(1)) or match.group(1)
                    return
    
    def _detect_logic_operators(self, question: str, result: ClassifiedQuestion) -> None:
        """Detect AND/OR logic operators."""
        if re.search(r"\band\b", question):
            result.logic_operator = LogicOperator.AND
        elif re.search(r"\bor\b", question):
            result.logic_operator = LogicOperator.OR
    
    def _detect_multi_field(self, question: str, result: ClassifiedQuestion) -> None:
        """Detect if question has multiple fields (e.g., year AND discoverer)."""
        # Look for patterns like "year ... and ... discoverer"
        if re.search(r"(year|when).*\band\b.*(discoverer|who|by)", question):
            result.is_multi_field = True
            result.multi_field_predicates = ["discovered_on", "discovered_by"]
        elif re.search(r"(discoverer|who).*\band\b.*(year|when)", question):
            result.is_multi_field = True
            result.multi_field_predicates = ["discovered_by", "discovered_on"]
        elif re.search(r"\bwhat\s+(?:year|date|month)\b.*\band\b.*\bwho\b", question):
            result.is_multi_field = True
    
    def _classify_types(self, question: str, result: ClassifiedQuestion) -> None:
        """Classify into question types with confidence scores."""
        type_scores: Dict[QuestionType, float] = {}
        
        for qtype, patterns_dict in self.TYPE_PATTERNS.items():
            score = 0.0
            
            # Pattern matching
            for pattern in patterns_dict["patterns"]:
                if re.search(pattern, question):
                    score += 0.6
                    break
            
            # Keyword matching
            for keyword in patterns_dict["keywords"]:
                if keyword in question:
                    score += 0.2
                    break
            
            type_scores[qtype] = score
        
        result.type_scores = type_scores
        
        # Sort by confidence, set primary and secondary
        sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
        if sorted_types and sorted_types[0][1] > 0:
            result.primary_type = sorted_types[0][0]
            result.confidence = sorted_types[0][1]
            if len(sorted_types) > 1 and sorted_types[1][1] > 0.2:
                result.secondary_types = [t[0] for t in sorted_types[1:3] if t[1] > 0.2]
    
    def _set_priority_and_operators(self, question: str, result: ClassifiedQuestion) -> None:
        """Adjust type priority based on priority rules."""
        # Force numeric/count interpretation for "how many"/count-style questions.
        if re.search(r"\bhow\s+many\b", question) or re.search(r"\b(?:count|number|total)\b", question):
            if result.primary_type != QuestionType.COUNT:
                result.secondary_types.insert(0, result.primary_type)
                result.primary_type = QuestionType.COUNT

        # Boolean has highest priority only for explicit yes/no style questions
        if (
            result.type_scores.get(QuestionType.BOOLEAN, 0) > 0.5 and
            not re.search(r"\bhow\s+many\b", question) and
            not re.search(r"\b(?:count|number|total)\b", question)
        ):
            result.primary_type = QuestionType.BOOLEAN
        
        # Time-sensitive gets high priority if time present
        if result.has_time_constraint:
            if result.primary_type in [QuestionType.COUNT, QuestionType.ENTITY]:
                result.secondary_types.insert(0, result.primary_type)
                result.primary_type = QuestionType.TIME_LOOKUP
        
        # Multi-field overrides simple types
        if result.is_multi_field:
            result.secondary_types.insert(0, result.primary_type)
            result.primary_type = QuestionType.MULTI_FIELD
        
        # ENTITY_LIST priority when asking for multiple entities
        # Detect plural forms and multiple-selection patterns
        if result.primary_type == QuestionType.ENTITY:
            # Check for plural indicators
            if re.search(r"\b(?:planets|dwarfs?|moons|satellites)\b", question):
                if result.type_scores.get(QuestionType.ENTITY_LIST, 0) > 0.2:
                    result.secondary_types.insert(0, result.primary_type)
                    result.primary_type = QuestionType.ENTITY_LIST
            # Check for "multiple", "all", "which ... have" patterns
            if re.search(r"(?:multiple|all|any|several)\s+(?:planets?|dwarfs?|moons?|satellites?)", question):
                if result.type_scores.get(QuestionType.ENTITY_LIST, 0) > 0.2:
                    result.secondary_types.insert(0, result.primary_type)
                    result.primary_type = QuestionType.ENTITY_LIST
    
    def _detect_special_attributes(self, question: str, result: ClassifiedQuestion) -> None:
        """Detect ordering, comparison, and filter attributes."""
        # Ordering attribute
        for attr, pattern in self.ORDERING_KEYWORDS.items():
            if re.search(pattern, question):
                result.ordering_attribute = attr
                if re.search(r"decreasing|most|largest|highest", question):
                    result.order_direction = "descending"
                else:
                    result.order_direction = "ascending"
                break
        
        # Comparison operator
        for op, keywords in self.COMPARISON_KEYWORDS.items():
            for keyword in keywords:
                if keyword in question:
                    result.comparison_operator = op
                    break
            if result.comparison_operator:
                break
        
        # Boolean value
        for value in self.BOOLEAN_VALUES:
            if re.search(rf"\b{re.escape(value)}\b", question):
                result.boolean_keyword = value
                break

    def _finalize_secondary_types(self, result: ClassifiedQuestion) -> None:
        """Remove duplicates, drop primary from secondary list, and keep top two."""
        deduped: List[QuestionType] = []
        for qtype in result.secondary_types:
            if qtype == result.primary_type:
                continue
            if qtype not in deduped:
                deduped.append(qtype)
        result.secondary_types = deduped[:2]
