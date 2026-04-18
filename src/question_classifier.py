"""
Question classifier focused on answer shape plus logical modifiers.
"""
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from src.question_parser import TIME_RANGE_SEPARATOR, TIME_TOKEN_PATTERN, _normalize_time_token


class QuestionType(Enum):
    """Primary answer-shape categories."""
    BOOLEAN = "boolean"
    ENTITY = "entity"
    COUNT = "count"
    LIST = "list"
    MULTI_FIELD = "multi_field"


class LogicalModifier(Enum):
    """Additional reasoning requirements layered on top of answer shape."""
    TIME_LOOKUP = "time_lookup"
    FILTER = "filter"
    ORDERING = "ordering"
    COMPARISON = "comparison"


class TimeSemantic(Enum):
    """How to interpret time constraint."""
    EXACT = "exact"
    BEFORE = "before"
    AFTER = "after"
    BETWEEN = "between"
    NONE = "none"


class LogicOperator(Enum):
    """How to combine multiple conditions."""
    AND = "and"
    OR = "or"
    NONE = "none"


@dataclass
class ClassifiedField:
    """Schema for one field inside a multi-field question."""
    name: str
    answer_type: QuestionType
    predicate: Optional[str]
    time_aware: bool
    logical_modifiers: List[LogicalModifier] = field(default_factory=list)


@dataclass
class ClassifiedQuestion:
    """Result of question classification."""
    original_question: str = ""
    primary_type: QuestionType = QuestionType.ENTITY
    secondary_types: List[QuestionType] = field(default_factory=list)
    logical_modifiers: List[LogicalModifier] = field(default_factory=list)
    has_time_constraint: bool = False
    time_semantic: TimeSemantic = TimeSemantic.NONE
    time_value: Optional[str] = None
    is_multi_field: bool = False
    fields: List[ClassifiedField] = field(default_factory=list)
    multi_field_predicates: List[str] = field(default_factory=list)
    logic_operator: LogicOperator = LogicOperator.NONE
    ordering_attribute: Optional[str] = None
    order_direction: str = "ascending"
    comparison_operator: Optional[str] = None
    list_target: Optional[str] = None
    entity_filter_conditions: List[Dict] = field(default_factory=list)
    boolean_keyword: Optional[str] = None
    confidence: float = 1.0
    type_scores: Dict[QuestionType, float] = field(default_factory=dict)


class QuestionClassifier:
    """Classify answer shape and logical modifiers for a question."""

    TYPE_PATTERNS = {
        QuestionType.BOOLEAN: {
            "patterns": [
                r"^(?:is|are|was|were|does|did)\b",
                r"\b(?:does|did)\s+.*\b(?:have|orbit|orbits|recognize|recognized|discover|discovered|contain)\b",
                r"\b(?:is|are|was|were)\s+.*\b(?:a|an|the)\b",
            ],
            "keywords": ["recognize", "recognized"],
        },
        QuestionType.COUNT: {
            "patterns": [
                r"\bhow\s+many\b",
                r"\b(?:count|number)\s+of\b",
                r"\btotal\s+(?:number\s+)?of\b",
                r"\bwhat\s+was\s+the\s+number\b",
            ],
            "keywords": ["how many", "count of", "number of", "total number"],
        },
        QuestionType.LIST: {
            "patterns": [
                r"\bwhich\s+(?:planets|dwarfs|moons|satellites)\b",
                r"\blist\b",
                r"\bnames?\b.*\b(?:planets|dwarfs|moons|satellites)\b",
                r"\ball\s+(?:planets|dwarfs|moons|satellites)\b",
            ],
            "keywords": ["which planets", "which moons", "all planets", "list the"],
        },
        QuestionType.ENTITY: {
            "patterns": [
                r"\bwhich\s+(?:planet|dwarf planet|object|body)\b",
                r"\bwho\s+(?:discovered|found)\b",
                r"\bwhat\s+type\s+of\s+planet\b",
                r"\bwhat\s+(?:planet|dwarf planet|object|body)\b",
                r"\bname\s+(?:the|a)\s+(?:planet|dwarf planet|object|body)\b",
            ],
            "keywords": ["who discovered", "who found", "what type of planet"],
        },
    }

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
        ],
    }

    ORDERING_PATTERNS = [
        r"\bin\s+order\b",
        r"\border\s+of\b",
        r"\brank(?:ed)?\b",
        r"\bdecreasing\b",
        r"\bincreasing\b",
        r"\bsmallest\s+to\s+largest\b",
        r"\blargest\s+to\s+smallest\b",
    ]

    COMPARISON_PATTERNS = [
        r"\bbetween\b.*\bwhich\b",
        r"\bcompared\s+to\b",
        r"\b(?:greater|less|more|fewer)\s+than\b",
        r"\bwhich\s+is\s+(?:greater|larger|smaller|farther|closer)\b",
    ]

    FILTER_PATTERNS = [
        r"\bfewer\s+than\b",
        r"\bmore\s+than\b",
        r"\bless\s+than\b",
        r"\bgreater\s+than\b",
        r"\blocated\s+in\b",
        r"\bfound\s+in\b",
        r"\bin\s+the\s+kuiper\s+belt\b",
        r"\bin\s+the\s+asteroid\s+belt\b",
        r"\borbit(?:s|ing)?\s+beyond\b",
    ]

    ORDERING_KEYWORDS = {
        "mass": r"(?:mass|massive|heaviest|lightest|weight)",
        "distance": r"(?:distance|farther|closer|away)",
        "size": r"(?:size|diameter|largest|smallest|big)",
        "discovered": r"(?:discovered|discovery)",
        "moons": r"(?:moon|moons|satellite)",
    }

    COMPARISON_KEYWORDS = {
        ">": ["greater", "more", "larger", "heavier", "farther"],
        "<": ["less", "fewer", "smaller", "lighter", "closer"],
        "==": ["same", "equal", "equal to", "same as"],
    }

    BOOLEAN_VALUES = ["yes", "no", "true", "false", "is", "was"]

    def classify(self, question: str) -> ClassifiedQuestion:
        result = ClassifiedQuestion()
        result.original_question = question
        question_lower = question.lower()

        self._detect_time_constraint(question_lower, result)
        self._detect_logic_operators(question_lower, result)
        self._detect_multi_field(question_lower, result)
        self._classify_primary_type(question_lower, result)
        self._detect_logical_modifiers(question_lower, result)
        self._detect_special_attributes(question_lower, result)
        self._finalize(result)

        return result

    def _detect_time_constraint(self, question: str, result: ClassifiedQuestion) -> None:
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
        if re.search(r"\band\b", question):
            result.logic_operator = LogicOperator.AND
        elif re.search(r"\bor\b", question):
            result.logic_operator = LogicOperator.OR

    def _detect_multi_field(self, question: str, result: ClassifiedQuestion) -> None:
        year_like = r"(?:what\s+(?:year|date|month)|in\s+what\s+year|when)"
        discoverer_like = r"(?:who|discoverer|found\s+by|discovered\s+by)"

        first_time = re.search(year_like, question)
        first_discoverer = re.search(discoverer_like, question)
        if first_time and first_discoverer and re.search(r"\band\b", question):
            result.is_multi_field = True
            if first_time.start() < first_discoverer.start():
                result.fields = [
                    ClassifiedField("field1", QuestionType.COUNT, "discovered_on", False, []),
                    ClassifiedField("field2", QuestionType.ENTITY, "discovered_by", False, []),
                ]
                result.multi_field_predicates = ["discovered_on", "discovered_by"]
            else:
                result.fields = [
                    ClassifiedField("field1", QuestionType.ENTITY, "discovered_by", False, []),
                    ClassifiedField("field2", QuestionType.COUNT, "discovered_on", False, []),
                ]
                result.multi_field_predicates = ["discovered_by", "discovered_on"]

    def _classify_primary_type(self, question: str, result: ClassifiedQuestion) -> None:
        if result.is_multi_field:
            result.primary_type = QuestionType.MULTI_FIELD
            result.type_scores = {QuestionType.MULTI_FIELD: 1.0}
            result.confidence = 1.0
            return

        type_scores: Dict[QuestionType, float] = {}
        for qtype, patterns_dict in self.TYPE_PATTERNS.items():
            score = 0.0
            for pattern in patterns_dict["patterns"]:
                if re.search(pattern, question):
                    score += 0.7
                    break
            for keyword in patterns_dict["keywords"]:
                if keyword in question:
                    score += 0.2
                    break
            type_scores[qtype] = score

        if re.search(r"\bhow\s+many\b", question) or re.search(r"\b(?:count|number|total)\b", question):
            type_scores[QuestionType.COUNT] = max(type_scores.get(QuestionType.COUNT, 0.0), 0.9)

        if re.search(r"\b(?:which|list|all)\s+(?:planets|dwarfs|moons|satellites)\b", question):
            type_scores[QuestionType.LIST] = max(type_scores.get(QuestionType.LIST, 0.0), 0.8)

        sorted_types = sorted(type_scores.items(), key=lambda item: item[1], reverse=True)
        result.type_scores = type_scores
        if sorted_types and sorted_types[0][1] > 0:
            result.primary_type = sorted_types[0][0]
            result.confidence = sorted_types[0][1]
            result.secondary_types = [t[0] for t in sorted_types[1:3] if t[1] > 0.2]

    def _detect_logical_modifiers(self, question: str, result: ClassifiedQuestion) -> None:
        if result.has_time_constraint:
            result.logical_modifiers.append(LogicalModifier.TIME_LOOKUP)

        if result.primary_type != QuestionType.BOOLEAN:
            if any(re.search(pattern, question) for pattern in self.ORDERING_PATTERNS):
                result.logical_modifiers.append(LogicalModifier.ORDERING)
            if any(re.search(pattern, question) for pattern in self.COMPARISON_PATTERNS):
                result.logical_modifiers.append(LogicalModifier.COMPARISON)
            if result.primary_type == QuestionType.LIST and (
                any(re.search(pattern, question) for pattern in self.FILTER_PATTERNS) or
                result.logic_operator == LogicOperator.AND
            ):
                result.logical_modifiers.append(LogicalModifier.FILTER)

    def _detect_special_attributes(self, question: str, result: ClassifiedQuestion) -> None:
        for attr, pattern in self.ORDERING_KEYWORDS.items():
            if re.search(pattern, question):
                result.ordering_attribute = attr
                if re.search(r"decreasing|most|largest|highest", question):
                    result.order_direction = "descending"
                else:
                    result.order_direction = "ascending"
                break

        for op, keywords in self.COMPARISON_KEYWORDS.items():
            for keyword in keywords:
                if keyword in question:
                    result.comparison_operator = op
                    break
            if result.comparison_operator:
                break

        for value in self.BOOLEAN_VALUES:
            if re.search(rf"\b{re.escape(value)}\b", question):
                result.boolean_keyword = value
                break

        if result.primary_type == QuestionType.LIST and LogicalModifier.FILTER in result.logical_modifiers:
            if re.search(r"\bfewer\b.*\bthan\b", question) or re.search(r"\bless\b.*\bthan\b", question):
                result.entity_filter_conditions.append({"operator": "<", "attribute": result.ordering_attribute or "unknown"})
            elif re.search(r"\bmore\b.*\bthan\b", question) or re.search(r"\bgreater\b.*\bthan\b", question):
                result.entity_filter_conditions.append({"operator": ">", "attribute": result.ordering_attribute or "unknown"})

        if result.primary_type == QuestionType.LIST:
            if re.search(r"\bplanets\b", question):
                result.list_target = "planets"
            elif re.search(r"\bdwarfs?\b", question):
                result.list_target = "dwarf_planets"
            elif re.search(r"\b(?:moons|satellites)\b", question):
                result.list_target = "moons"

            if re.search(r"\bterrestrial\b", question):
                result.entity_filter_conditions.append({"operator": "==", "attribute": "planet_type", "value": "terrestrial"})
            if re.search(r"\bgas\s+giant\b", question):
                result.entity_filter_conditions.append({"operator": "==", "attribute": "planet_type", "value": "gas giant"})
            if re.search(r"\bice\s+giant\b", question):
                result.entity_filter_conditions.append({"operator": "==", "attribute": "planet_type", "value": "ice giant"})
            if re.search(r"\bin\s+the\s+kuiper\s+belt\b", question):
                result.entity_filter_conditions.append({"operator": "==", "attribute": "location", "value": "Kuiper Belt"})
            if re.search(r"\bin\s+the\s+asteroid\s+belt\b", question):
                result.entity_filter_conditions.append({"operator": "==", "attribute": "location", "value": "Asteroid Belt"})
            if re.search(r"\bbeyond\s+earth\b", question):
                result.entity_filter_conditions.append({"operator": ">", "attribute": "distance_from_sun", "reference_entity": "Earth"})

    def _finalize(self, result: ClassifiedQuestion) -> None:
        deduped_secondaries: List[QuestionType] = []
        for qtype in result.secondary_types:
            if qtype == result.primary_type:
                continue
            if qtype not in deduped_secondaries:
                deduped_secondaries.append(qtype)
        result.secondary_types = deduped_secondaries[:2]

        deduped_modifiers: List[LogicalModifier] = []
        for modifier in result.logical_modifiers:
            if modifier not in deduped_modifiers:
                deduped_modifiers.append(modifier)
        result.logical_modifiers = deduped_modifiers

        if result.primary_type == QuestionType.MULTI_FIELD:
            for field_spec in result.fields:
                field_spec.time_aware = result.has_time_constraint
                if result.has_time_constraint and LogicalModifier.TIME_LOOKUP not in field_spec.logical_modifiers:
                    field_spec.logical_modifiers.append(LogicalModifier.TIME_LOOKUP)
