"""
Question classifier focused on answer shape plus logical modifiers.
"""
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

from src.question_parser import (
    TIME_RANGE_SEPARATOR,
    TIME_TOKEN_PATTERN,
    _normalize_time_token,
    extract_entities,
    locate_predicate_mentions,
)


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
    entity: Optional[str]
    answer_type: QuestionType
    predicate: Optional[str]
    time_aware: bool
    sub_question: str = ""
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
    multi_field_mode: Optional[str] = None
    major_entities: List[str] = field(default_factory=list)
    helper_entities: List[str] = field(default_factory=list)
    major_predicates: List[str] = field(default_factory=list)
    helper_predicates: List[str] = field(default_factory=list)
    logic_operator: LogicOperator = LogicOperator.NONE
    ordering_attribute: Optional[str] = None
    order_direction: str = "ascending"
    comparison_operator: Optional[str] = None
    list_target: Optional[str] = None
    target_entity_class: Optional[str] = None
    entity_filter_conditions: List[Dict] = field(default_factory=list)
    boolean_keyword: Optional[str] = None
    confidence: float = 1.0
    type_scores: Dict[QuestionType, float] = field(default_factory=dict)


class QuestionClassifier:
    """Classify answer shape and logical modifiers for a question."""

    TYPE_PATTERNS = {
        QuestionType.BOOLEAN: {
            "patterns": [
                r"^(?:is|are|was|were|does|did)\b(?!.*\bhow\s+many\b)",
                r"^(?:does|did)\s+.*\b(?:have|orbit|orbits|contain)\b(?!.*\bhow\s+many\b)",
                r"^(?:is|are|was|were)\s+.*\b(?:a|an|the)\b",
                r"^(?:is|are|was|were)\s+.*\b(?:located\s+in|found\s+in|recognized\s+as)\b",
            ],
            "keywords": [],
        },
        QuestionType.COUNT: {
            "patterns": [
                r"\bhow\s+many\b",
                r"\b(?:count|number)\s+of\b",
                r"\btotal\s+(?:number|count)\s+of\b",
                r"\bwhat\s+(?:is|was)\s+the\s+number\s+of\b",
            ],
            "keywords": ["how many", "count of", "number of", "total number of", "total count of"],
        },
        QuestionType.LIST: {
            "patterns": [
                r"\bwhich\s+(?:planets|dwarf\s+planets|dwarfs|moons|satellites|objects|bodies)\b",
                r"\b(?:list|name)\s+(?:the\s+)?(?:(?:terrestrial|gas\s+giant|ice\s+giant|confirmed|known)\s+){0,2}(?:planets|dwarf\s+planets|dwarfs|moons|satellites|objects|bodies)\b",
                r"\border\s+(?:the\s+)?(?:(?:terrestrial|gas\s+giant|ice\s+giant|confirmed|known)\s+){0,2}(?:planets|dwarf\s+planets|dwarfs|moons|satellites|objects|bodies)\b",
                r"\ball\s+(?:planets|dwarf\s+planets|dwarfs|moons|satellites|objects|bodies)\b",
                r"\bwhat\s+are\s+the\s+(?:planets|dwarf\s+planets|moons|satellites|objects|bodies)\b",
            ],
            "keywords": ["which planets", "which dwarf planets", "which moons", "all planets", "all dwarf planets"],
        },
        QuestionType.ENTITY: {
            "patterns": [
                r"\bwhich\s+(?:(?:dwarf\s+)?planet|object|body)\b",
                r"\bwho\s+(?:discovered|found)\b",
                r"\bwhat\s+type\s+of\s+planet\b",
                r"\bwhat\s+(?:(?:dwarf\s+)?planet|object|body)\b",
                r"\bname\s+(?:the|a)\s+(?:(?:dwarf\s+)?planet|object|body)\b",
                r"\bwhich\s+(?:astronomer|discoverer)\b",
            ],
            "keywords": ["who discovered", "who found", "what type of planet", "which discoverer"],
        },
    }

    TIME_PATTERNS = {
        TimeSemantic.EXACT: [
            rf"(?:as\s+of(?:\s+the\s+(?:start|end)\s+of)?|by|on|at|during|in)\s+({TIME_TOKEN_PATTERN})",
        ],
        TimeSemantic.BEFORE: [
            rf"(?:before|prior\s+to|until|up\s+to)\s+({TIME_TOKEN_PATTERN})",
        ],
        TimeSemantic.AFTER: [
            rf"(?:after|since)\s+({TIME_TOKEN_PATTERN})",
        ],
        TimeSemantic.BETWEEN: [
            rf"(?:between|from)\s+({TIME_TOKEN_PATTERN})\s+(?:and|to|through)\s+({TIME_TOKEN_PATTERN})",
        ],
    }

    ORDERING_PATTERNS = [
        r"\bin\s+order\b",
        r"\border\s+of\b",
        r"\brank(?:ed)?\b",
        r"\b(?:in|of)\s+decreasing\s+order\b",
        r"\b(?:in|of)\s+increasing\s+order\b",
        r"\bby\s+decreasing\b",
        r"\bby\s+increasing\b",
        r"\border(?:ed)?\s+by\b",
        r"\bsort(?:ed)?\s+by\b",
        r"\bsmallest\s+to\s+largest\b",
        r"\blargest\s+to\s+smallest\b",
        r"\bfrom\s+smallest\s+to\s+largest\b",
        r"\bfrom\s+largest\s+to\s+smallest\b",
    ]

    COMPARISON_PATTERNS = [
        r"\bbetween\b.*\bwhich\b",
        r"\bcompared\s+to\b",
        r"\b(?:greater|less|more|fewer|larger|smaller|heavier|lighter|farther|closer)\s+than\b",
        r"\bwhich\s+(?:is|planet\s+is|one\s+is)\s+(?:greater|larger|smaller|heavier|lighter|farther|closer)\b",
    ]

    FILTER_PATTERNS = [
        r"\bfewer\s+than\b",
        r"\bmore\s+than\b",
        r"\bless\s+than\b",
        r"\bgreater\s+than\b",
        r"\bcloser\s+than\b",
        r"\bfarther\s+than\b",
        r"\blocated\s+in\b",
        r"\bfound\s+in\b",
        r"\bin\s+the\s+kuiper\s+belt\b",
        r"\bin\s+the\s+asteroid\s+belt\b",
        r"\borbit(?:s|ing)?\s+beyond\b",
        r"\bbeyond\s+earth\b",
    ]

    ORDERING_KEYWORDS = {
        "mass": r"(?:mass|heaviest|lightest|heavier|lighter|weight)",
        "distance": r"(?:distance|farther|closer|distance\s+from\s+the?\s*sun)",
        "size": r"(?:size|diameter|largest|smallest)",
        "discovered": r"(?:discovery\s+date|discovery\s+year|when\s+discovered)",
        "moons": r"(?:moons|moon\s+count|satellites)",
    }
    TIME_ORDERABLE_PREDICATES = {"discovered_on", "recognized_on", "confirmed_on", "first_observed_on"}

    COMPARISON_KEYWORDS = {
        ">": ["greater", "larger", "heavier", "farther"],
        "<": ["fewer", "smaller", "lighter", "closer"],
        "==": ["same", "equal", "equal to", "same as"],
    }

    BOOLEAN_VALUES = ["yes", "no", "true", "false"]
    MULTI_FIELD_EXCLUDED_PREDICATES = {"ordering", "comparison"}
    FACTUAL_PREDICATES = {
        "moon_count",
        "discovered_on",
        "discovered_by",
        "recognized_on",
        "confirmed_on",
        "first_observed_on",
        "mass",
        "distance_from_sun",
        "surface_gravity",
        "classification",
        "planet_type",
        "location",
    }
    HELPER_PREDICATES = {"comparison", "ordering"}
    PREDICATE_QUESTION_TEMPLATES = {
        "moon_count": "How many moons does {entity} have?",
        "discovered_on": "In what year was {entity} discovered?",
        "discovered_by": "Who discovered {entity}?",
        "recognized_on": "In what year was {entity} recognized?",
        "confirmed_on": "In what year was {entity} confirmed?",
        "first_observed_on": "In what year was {entity} first observed?",
        "mass": "What is the mass of {entity}?",
        "distance_from_sun": "What is {entity}'s distance from the Sun?",
        "surface_gravity": "What is the surface gravity of {entity}?",
        "classification": "What is the classification of {entity}?",
        "planet_type": "What type of planet is {entity}?",
        "location": "Where is {entity} located?",
    }

    def classify(self, question: str) -> ClassifiedQuestion:
        result = ClassifiedQuestion()
        result.original_question = question
        question_lower = question.lower()

        self._detect_time_constraint(question, question_lower, result)
        self._detect_logic_operators(question_lower, result)
        self._derive_major_helper_signals(question, question_lower, result)
        self._detect_multi_field(question_lower, result)
        self._classify_primary_type(question_lower, result)
        self._detect_target_entity_class(result, question_lower)
        self._detect_logical_modifiers(question_lower, result)
        self._detect_special_attributes(question_lower, result)
        self._finalize(result)

        return result

    def _detect_time_constraint(self, original_question: str, question_lower: str, result: ClassifiedQuestion) -> None:
        """
        Detect time semantics from the question text.

        Uses lowercased text for regex matching while keeping the original question
        available for future context-sensitive extensions.
        """
        semantic_order = [
            TimeSemantic.BETWEEN,
            TimeSemantic.BEFORE,
            TimeSemantic.AFTER,
            TimeSemantic.EXACT,
        ]

        for semantic in semantic_order:
            patterns = self.TIME_PATTERNS[semantic]
            for pattern in patterns:
                match = re.search(pattern, question_lower)
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

    def _derive_major_helper_signals(self, original_question: str, question: str, result: ClassifiedQuestion) -> None:
        """Derive major/helper entity and predicate signals for downstream logic."""
        entities = extract_entities(original_question)
        predicate_mentions = locate_predicate_mentions(original_question)

        for _start, predicate in predicate_mentions:
            if predicate in self.FACTUAL_PREDICATES and predicate not in result.major_predicates:
                result.major_predicates.append(predicate)
            elif predicate in self.HELPER_PREDICATES and predicate not in result.helper_predicates:
                result.helper_predicates.append(predicate)

        if not entities:
            return

        major_entities, helper_entities = self._assign_entity_roles(question, entities, result.major_predicates)
        result.major_entities = major_entities
        result.helper_entities = helper_entities

    def _assign_entity_roles(self, question: str, entities: List[str], major_predicates: List[str]) -> tuple[List[str], List[str]]:
        """Assign entity roles using current project heuristics."""
        if self._looks_like_list_target(question):
            return [], entities[:]

        if self._looks_like_comparison(question) and len(entities) >= 2:
            return entities[:2], entities[2:]

        if re.search(r"^(?:is|are|was|were|does|did)\b", question) and len(entities) >= 2:
            return [entities[0]], entities[1:]

        if len(entities) >= 2 and len(major_predicates) == 1 and re.search(r"\band\b", question):
            return entities[:], []

        if len(entities) == 1 and major_predicates:
            return entities[:], []

        return entities[:1], entities[1:]

    def _detect_multi_field(self, question: str, result: ClassifiedQuestion) -> None:
        if result.logic_operator != LogicOperator.AND:
            return

        if self._looks_like_compound_filter_or_list(question):
            return
        if len(result.major_entities) == 1 and len(result.major_predicates) >= 2:
            # Intentionally cap the current system at 2 fields.
            field_predicates = result.major_predicates[:2]
            if not self._has_true_multi_answer_structure(question, field_predicates):
                return

            subject_entity = result.major_entities[0]
            fields: List[ClassifiedField] = []
            for index, predicate in enumerate(field_predicates, start=1):
                sub_question = self._build_sub_question(predicate, subject_entity)
                if not sub_question:
                    return
                fields.append(
                    ClassifiedField(
                        name=f"field{index}",
                        entity=subject_entity,
                        answer_type=self._predicate_answer_type(predicate),
                        predicate=predicate,
                        time_aware=False,
                        sub_question=sub_question,
                        logical_modifiers=[],
                    )
                )

            result.is_multi_field = True
            result.multi_field_mode = "single_entity_multi_predicate"
            result.fields = fields
            result.multi_field_predicates = field_predicates
            return

        if len(result.major_entities) >= 2 and len(result.major_predicates) == 1:
            # Intentionally cap the current system at 2 fields.
            shared_predicate = result.major_predicates[0]
            if not self._has_true_multi_answer_structure(question, [shared_predicate]):
                return

            fields: List[ClassifiedField] = []
            for index, entity in enumerate(result.major_entities[:2], start=1):
                sub_question = self._build_sub_question(shared_predicate, entity)
                if not sub_question:
                    return
                fields.append(
                    ClassifiedField(
                        name=f"field{index}",
                        entity=entity,
                        answer_type=self._predicate_answer_type(shared_predicate),
                        predicate=shared_predicate,
                        time_aware=False,
                        sub_question=sub_question,
                        logical_modifiers=[],
                    )
                )

            result.is_multi_field = True
            result.multi_field_mode = "multi_entity_single_predicate"
            result.fields = fields
            result.multi_field_predicates = [shared_predicate]

    def _looks_like_compound_filter_or_list(self, question: str) -> bool:
        if self._looks_like_list_target(question):
            return True
        if re.search(r"\b(?:yet|that|which)\s+(?:have|has|orbit|orbits|are|were|contain|contains|lie|lies)\b", question):
            return True
        if re.search(r"\band\s+(?:have|has|orbit|orbits|are|were|is|located|found|contains|lies|with)\b", question):
            return True
        if self._looks_like_comparison(question):
            return True
        return False

    def _looks_like_list_target(self, question: str) -> bool:
        return bool(
            re.search(
                r"^(?:which|list|name|all|what\s+are\s+the)\s+"
                r"(?:planets|dwarf\s+planets|dwarfs|moons|satellites|objects|bodies)\b",
                question,
            )
        )

    def _looks_like_comparison(self, question: str) -> bool:
        return any(re.search(pattern, question) for pattern in self.COMPARISON_PATTERNS)

    def _has_true_multi_answer_structure(self, question: str, predicates: List[str]) -> bool:
        interrogative_starts = r"(?:who|what|which|when|where|how\s+many|in\s+what\s+year|what\s+year|which\s+year)"
        optional_time_prefix = rf"(?:(?:as\s+of|by|on|at|during|in|before|after|since|from|between)\s+{TIME_TOKEN_PATTERN}\s*,?\s+)?"

        if len(predicates) == 1:
            return bool(
                re.search(rf"^{optional_time_prefix}(?:what|which)\s+(?:is|are|was|were)\b", question) or
                re.search(rf"^\s*{optional_time_prefix}{interrogative_starts}\b.*\band\b", question)
            )
        if len(predicates) != 2:
            return False

        if re.search(rf",\s*and\s+{interrogative_starts}\b", question):
            return True
        if re.search(rf"^\s*{optional_time_prefix}{interrogative_starts}\b.*\band\s+{interrogative_starts}\b", question):
            return True

        # Shared-scaffold form, e.g. "what is Saturn's mass and surface gravity?"
        if re.search(rf"^{optional_time_prefix}(?:what|which)\s+(?:is|are|was|were)\b", question):
            return True

        return False

    def _predicate_answer_type(self, predicate: str) -> QuestionType:
        # Numeric outputs are routed through COUNT even when they represent
        # a year-like value such as discovered_on.
        if predicate in {"moon_count", "discovered_on", "recognized_on", "confirmed_on", "first_observed_on"}:
            return QuestionType.COUNT
        return QuestionType.ENTITY

    def _build_sub_question(self, predicate: str, entity: str) -> str:
        template = self.PREDICATE_QUESTION_TEMPLATES.get(predicate)
        if not template:
            return ""
        return template.format(entity=entity)

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

        if (
            re.search(r"\bhow\s+many\b", question) or
            re.search(r"\b(?:count|number)\s+of\b", question) or
            re.search(r"\btotal\s+(?:number|count)\s+of\b", question) or
            re.search(r"\bwhat\s+(?:is|was)\s+the\s+number\s+of\b", question)
        ):
            type_scores[QuestionType.COUNT] = max(type_scores.get(QuestionType.COUNT, 0.0), 0.9)

        if (
            self._looks_like_list_target(question) or
            re.search(r"\border\s+(?:the\s+)?(?:(?:terrestrial|gas\s+giant|ice\s+giant|confirmed|known)\s+){0,2}(?:planets|dwarf\s+planets|dwarfs|moons|satellites|objects|bodies)\b", question)
        ):
            type_scores[QuestionType.LIST] = max(type_scores.get(QuestionType.LIST, 0.0), 0.8)

        if re.search(r"\bhow\s+many\b", question):
            type_scores[QuestionType.BOOLEAN] = 0.0
        elif self._looks_like_list_target(question):
            type_scores[QuestionType.BOOLEAN] = min(type_scores.get(QuestionType.BOOLEAN, 0.0), 0.1)
        elif self._looks_like_comparison(question) or any(re.search(pattern, question) for pattern in self.ORDERING_PATTERNS):
            type_scores[QuestionType.BOOLEAN] = min(type_scores.get(QuestionType.BOOLEAN, 0.0), 0.1)

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
            has_ordering = self._has_ordering_signal(question, result.major_predicates)
            has_comparison = any(re.search(pattern, question) for pattern in self.COMPARISON_PATTERNS)
            has_filter = any(re.search(pattern, question) for pattern in self.FILTER_PATTERNS)

            if has_ordering:
                result.logical_modifiers.append(LogicalModifier.ORDERING)
            if has_comparison:
                result.logical_modifiers.append(LogicalModifier.COMPARISON)
            if self._supports_candidate_pool_reasoning(result) and (has_filter or result.logic_operator == LogicOperator.AND):
                result.logical_modifiers.append(LogicalModifier.FILTER)

    def _detect_special_attributes(self, question: str, result: ClassifiedQuestion) -> None:
        self._detect_ordering_attributes(question, result)
        self._detect_comparison_operator(question, result)
        self._detect_boolean_value(question, result)
        self._detect_target_entity_class(result, question)
        if self._supports_candidate_pool_reasoning(result):
            self._detect_list_target(result, question)
            self._detect_list_filter_conditions(question, result)

    def _has_temporal_ordering_signal(self, question: str, predicates: List[str]) -> bool:
        if not any(predicate in self.TIME_ORDERABLE_PREDICATES for predicate in predicates):
            return False
        return bool(
            re.search(
                r"\b(?:first|earliest|last|latest|most\s+recently)\b",
                question,
            )
        )

    def _first_time_orderable_predicate(self, predicates: List[str]) -> Optional[str]:
        for predicate in predicates:
            if predicate in self.TIME_ORDERABLE_PREDICATES:
                return predicate
        return None

    def _has_ordering_signal(self, question: str, predicates: List[str]) -> bool:
        return (
            any(re.search(pattern, question) for pattern in self.ORDERING_PATTERNS) or
            self._has_temporal_ordering_signal(question, predicates)
        )

    def _supports_candidate_pool_reasoning(self, result: ClassifiedQuestion) -> bool:
        return result.primary_type == QuestionType.LIST or result.target_entity_class is not None

    def _detect_ordering_attributes(self, question: str, result: ClassifiedQuestion) -> None:
        temporal_predicate = self._first_time_orderable_predicate(result.major_predicates)
        if temporal_predicate and self._has_temporal_ordering_signal(question, result.major_predicates) and re.search(r"\b(?:first|earliest)\b", question):
            result.ordering_attribute = temporal_predicate
            result.order_direction = "ascending"
            return
        if temporal_predicate and self._has_temporal_ordering_signal(question, result.major_predicates) and re.search(r"\b(?:last|latest|most\s+recently)\b", question):
            result.ordering_attribute = temporal_predicate
            result.order_direction = "descending"
            return

        for attr, pattern in self.ORDERING_KEYWORDS.items():
            if re.search(pattern, question):
                result.ordering_attribute = attr
                if re.search(r"decreasing|most|largest|highest|heaviest|farthest", question):
                    result.order_direction = "descending"
                elif re.search(r"increasing|least|smallest|lowest|lightest|closest", question):
                    result.order_direction = "ascending"
                else:
                    result.order_direction = "ascending"
                break

    def _detect_comparison_operator(self, question: str, result: ClassifiedQuestion) -> None:
        for op, keywords in self.COMPARISON_KEYWORDS.items():
            for keyword in keywords:
                if keyword in question:
                    result.comparison_operator = op
                    break
            if result.comparison_operator:
                break

    def _detect_boolean_value(self, question: str, result: ClassifiedQuestion) -> None:
        for value in self.BOOLEAN_VALUES:
            if re.search(rf"\b{re.escape(value)}\b", question):
                result.boolean_keyword = value
                break

    def _detect_list_target(self, result: ClassifiedQuestion, question: str) -> None:
        if re.search(r"\bdwarf\s+planets\b|\bdwarfs?\b", question):
            result.list_target = "dwarf_planets"
        elif re.search(r"\bplanets\b", question):
            result.list_target = "planets"
        elif re.search(r"\b(?:moons|satellites)\b", question):
            result.list_target = "moons"

    def _detect_target_entity_class(self, result: ClassifiedQuestion, question: str) -> None:
        if result.primary_type != QuestionType.ENTITY:
            return
        if self._looks_like_comparison(question) and len(result.major_entities) >= 2:
            return

        if re.search(r"\b(?:which|what)\s+dwarf\s+planet\b|\bname\s+(?:the|a)\s+dwarf\s+planet\b", question):
            result.target_entity_class = "dwarf_planets"
        elif re.search(r"\b(?:which|what)\s+planet\b|\bname\s+(?:the|a)\s+planet\b", question):
            result.target_entity_class = "planets"
        elif re.search(r"\b(?:which|what)\s+(?:moon|satellite)\b|\bname\s+(?:the|a)\s+(?:moon|satellite)\b", question):
            result.target_entity_class = "moons"

    def _detect_list_filter_conditions(self, question: str, result: ClassifiedQuestion) -> None:
        if LogicalModifier.FILTER in result.logical_modifiers:
            if re.search(r"\bfewer\b.*\bthan\b", question) or re.search(r"\bless\b.*\bthan\b", question):
                result.entity_filter_conditions.append({"operator": "<", "attribute": result.ordering_attribute or "unknown"})
            elif re.search(r"\bmore\b.*\bthan\b", question) or re.search(r"\bgreater\b.*\bthan\b", question):
                result.entity_filter_conditions.append({"operator": ">", "attribute": result.ordering_attribute or "unknown"})

            reference_entity = result.helper_entities[0] if result.helper_entities else None
            if re.search(r"\bfarther\b.*\bthan\b", question) and reference_entity:
                result.entity_filter_conditions.append({"operator": ">", "attribute": "distance_from_sun", "reference_entity": reference_entity})
            elif re.search(r"\bcloser\b.*\bthan\b", question) and reference_entity:
                result.entity_filter_conditions.append({"operator": "<", "attribute": "distance_from_sun", "reference_entity": reference_entity})

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
        result.secondary_types = self._dedupe_secondary_types(result.secondary_types, result.primary_type)
        result.logical_modifiers = self._dedupe_preserve_order(result.logical_modifiers)

        if result.primary_type == QuestionType.MULTI_FIELD:
            for field_spec in result.fields:
                field_spec.time_aware = result.has_time_constraint
                if result.has_time_constraint and LogicalModifier.TIME_LOOKUP not in field_spec.logical_modifiers:
                    field_spec.logical_modifiers.append(LogicalModifier.TIME_LOOKUP)
                field_spec.logical_modifiers = self._dedupe_preserve_order(field_spec.logical_modifiers)

    def _dedupe_secondary_types(self, secondary_types: List[QuestionType], primary_type: QuestionType) -> List[QuestionType]:
        deduped: List[QuestionType] = []
        for qtype in secondary_types:
            if qtype == primary_type:
                continue
            if qtype not in deduped:
                deduped.append(qtype)
        return deduped[:2]

    def _dedupe_preserve_order(self, values: List) -> List:
        deduped = []
        for value in values:
            if value not in deduped:
                deduped.append(value)
        return deduped
