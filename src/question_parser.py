"""
Question Parser: Extract entities, predicates, and time constraints from questions
"""
import re
from typing import Dict, List, Tuple, Optional


MONTH_MAP = {
    "jan": "01",
    "january": "01",
    "feb": "02",
    "february": "02",
    "mar": "03",
    "march": "03",
    "apr": "04",
    "april": "04",
    "may": "05",
    "jun": "06",
    "june": "06",
    "jul": "07",
    "july": "07",
    "aug": "08",
    "august": "08",
    "sep": "09",
    "sept": "09",
    "september": "09",
    "oct": "10",
    "october": "10",
    "nov": "11",
    "november": "11",
    "dec": "12",
    "december": "12",
}

MOON_LIKE_BODY_PATTERN = r"(?:moons|satellites|orbital bodies)"
MONTH_NAME_PATTERN = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t|tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
TIME_RANGE_SEPARATOR = ".."
TIME_TOKEN_PATTERN = rf"(?:{MONTH_NAME_PATTERN}\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,)?\s+\d{{4}}|{MONTH_NAME_PATTERN}\s+\d{{4}}|\d{{4}}[-/]\d{{1,2}}(?:[-/]\d{{1,2}})?|\d{{1,2}}[-/]\d{{4}}|\d{{4}})"


# Predicate synonyms and keyword mappings with COMPLEX question patterns
PREDICATE_KEYWORDS = {
    "moon_count": {
        "keywords": [],
        "pattern": rf"(?:how many|count of|number of|total number of)\s+(?:officially\s+)?(?:confirmed\s+)?(?:known\s+)?{MOON_LIKE_BODY_PATTERN}|how many\s+(?:officially\s+)?(?:confirmed\s+)?(?:known\s+)?{MOON_LIKE_BODY_PATTERN}\s+.*(?:did|does)\s+.*have"
    },
    "discovered_on": {
        "keywords": ["discovered", "discovery", "when discovered", "found", "when found", "discovery year", "found in"],
        "pattern": r"(?:when|what year|which year|in what year)\s+(?:was\s+)?(?:.*\s+)?(?:discovered|found)|\b(?:discovered|found)\b\s+.*\bwhen\b|\bdiscovery year\b"
    },
    "discovered_by": {
        "keywords": ["discovered by", "who discovered", "discoverer", "found by", "who found", "finder"],
        "pattern": r"(?:who|which)\s+(?:astronomer\s+)?(?:discovered|found)|(?:discovered|found)\s+by|who\s+was\s+.*(?:discovered|found)\s+by|name\s+the\s+(?:astronomer|discoverer|finder)"
    },
    "mass": {
        "keywords": ["mass", "weight", "massive", "heavier", "lighter", "heaviest", "lightest"],
        "pattern": r"(?:what\s+is\s+the\s+mass|mass\s+of|greater\s+mass|more\s+massive|less\s+massive|heaviest|lightest|heavier|lighter)"
    },
    "distance_from_sun": {
        "keywords": ["distance from sun", "distance to sun", "away from sun", "far from sun", "farther from sun", "closer to sun"],
        "pattern": r"(?:distance\s+(?:from|to)\s+the?\s*sun|distance\s+from\s+sun|far(?:ther)?\s+from\s+the?\s*sun|closer\s+to\s+the?\s*sun|which\s+.*(?:farther\s+from|closer\s+to)\s+the?\s*sun)"
    },
    "surface_gravity": {
        "keywords": ["surface gravity", "gravitational pull", "gravity of", "stronger gravity", "weaker gravity", "higher gravity", "lower gravity", "stronger surface gravity", "weaker surface gravity", "higher surface gravity", "lower surface gravity"],
        "pattern": r"(?:surface\s+gravity|gravity\s+of|gravitational\s+pull|stronger\s+(?:surface\s+)?gravity|weaker\s+(?:surface\s+)?gravity|higher\s+(?:surface\s+)?gravity|lower\s+(?:surface\s+)?gravity)"
    },
    "classification": {
        "keywords": ["classification", "classified", "recognized as", "classify as"],
        "pattern": r"(?:classification\s+of|how\s+is\s+.*classified|classif(?:y|ied)\s+as|is\s+.*(?:a|an)\s+(?:dwarf\s+)?planet|recogniz(?:e|ed)\s+.*(?:as\s+)?(?:a\s+)?(?:dwarf\s+)?planet|recognized\s+as)"
    },
    "planet_type": {
        "keywords": ["type of planet", "planet type", "terrestrial", "ice giant", "gas giant"],
        "pattern": r"(?:type\s+of\s+planet|planet\s+type|what\s+type\s+of\s+planet|terrestrial|gas\s+giant|ice\s+giant)"
    },
    "location": {
        "keywords": ["location", "located in", "found in", "in the kuiper belt", "in the asteroid belt"],
        "pattern": r"(?:location\s+of|located\s+in|where\s+is\s+.*(?:located|found)|found\s+in|in\s+the\s+kuiper\s+belt|in\s+the\s+asteroid\s+belt)"
    },
    "ordering": {
        "keywords": ["order", "rank", "decreasing", "increasing", "smallest to largest", "largest to smallest"],
        "pattern": r"(?:order\s+of|in\s+order|rank(?:ed)?|decreasing\s+order|increasing\s+order|smallest\s+to\s+largest|largest\s+to\s+smallest)"
    },
    "comparison": {
        "keywords": ["between", "compared to", "greater than", "less than", "more than", "fewer than"],
        "pattern": r"(?:between\s+.*\s+which|compared\s+to|greater\s+than|less\s+than|more\s+than|fewer\s+than|which\s+is\s+(?:greater|larger|smaller|farther|closer))"
    }
}

# Entity name patterns (planets, dwarf planets, AND other astronomical features)
PRIMARY_KNOWN_ENTITIES = {
    "Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune",
    "Pluto", "Ceres", "Eris", "Makemake", "Haumea",
    "Charon", "Triton", "Europa", "Ganymede", "Callisto", "Io",
    "Johann Galle", "Clyde Tombaugh", "Johann Gottfried Galle",
    "Kuiper Belt", "Asteroid Belt", "Solar System", "Oort Cloud"  # Astronomical regions
}

MOON_ENTITY_PATTERN = re.compile(r"\bmoon\b")
MOONS_ENTITY_PATTERN = re.compile(r"\bmoons\b")


def _find_entity_spans(question: str, entities: set[str]) -> List[tuple[int, str]]:
    """Find known entity mentions with word-boundary-aware matching."""
    question_lower = question.lower()
    matches: List[tuple[int, str]] = []

    for entity in sorted(entities, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(entity.lower())}\b")
        match = pattern.search(question_lower)
        if match:
            matches.append((match.start(), entity))

    matches.sort(key=lambda item: item[0])
    return matches


def extract_entities(question: str) -> List[str]:
    """
    Extract entity names from question using keyword matching.
    Returns list of entity names found.
    """
    entity_matches = _find_entity_spans(question, PRIMARY_KNOWN_ENTITIES)
    entities = [entity for _, entity in entity_matches]

    # Treat singular/plural moon mentions as fallback entities only when
    # no higher-priority named entities were detected.
    if not entities:
        if MOON_ENTITY_PATTERN.search(question.lower()):
            entities.append("Moon")
        if MOONS_ENTITY_PATTERN.search(question.lower()):
            entities.append("Moons")

    return entities


def extract_time_constraint(question: str) -> Optional[str]:
    """
    Extract time constraint from question.
    Returns time string (e.g., "2022-08", "2024-01", year like "1981") or None.
    """
    question_lower = question.lower()

    between_match = re.search(
        rf"(?:between|from)\s+({TIME_TOKEN_PATTERN})\s+(?:and|to)\s+({TIME_TOKEN_PATTERN})",
        question_lower
    )
    if between_match:
        start = _normalize_time_token(between_match.group(1))
        end = _normalize_time_token(between_match.group(2))
        if start and end:
            return f"{start}{TIME_RANGE_SEPARATOR}{end}"

    single_patterns = [
        rf"(?:as of|by|on|during|in)\s+({TIME_TOKEN_PATTERN})",
        rf"(?:before|prior to|until|up to)\s+({TIME_TOKEN_PATTERN})",
        rf"(?:after|since)\s+({TIME_TOKEN_PATTERN})",
    ]

    for pattern in single_patterns:
        match = re.search(pattern, question_lower)
        if match:
            normalized = _normalize_time_token(match.group(1))
            if normalized:
                return normalized
    
    return None


def _normalize_time_token(token: str) -> Optional[str]:
    """Normalize varied date expressions to YYYY, YYYY-MM, or YYYY-MM-DD."""
    text = re.sub(r"\s+", " ", token.strip().lower().replace(",", ""))
    text = re.sub(r"(\d)(st|nd|rd|th)\b", r"\1", text)

    month_day_year = re.fullmatch(rf"({MONTH_NAME_PATTERN})\s+(\d{{1,2}})\s+(\d{{4}})", text)
    if month_day_year:
        month_name, day, year = month_day_year.groups()
        month = MONTH_MAP.get(month_name)
        if month:
            return f"{year}-{month}-{int(day):02d}"

    month_year = re.fullmatch(rf"({MONTH_NAME_PATTERN})\s+(\d{{4}})", text)
    if month_year:
        month_name, year = month_year.groups()
        month = MONTH_MAP.get(month_name)
        if month:
            return f"{year}-{month}"

    iso_like = re.fullmatch(r"(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?", text)
    if iso_like:
        year, month, day = iso_like.groups()
        if day:
            return f"{year}-{int(month):02d}-{int(day):02d}"
        return f"{year}-{int(month):02d}"

    month_slash_year = re.fullmatch(r"(\d{1,2})[-/](\d{4})", text)
    if month_slash_year:
        month, year = month_slash_year.groups()
        return f"{year}-{int(month):02d}"

    if re.fullmatch(r"\d{4}", text):
        return text

    return None


def infer_predicates(question: str) -> List[str]:
    """
    Infer which predicates the question is asking about.
    Returns list of predicate names.
    """
    predicates = []
    question_lower = question.lower()
    
    for predicate, info in PREDICATE_KEYWORDS.items():
        # Check pattern first for higher confidence
        if re.search(info["pattern"], question_lower):
            predicates.append(predicate)
        else:
            # Fall back to keyword matching
            for keyword in info["keywords"]:
                if keyword.lower() in question_lower:
                    predicates.append(predicate)
                    break
    
    return list(set(predicates))  # Remove duplicates


def parse_question(question: str) -> Dict[str, any]:
    """
    Parse a question and extract structured information.
    
    Returns:
        {
            "question": str,
            "entities": List[str],
            "predicates": List[str],
            "time_constraint": Optional[str]
        }
    """
    return {
        "question": question,
        "entities": extract_entities(question),
        "predicates": infer_predicates(question),
        "time_constraint": extract_time_constraint(question)
    }


# Test
if __name__ == "__main__":
    test_questions = [
        "As of November 2022, how many confirmed moons orbit Jupiter?",
        "In what year was Neptune discovered, and who discovered it?",
        "As of February 2025, what is the total number of confirmed moons of Uranus?",
        "Which dwarf planet located in the Kuiper Belt was discovered first?",
    ]
    
    for q in test_questions:
        result = parse_question(q)
        print(f"Q: {q}")
        print(f"  Entities: {result['entities']}")
        print(f"  Predicates: {result['predicates']}")
        print(f"  Time: {result['time_constraint']}")
        print()
