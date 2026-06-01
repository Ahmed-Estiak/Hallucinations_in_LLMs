"""Extract validated temporal count evidence from astronomy source text."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.time_utils import time_window

SATELLITE_HEADER_RE = re.compile(r"^Satellites of (.+?):$")
ROMAN_RE = re.compile(
    r"^(?=[MDCLXVI]+$)M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)
SATELLITE_DESIGNATION_RE = re.compile(r"^S/\d{4}\s+[A-Z]\d+$")
MPEC_RE = re.compile(r"\bMPEC\s+(\d{4})-([A-Z])\d+\b")
DATE_TEXT_PATTERN = (
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)(?:\s+\d{1,2}(?:st|nd|rd|th)?,?)?\s+\d{4}|\d{4}"
)
EXPLICIT_COUNT_RE = re.compile(
    rf"\bAs of\s+(?P<date>{DATE_TEXT_PATTERN})\s*,?\s*"
    r"(?P<subject>[A-Z][A-Za-z0-9 -]{1,50}?)\s+(?:had|has)\s+"
    r"(?P<value>\d[\d,]*)\s+(?P<claim>(?:officially\s+)?confirmed|known)\s+"
    r"(?:moon|moons|natural\s+satellites|satellites)\b",
    re.IGNORECASE,
)
CURRENT_COUNT_RE = re.compile(
    r"\b(?P<subject>[A-Z][A-Za-z0-9 -]{1,50}?)\s+(?:currently\s+)?(?:has|have)\s+"
    r"(?P<value>\d[\d,]*)\s+(?:(?P<claim>(?:officially\s+)?confirmed|known)\s+)?"
    r"(?:moon|moons|natural\s+satellites|satellites)\b",
)
COUNT_CLAIM_RE = re.compile(
    r"\b(?P<subject>the\s+planet|it|[A-Z][A-Za-z0-9 -]{1,50}?)\s+"
    r"(?:now\s+)?(?:has|have|had)\s+"
    r"(?P<value>\d[\d,]*)\s+"
    r"(?:(?P<claim>(?:officially\s+)?confirmed|known|named|listed)\s+)?"
    r"(?:moon|moons|natural\s+satellites|satellites)\b",
)
POSSESSIVE_MOONS_RE = re.compile(
    r"\b(?P<subject>[A-Z][A-Za-z0-9-]*(?:\s+[A-Z][A-Za-z0-9-]*){0,3})['’]s\s+"
    r"(?:known\s+|confirmed\s+|named\s+|listed\s+)?(?:moon|moons|satellite|satellites)\b"
)

MONTHS = {
    "january": ("January", 1),
    "february": ("February", 2),
    "march": ("March", 3),
    "april": ("April", 4),
    "may": ("May", 5),
    "june": ("June", 6),
    "july": ("July", 7),
    "august": ("August", 8),
    "september": ("September", 9),
    "october": ("October", 10),
    "november": ("November", 11),
    "december": ("December", 12),
}

HALF_MONTHS = {
    "A": ("January", 1), "B": ("January", 1),
    "C": ("February", 2), "D": ("February", 2),
    "E": ("March", 3), "F": ("March", 3),
    "G": ("April", 4), "H": ("April", 4),
    "J": ("May", 5), "K": ("May", 5),
    "L": ("June", 6), "M": ("June", 6),
    "N": ("July", 7), "O": ("July", 7),
    "P": ("August", 8), "Q": ("August", 8),
    "R": ("September", 9), "S": ("September", 9),
    "T": ("October", 10), "U": ("October", 10),
    "V": ("November", 11), "W": ("November", 11),
    "X": ("December", 12), "Y": ("December", 12),
}


@dataclass(frozen=True)
class SatelliteEntry:
    subject: str
    name: str
    designation: str
    discovery_year: str
    reference: str


@dataclass(frozen=True)
class StructuredFact:
    fact_id: str
    heading: str
    text: str
    subject: str
    predicate: str
    value: int
    evidence_type: str
    validation_status: str
    claim_type: str
    observed_at: str = ""
    valid_until_exclusive: str = ""
    interval_semantics: str = ""

    def metadata(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "predicate": self.predicate,
            "value": self.value,
            "evidence_type": self.evidence_type,
            "validation_status": self.validation_status,
            "claim_type": self.claim_type,
            "observed_at": self.observed_at,
            "valid_until_exclusive": self.valid_until_exclusive,
            "interval_semantics": self.interval_semantics,
        }


def extract_temporal_count_facts(text: str) -> list[StructuredFact]:
    return dedupe_facts([
        *extract_explicit_temporal_count_facts(text),
        *extract_explicit_current_count_facts(text),
        *extract_satellite_count_facts(text),
    ])


def extract_explicit_temporal_count_facts(text: str) -> list[StructuredFact]:
    facts = []
    for match in EXPLICIT_COUNT_RE.finditer(text):
        subject = match.group("subject").strip()
        observed_at = normalize_date(match.group("date"))
        if not observed_at or time_window(observed_at) is None:
            continue
        value = int(match.group("value").replace(",", ""))
        claim = match.group("claim").lower()
        claim_type = "confirmed_moons" if "confirmed" in claim else "known_moons"
        subject_slug = slug(subject)
        facts.append(StructuredFact(
            fact_id=f"{subject_slug}_moon_count_explicit_{observed_at.replace('-', '_')}_{value}",
            heading=f"Explicit Temporal Moon Count - {subject}",
            text=(
                f"Explicit dated source claim: As of {display_date(observed_at)}, "
                f"{subject} had {value} {claim.replace('  ', ' ')} moons."
            ),
            subject=subject,
            predicate="moon_count",
            value=value,
            evidence_type="explicit_dated_sentence",
            validation_status="validated",
            claim_type=claim_type,
            observed_at=observed_at,
        ))
    facts.extend(extract_dated_sentence_count_facts(text))
    return dedupe_facts(facts)


def extract_dated_sentence_count_facts(text: str) -> list[StructuredFact]:
    """Extract generic dated count claims from one sentence.

    This covers wording such as "Updating the count of Saturn's moons in 2019,
    the planet now has 82 named moons." The date and count do not need to be in
    the fixed "As of ..." order, but they must be in the same sentence to avoid
    borrowing dates from unrelated nearby statements.
    """
    facts: list[StructuredFact] = []
    for sentence in iter_sentences(text):
        dates = list(re.finditer(DATE_TEXT_PATTERN, sentence, flags=re.IGNORECASE))
        if not dates:
            continue
        for count_match in COUNT_CLAIM_RE.finditer(sentence):
            observed_at = closest_observed_date(sentence, dates, count_match.start())
            if not observed_at or time_window(observed_at) is None:
                continue
            subject = normalize_count_subject(count_match.group("subject"), sentence)
            if not subject:
                continue
            value = int(count_match.group("value").replace(",", ""))
            claim = (count_match.group("claim") or "").lower()
            claim_type = (
                "confirmed_moons" if "confirmed" in claim
                else "known_moons" if "known" in claim
                else "named_moons" if "named" in claim
                else "listed_satellites" if "listed" in claim
                else "moon_count"
            )
            facts.append(StructuredFact(
                fact_id=f"{slug(subject)}_moon_count_explicit_{observed_at.replace('-', '_')}_{value}",
                heading=f"Explicit Temporal Moon Count - {subject}",
                text=(
                    f"Explicit dated source claim: As of {display_date(observed_at)}, "
                    f"{subject} had {value} {claim_type.replace('_', ' ')}."
                ),
                subject=subject,
                predicate="moon_count",
                value=value,
                evidence_type="explicit_dated_sentence",
                validation_status="validated",
                claim_type=claim_type,
                observed_at=observed_at,
            ))
    return dedupe_facts(facts)


def extract_explicit_current_count_facts(text: str) -> list[StructuredFact]:
    facts = []
    dated_prefix = re.compile(rf"\bAs of\s+(?:{DATE_TEXT_PATTERN})\s*,?\s*$", re.IGNORECASE)
    for match in CURRENT_COUNT_RE.finditer(text):
        prefix = text[max(0, match.start() - 60):match.start()]
        if dated_prefix.search(prefix):
            continue
        subject = match.group("subject").strip()
        value = int(match.group("value").replace(",", ""))
        claim = (match.group("claim") or "").lower()
        claim_type = (
            "confirmed_moons" if "confirmed" in claim
            else "known_moons" if claim
            else "moon_count"
        )
        display_claim = f"{claim} " if claim else ""
        facts.append(StructuredFact(
            fact_id=f"{slug(subject)}_moon_count_current_assertion_{value}_{slug(claim_type)}",
            heading=f"Current Moon Count Assertion - {subject}",
            text=(
                f"Direct current source assertion: {subject} has {value} "
                f"{display_claim}moons."
            ),
            subject=subject,
            predicate="moon_count",
            value=value,
            evidence_type="explicit_current_sentence",
            validation_status="source_asserted",
            claim_type=claim_type,
        ))
    return dedupe_facts(facts)


def extract_satellite_count_facts(text: str) -> list[StructuredFact]:
    """Build facts only from internally consistent satellite table sections."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    facts: list[StructuredFact] = []
    for subject, declared_total, section_lines in iter_satellite_sections(lines):
        entries = parse_satellite_entries(subject, section_lines)
        if declared_total is None or not entries or len(entries) != declared_total:
            continue
        facts.extend(build_count_facts(subject, declared_total, entries))
    return dedupe_facts(facts)


def iter_satellite_sections(lines: list[str]) -> list[tuple[str, int | None, list[str]]]:
    sections: list[tuple[str, int | None, list[str]]] = []
    header_indexes = [
        index for index, line in enumerate(lines)
        if SATELLITE_HEADER_RE.match(line)
    ]
    for offset, start in enumerate(header_indexes):
        match = SATELLITE_HEADER_RE.match(lines[start])
        if not match:
            continue
        end = header_indexes[offset + 1] if offset + 1 < len(header_indexes) else len(lines)
        subject = match.group(1).strip()
        declared_total = parse_int(lines[start + 1]) if start + 1 < end else None
        section_start = start + 2 if declared_total is not None else start + 1
        sections.append((subject, declared_total, lines[section_start:end]))
    return sections


def parse_satellite_entries(subject: str, lines: list[str]) -> list[SatelliteEntry]:
    entries: list[SatelliteEntry] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if ROMAN_RE.match(line):
            if index + 4 >= len(lines):
                break
            name = lines[index + 1]
            cursor = index + 2
            designation = ""
            if cursor < len(lines) and SATELLITE_DESIGNATION_RE.match(lines[cursor]):
                designation = lines[cursor]
                cursor += 1
            if cursor + 2 >= len(lines):
                break
            entries.append(SatelliteEntry(
                subject=subject,
                name=name,
                designation=designation,
                discovery_year=lines[cursor],
                reference=lines[cursor + 2],
            ))
            index = cursor + 3
            continue

        if SATELLITE_DESIGNATION_RE.match(line):
            if index + 3 >= len(lines):
                break
            entries.append(SatelliteEntry(
                subject=subject,
                name=line,
                designation=line,
                discovery_year=lines[index + 1],
                reference=lines[index + 3],
            ))
            index += 4
            continue
        index += 1
    return entries


def build_count_facts(
    subject: str,
    declared_total: int,
    entries: list[SatelliteEntry],
) -> list[StructuredFact]:
    subject_slug = slug(subject)
    facts = [StructuredFact(
        fact_id=f"{subject_slug}_satellite_count_current",
        heading=f"Validated Satellite Counts - {subject}",
        text=(
            f"Validated table fact: {subject} has {declared_total} listed satellites "
            f"in this source's current table."
        ),
        subject=subject,
        predicate="moon_count",
        value=declared_total,
        evidence_type="validated_table_current",
        validation_status="validated",
        claim_type="listed_satellites",
    )]
    anchors = []
    for year, month_number, month_name in sorted({
        key for entry in entries
        if (key := mpec_sort_key(entry.reference)) is not None
    }):
        included = [
            entry for entry in entries
            if entry_is_available_by(entry, year, month_number)
        ]
        if included:
            anchors.append((year, month_number, month_name, len(included)))

    for index, (year, month_number, month_name, count) in enumerate(anchors):
        observed_at = f"{year:04d}-{month_number:02d}"
        next_date = ""
        if index + 1 < len(anchors):
            next_year, next_month, _next_name, _next_count = anchors[index + 1]
            next_date = f"{next_year:04d}-{next_month:02d}"
        interval_text = (
            f" Under this validated table timeline, this value applies until the next "
            f"recorded change in {display_date(next_date)}."
            if next_date
            else " No later validated change boundary is available in this table extract."
        )
        facts.append(StructuredFact(
            fact_id=f"{subject_slug}_satellite_count_as_of_{year}_{month_number:02d}",
            heading=f"Validated Temporal Satellite Counts - {subject}",
            text=(
                f"Validated table-derived temporal fact: As of {month_name} {year}, "
                f"{subject} had {count} listed satellites in this discovery table."
                f"{interval_text}"
            ),
            subject=subject,
            predicate="moon_count",
            value=count,
            evidence_type="derived_validated_timeline",
            validation_status="validated",
            claim_type="listed_satellites",
            observed_at=observed_at,
            valid_until_exclusive=next_date,
            interval_semantics="carry_forward_until_next_validated_change" if next_date else "",
        ))
    return facts


def mpec_sort_key(reference: str) -> tuple[int, int, str] | None:
    match = MPEC_RE.search(reference)
    if not match:
        return None
    year = int(match.group(1))
    month = HALF_MONTHS.get(match.group(2))
    if month is None:
        return None
    month_name, month_number = month
    return year, month_number, month_name


def entry_is_available_by(entry: SatelliteEntry, year: int, month_number: int) -> bool:
    key = mpec_sort_key(entry.reference)
    if key is not None:
        ref_year, ref_month, _ = key
        return (ref_year, ref_month) <= (year, month_number)
    discovery_years = [int(value) for value in re.findall(r"\b\d{4}\b", entry.discovery_year)]
    return bool(discovery_years) and max(discovery_years) < year


def dedupe_facts(facts: list[StructuredFact]) -> list[StructuredFact]:
    seen: set[str] = set()
    deduped = []
    for fact in facts:
        if fact.fact_id not in seen:
            seen.add(fact.fact_id)
            deduped.append(fact)
    return deduped


def iter_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip()
    ]


def closest_observed_date(sentence: str, dates: list[re.Match[str]], position: int) -> str:
    preceding = [match for match in dates if match.start() <= position]
    candidates = preceding or dates
    closest = min(candidates, key=lambda match: abs(position - match.start()))
    return normalize_date(closest.group(0))


def normalize_count_subject(raw_subject: str, sentence: str) -> str:
    subject = raw_subject.strip()
    if subject.lower() in {"the planet", "it"} or " of " in subject.lower():
        possessive = POSSESSIVE_MOONS_RE.search(sentence)
        if not possessive:
            return ""
        subject = possessive.group("subject").strip()
    return re.sub(r"\s+", " ", subject)


def normalize_date(value: str) -> str:
    text = value.strip().lower().replace(",", "")
    text = re.sub(r"(\d)(?:st|nd|rd|th)\b", r"\1", text)
    if text.isdigit() and len(text) == 4:
        return text
    match = re.match(r"([a-z]+)(?:\s+(\d{1,2}))?\s+(\d{4})$", text)
    if not match or match.group(1) not in MONTHS:
        return ""
    year = match.group(3)
    month = MONTHS[match.group(1)][1]
    if match.group(2):
        return f"{year}-{month:02d}-{int(match.group(2)):02d}"
    return f"{year}-{month:02d}"


def display_date(value: str) -> str:
    if not value:
        return ""
    if "-" not in value:
        return value
    date_parts = value.split("-")
    year, month = date_parts[:2]
    month_name = next(name for name, number in MONTHS.values() if number == int(month))
    if len(date_parts) == 3:
        return f"{month_name} {int(date_parts[2])}, {year}"
    return f"{month_name} {year}"


def parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
