"""Derive count facts from flattened planetary satellite tables."""

from __future__ import annotations

import re
from dataclasses import dataclass


PLANET_HEADER_RE = re.compile(r"^Satellites of ([A-Z][a-z]+):$")
ROMAN_RE = re.compile(
    r"^(?=[MDCLXVI]+$)M{0,4}(CM|CD|D?C{0,3})(XC|XL|L?X{0,3})(IX|IV|V?I{0,3})$"
)
SATELLITE_DESIGNATION_RE = re.compile(r"^S/\d{4}\s+[A-Z]\d+$")
MPEC_RE = re.compile(r"\bMPEC\s+(\d{4})-([A-Z])\d+\b")

HALF_MONTHS = {
    "A": ("January", 1),
    "B": ("January", 1),
    "C": ("February", 2),
    "D": ("February", 2),
    "E": ("March", 3),
    "F": ("March", 3),
    "G": ("April", 4),
    "H": ("April", 4),
    "J": ("May", 5),
    "K": ("May", 5),
    "L": ("June", 6),
    "M": ("June", 6),
    "N": ("July", 7),
    "O": ("July", 7),
    "P": ("August", 8),
    "Q": ("August", 8),
    "R": ("September", 9),
    "S": ("September", 9),
    "T": ("October", 10),
    "U": ("October", 10),
    "V": ("November", 11),
    "W": ("November", 11),
    "X": ("December", 12),
    "Y": ("December", 12),
}


@dataclass(frozen=True)
class SatelliteEntry:
    planet: str
    name: str
    designation: str
    discovery_year: str
    reference: str


@dataclass(frozen=True)
class StructuredFact:
    fact_id: str
    heading: str
    text: str


def extract_satellite_count_facts(text: str) -> list[StructuredFact]:
    """Extract compact count facts from JPL/NASA-style satellite tables.

    The generic HTML cleaner flattens table rows into one value per line. This
    parser keeps the logic conservative: it only activates for sections headed
    "Satellites of <Planet>:" and rows matching the known satellite designation
    pattern.
    """

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    facts: list[StructuredFact] = []
    for planet, declared_total, section_lines in iter_satellite_sections(lines):
        entries = parse_satellite_entries(planet, section_lines)
        if not entries:
            continue
        facts.extend(build_count_facts(planet, declared_total, entries))
    return facts


def iter_satellite_sections(lines: list[str]) -> list[tuple[str, int | None, list[str]]]:
    sections: list[tuple[str, int | None, list[str]]] = []
    header_indexes = [
        index for index, line in enumerate(lines)
        if PLANET_HEADER_RE.match(line)
    ]
    for offset, start in enumerate(header_indexes):
        match = PLANET_HEADER_RE.match(lines[start])
        if not match:
            continue
        end = header_indexes[offset + 1] if offset + 1 < len(header_indexes) else len(lines)
        planet = match.group(1)
        declared_total = parse_int(lines[start + 1]) if start + 1 < end else None
        section_start = start + 2 if declared_total is not None else start + 1
        sections.append((planet, declared_total, lines[section_start:end]))
    return sections


def parse_satellite_entries(planet: str, lines: list[str]) -> list[SatelliteEntry]:
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
                planet=planet,
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
                planet=planet,
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
    planet: str,
    declared_total: int | None,
    entries: list[SatelliteEntry],
) -> list[StructuredFact]:
    facts: list[StructuredFact] = []
    planet_lower = planet.lower()

    if declared_total is not None:
        facts.append(StructuredFact(
            fact_id=f"{planet_lower}_satellite_count_current",
            heading=f"Structured Satellite Counts - {planet}",
            text=(
                f"Structured satellite count fact: {planet} has {declared_total} listed "
                f"satellites or moons in this source's current table."
            ),
        ))

    mpec_keys = sorted({
        key for entry in entries
        if (key := mpec_sort_key(entry.reference)) is not None
    })
    for year, month_number, month_name in mpec_keys:
        included = [
            entry for entry in entries
            if reference_is_available_by(entry.reference, year, month_number)
        ]
        if not included:
            continue
        latest_entries = [
            entry for entry in entries
            if mpec_sort_key(entry.reference) == (year, month_number, month_name)
        ]
        latest_entry = latest_entries[-1] if latest_entries else None
        latest_ref = latest_entry.reference if latest_entry else ""
        latest_name = (latest_entry.designation or latest_entry.name) if latest_entry else ""
        facts.append(StructuredFact(
            fact_id=f"{planet_lower}_satellite_count_as_of_{year}_{month_number:02d}",
            heading=f"Structured Satellite Counts - {planet}",
            text=(
                f"As of {month_name} {year}, {planet} had {len(included)} confirmed "
                f"moons or listed satellites in this source's discovery table. The count "
                f"includes entries available through {latest_ref}"
                f"{f' ({latest_name})' if latest_name else ''} and excludes later MPEC entries."
            ),
        ))

    return dedupe_facts(facts)


def mpec_sort_key(reference: str) -> tuple[int, int, str] | None:
    match = MPEC_RE.search(reference)
    if not match:
        return None
    year = int(match.group(1))
    half_month = match.group(2)
    month = HALF_MONTHS.get(half_month)
    if month is None:
        return year, 0, "Unknown"
    month_name, month_number = month
    return year, month_number, month_name


def reference_is_available_by(reference: str, year: int, month_number: int) -> bool:
    key = mpec_sort_key(reference)
    if key is None:
        return True
    ref_year, ref_month, _ = key
    return (ref_year, ref_month) <= (year, month_number)


def dedupe_facts(facts: list[StructuredFact]) -> list[StructuredFact]:
    seen: set[str] = set()
    deduped: list[StructuredFact] = []
    for fact in facts:
        if fact.fact_id in seen:
            continue
        seen.add(fact.fact_id)
        deduped.append(fact)
    return deduped


def parse_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None
