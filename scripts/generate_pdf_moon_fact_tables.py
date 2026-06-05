"""Generate audit CSVs for moon-count facts found in the PDF-only RAG index."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.rag.pdf_paths import PDF_CHUNKS_PATH
from src.rag.retriever import parse_count_phrase


OUTPUT_DIR = PROJECT_ROOT / "reports" / "debug" / "pdf"
CURRENT_ASSERTIONS_CSV = OUTPUT_DIR / "pdf_moon_current_assertions.csv"
TIMELINE_ASSERTIONS_CSV = OUTPUT_DIR / "pdf_moon_timeline_assertions.csv"

PLANETS = ("Mercury", "Venus", "Earth", "Mars", "Jupiter", "Saturn", "Uranus", "Neptune")
MOON_TERMS = r"(?:moon|moons|natural\s+satellites|satellite|satellites)"
NUMBER_WORDS = (
    "zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    "thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|"
    "thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred"
)
COUNT_PHRASE = rf"(?:\d[\d,]*|(?:(?:{NUMBER_WORDS})(?:[-\s]+(?:and\s+)?(?:{NUMBER_WORDS})){{0,4}}))"
DATE_TEXT = (
    r"(?:January|February|March|April|May|June|July|August|September|October|"
    r"November|December)(?:\s+\d{1,2}(?:st|nd|rd|th)?,?)?\s+\d{4}|\d{4}"
)


def main() -> int:
    chunks = load_jsonl(PROJECT_ROOT / PDF_CHUNKS_PATH)
    rows = dedupe_rows(extract_rows(chunks))
    current_rows = sorted(rows, key=row_sort_key)
    timeline_rows = sorted(rows, key=timeline_sort_key)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(CURRENT_ASSERTIONS_CSV, current_rows)
    write_csv(TIMELINE_ASSERTIONS_CSV, timeline_rows)

    print(f"Rows: {len(rows)}")
    print(f"Wrote: {CURRENT_ASSERTIONS_CSV}")
    print(f"Wrote: {TIMELINE_ASSERTIONS_CSV}")
    return 0


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_rows(chunks: list[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for chunk in chunks:
        if chunk.get("source_type") != "pdf":
            continue
        if isinstance(chunk.get("temporal_fact"), dict):
            yield row_from_temporal_fact(chunk)
        yield from rows_from_text_chunk(chunk)


def row_from_temporal_fact(chunk: dict[str, Any]) -> dict[str, Any]:
    fact = chunk["temporal_fact"]
    observed_at = fact.get("observed_at", "")
    subject = normalize_subject(fact.get("subject", ""))
    return base_row(
        chunk,
        subject=subject,
        value=fact.get("value", ""),
        claim_type=fact.get("claim_type", ""),
        evidence_type=fact.get("evidence_type", ""),
        validation_status=fact.get("validation_status", ""),
        observed_at=observed_at,
        valid_until_exclusive=fact.get("valid_until_exclusive", ""),
        interval_semantics=fact.get("interval_semantics", ""),
        date_status="explicit_or_derived" if observed_at else "unknown",
        extraction_method="structured_fact",
        evidence=chunk.get("text", ""),
    )


def rows_from_text_chunk(chunk: dict[str, Any]) -> Iterable[dict[str, Any]]:
    text = chunk.get("text", "")
    for sentence in split_sentences(text):
        yield from rows_from_sentence(chunk, sentence)


def rows_from_sentence(chunk: dict[str, Any], sentence: str) -> Iterable[dict[str, Any]]:
    for subject in PLANETS:
        subject_re = re.escape(subject)
        for match in re.finditer(
            rf"\b{subject_re}\b\s+(?:currently\s+)?(?:has|have|had|includes?|possesses?)\s+"
            rf"(?:at\s+least\s+)?(?P<count>{COUNT_PHRASE})\s+"
            rf"(?P<claim>(?:(?:known|confirmed|officially\s+confirmed|natural|small|relatively|awaiting|official|recognition|and|naming)\s+){{0,8}})"
            rf"{MOON_TERMS}\b",
            sentence,
            re.IGNORECASE,
        ):
            value = parse_count(match.group("count"))
            if value is None:
                continue
            yield text_row(chunk, subject, value, sentence, match.group("claim"), "subject_has_count_moons")

        for match in re.finditer(
            rf"\b(?P<count>{COUNT_PHRASE})\s+"
            rf"(?:(?:known|confirmed|natural|small|relatively)\s+){{0,8}}"
            rf"{MOON_TERMS}\s+of\s+\b{subject_re}\b",
            sentence,
            re.IGNORECASE,
        ):
            value = parse_count(match.group("count"))
            if value is None:
                continue
            yield text_row(chunk, subject, value, sentence, "", "count_moons_of_subject")

    yield from zero_moon_rows(chunk, sentence)
    yield from antecedent_moon_rows(chunk, sentence)


def zero_moon_rows(chunk: dict[str, Any], sentence: str) -> Iterable[dict[str, Any]]:
    pattern = re.compile(
        r"\bneither\s+(?P<left>Mercury|Venus|Earth|Mars|Jupiter|Saturn|Uranus|Neptune)\s+"
        r"nor\s+(?P<right>Mercury|Venus|Earth|Mars|Jupiter|Saturn|Uranus|Neptune)\s+"
        r"(?:has|have|had)\s+(?:any|no)\s+moons?\b",
        re.IGNORECASE,
    )
    for match in pattern.finditer(sentence):
        for subject in (match.group("left"), match.group("right")):
            yield text_row(chunk, title_case_planet(subject), 0, sentence, "", "neither_subject_nor_subject_has_moons")


def antecedent_moon_rows(chunk: dict[str, Any], sentence: str) -> Iterable[dict[str, Any]]:
    if not re.search(MOON_TERMS, sentence, re.IGNORECASE):
        return
    for match in re.finditer(
        rf"\b(?P<subject>{'|'.join(PLANETS)})\b\s+(?:has|have|had)\s+"
        rf"(?:its\s+)?(?P<count>{COUNT_PHRASE})\s+"
        rf"(?:(?:known|confirmed|natural|small|relatively)\s+){{0,8}}"
        rf"(?:{MOON_TERMS})?\b",
        sentence,
        re.IGNORECASE,
    ):
        value = parse_count(match.group("count"))
        if value is None:
            continue
        subject = title_case_planet(match.group("subject"))
        yield text_row(chunk, subject, value, sentence, "", "moon_antecedent_count")


def text_row(
    chunk: dict[str, Any],
    subject: str,
    value: int,
    evidence: str,
    claim_text: str,
    method: str,
) -> dict[str, Any]:
    observed_at, date_status = nearby_date(evidence)
    claim = "known_moons" if "known" in claim_text.lower() else "confirmed_moons" if "confirmed" in claim_text.lower() else "moon_count"
    return base_row(
        chunk,
        subject=subject,
        value=value,
        claim_type=claim,
        evidence_type="text_extracted_count_assertion",
        validation_status="source_asserted",
        observed_at=observed_at,
        valid_until_exclusive="",
        interval_semantics="",
        date_status=date_status,
        extraction_method=method,
        evidence=evidence,
    )


def base_row(
    chunk: dict[str, Any],
    *,
    subject: str,
    value: Any,
    claim_type: str,
    evidence_type: str,
    validation_status: str,
    observed_at: str,
    valid_until_exclusive: str,
    interval_semantics: str,
    date_status: str,
    extraction_method: str,
    evidence: str,
) -> dict[str, Any]:
    return {
        "subject": subject,
        "predicate": "moon_count",
        "value": value,
        "observed_at": observed_at,
        "valid_until_exclusive": valid_until_exclusive,
        "date_status": date_status,
        "interval_semantics": interval_semantics,
        "claim_type": claim_type,
        "evidence_type": evidence_type,
        "validation_status": validation_status,
        "source_id": chunk.get("source_id", ""),
        "title": chunk.get("title", ""),
        "chunk_id": chunk.get("chunk_id", ""),
        "page_start": chunk.get("page_start", ""),
        "page_end": chunk.get("page_end", ""),
        "section": chunk.get("section", ""),
        "extraction_method": extraction_method,
        "evidence": compact(evidence),
    }


def parse_count(value: str) -> int | None:
    tokens = re.findall(r"\d[\d,]*|[a-z]+", value.lower().replace("-", " "))
    return parse_count_phrase(tokens)


def nearby_date(text: str) -> tuple[str, str]:
    patterns = [
        rf"\bas\s+of\s+(?P<date>{DATE_TEXT})\b",
        rf"\bwith\s+(?P<date>{DATE_TEXT})\s+data\b",
        rf"\bupdat\w+\s+[^.]{0,80}?\bwith\s+(?P<date>{DATE_TEXT})\s+data\b",
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            return normalize_date(matches[-1].group("date")), "nearby_text"
    return "", "unknown"


def normalize_date(value: str) -> str:
    text = value.strip().replace(",", "")
    if re.fullmatch(r"\d{4}", text):
        return text
    month_names = {
        name.lower(): index
        for index, name in enumerate(
            ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"],
            start=1,
        )
    }
    match = re.match(r"([A-Za-z]+)(?:\s+\d{1,2}(?:st|nd|rd|th)?)?\s+(\d{4})$", text)
    if not match:
        return text
    month = month_names.get(match.group(1).lower())
    return f"{match.group(2)}-{month:02d}" if month else text


def split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text)
        if sentence.strip()
    ]


def title_case_planet(value: str) -> str:
    lookup = {planet.lower(): planet for planet in PLANETS}
    return lookup.get(value.lower(), value)


def normalize_subject(value: str) -> str:
    for planet in PLANETS:
        if re.fullmatch(rf"{re.escape(planet)}(?:\s+was\s+known\s+to)?", value, re.IGNORECASE):
            return planet
    return value


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def dedupe_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    deduped = []
    for row in rows:
        if row["subject"] not in PLANETS:
            continue
        key = (
            row["subject"],
            row["value"],
            row["observed_at"],
            row["source_id"],
            row["chunk_id"],
            row["evidence"][:180],
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def row_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (row["subject"], str(row["observed_at"] or "9999"), row["source_id"], row["chunk_id"], int(row["value"]))


def timeline_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    observed = row["observed_at"] or "9999"
    return (row["subject"], observed, int(row["value"]), row["source_id"], row["chunk_id"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "subject",
        "predicate",
        "value",
        "observed_at",
        "valid_until_exclusive",
        "date_status",
        "interval_semantics",
        "claim_type",
        "evidence_type",
        "validation_status",
        "source_id",
        "title",
        "chunk_id",
        "page_start",
        "page_end",
        "section",
        "extraction_method",
        "evidence",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
