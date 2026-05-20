"""Source manifest helpers for RAG ingestion."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RagSource:
    source_id: str
    url: str
    cleaner: str
    domain: str = "astronomy"
    target_questions: list[int] = field(default_factory=list)
    needed_evidence: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RagSource":
        return cls(
            source_id=data["source_id"],
            url=data["url"],
            cleaner=data["cleaner"],
            domain=data.get("domain", "astronomy"),
            target_questions=list(data.get("target_questions", [])),
            needed_evidence=list(data.get("needed_evidence", [])),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "url": self.url,
            "cleaner": self.cleaner,
            "domain": self.domain,
            "target_questions": self.target_questions,
            "needed_evidence": self.needed_evidence,
        }


def load_sources(path: str | Path) -> list[RagSource]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)
    return [RagSource.from_dict(item) for item in data]

