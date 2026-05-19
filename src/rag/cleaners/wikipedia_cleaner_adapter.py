"""Adapter for the external Wikipedia text extractor.

The external project is intentionally kept behind this small interface so the
RAG ingestion code can depend on stable input/output paths instead of the
cleaner's internal CLI layout.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType
from typing import Any
from urllib.parse import unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CLEANER_DIR = PROJECT_ROOT / "external" / "Wikipedia_text_extractor"
DEFAULT_DOCS_CLEAN_DIR = PROJECT_ROOT / "data" / "rag_sources" / "docs_clean" / "wikipedia"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "rag_sources" / "web_raw" / "wikipedia"


@dataclass
class WikipediaCleanResult:
    source_id: str
    url: str | None
    title: str
    lang: str
    clean_text_path: str
    raw_html_path: str | None
    references_text_path: str | None
    math_mode: str
    content_hash: str
    char_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


def resolve_cleaner_dir(cleaner_dir: str | Path | None = None) -> Path:
    """Resolve and validate the external Wikipedia cleaner directory."""
    candidate = (
        Path(cleaner_dir)
        if cleaner_dir is not None
        else Path(os.environ.get("WIKIPEDIA_TEXT_EXTRACTOR_DIR", DEFAULT_CLEANER_DIR))
    )
    candidate = candidate.expanduser().resolve()
    expected_file = candidate / "wiki_text_extractor.py"
    if not expected_file.exists():
        raise FileNotFoundError(
            "Wikipedia_text_extractor was not found. "
            "Clone https://github.com/Ahmed-Estiak/Wikipedia_text_extractor.git "
            f"to {DEFAULT_CLEANER_DIR} or set WIKIPEDIA_TEXT_EXTRACTOR_DIR."
        )
    return candidate


def source_id_from_url(url: str) -> str:
    """Create a stable filesystem-safe source id from a Wikipedia URL."""
    parsed = urlparse(url)
    title = unquote(parsed.path.rstrip("/").split("/")[-1] or parsed.netloc)
    return _safe_source_id(title)


def source_id_from_title(title: str) -> str:
    """Create a stable filesystem-safe source id from a Wikipedia title."""
    return _safe_source_id(title)


def clean_wikipedia_page(
    *,
    url: str | None = None,
    title: str | None = None,
    lang: str = "en",
    source_id: str | None = None,
    math_mode: str = "remove",
    docs_clean_dir: str | Path = DEFAULT_DOCS_CLEAN_DIR,
    raw_dir: str | Path = DEFAULT_RAW_DIR,
    cleaner_dir: str | Path | None = None,
    save_raw: bool = True,
    save_references: bool = False,
    references_end_only: bool = False,
) -> WikipediaCleanResult:
    """Fetch and clean one Wikipedia page into the RAG source directory.

    Exactly one of ``url`` or ``title`` must be provided.
    """
    if bool(url) == bool(title):
        raise ValueError("Provide exactly one of url or title.")
    if math_mode not in {"remove", "latex", "keep"}:
        raise ValueError("math_mode must be one of: remove, latex, keep.")

    cleaner_root = resolve_cleaner_dir(cleaner_dir)
    cleaner = _load_cleaner_module(cleaner_root)
    page = cleaner.page_request_from_url(url) if url else cleaner.PageRequest(title, lang)
    resolved_source_id = source_id or (source_id_from_url(url) if url else source_id_from_title(title or page.title))

    html = cleaner.fetch_page_html(page)
    clean_text = cleaner.add_topic_heading(
        cleaner.clean_wikipedia_html(html, math_mode),
        page,
    )

    clean_path = Path(docs_clean_dir) / f"{resolved_source_id}.txt"
    _write_text(clean_path, clean_text)

    raw_path = None
    if save_raw:
        raw_path = Path(raw_dir) / f"{resolved_source_id}.html"
        _write_text(raw_path, html)

    references_path = None
    if save_references:
        references_text = cleaner.clean_wikipedia_html_with_references(
            html,
            math_mode,
            include_inline_markers=not references_end_only,
        )
        references_text = cleaner.add_topic_heading(references_text, page)
        references_path = Path(docs_clean_dir) / f"{resolved_source_id}_references.txt"
        _write_text(references_path, references_text)

    content_hash = hashlib.sha256(clean_text.encode("utf-8")).hexdigest()
    return WikipediaCleanResult(
        source_id=resolved_source_id,
        url=url,
        title=page.title,
        lang=page.lang,
        clean_text_path=str(clean_path),
        raw_html_path=str(raw_path) if raw_path else None,
        references_text_path=str(references_path) if references_path else None,
        math_mode=math_mode,
        content_hash=content_hash,
        char_count=len(clean_text),
    )


def _load_cleaner_module(cleaner_dir: Path) -> ModuleType:
    module_path = cleaner_dir / "wiki_text_extractor.py"
    module_name = "_rag_external_wiki_text_extractor"
    existing_module = sys.modules.get(module_name)
    if existing_module is not None:
        return existing_module

    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load cleaner module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _safe_source_id(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-").lower()
    return value or "wikipedia_page"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

