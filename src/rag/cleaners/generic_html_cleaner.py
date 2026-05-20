"""Small dependency-free HTML cleaner for non-Wikipedia RAG sources."""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen


DEFAULT_USER_AGENT = "HallucinationsInLLMsRAG/0.1 (+https://github.com/Ahmed-Estiak)"


@dataclass
class GenericHtmlCleanResult:
    source_id: str
    url: str
    title: str
    clean_text_path: str
    raw_html_path: str
    content_hash: str
    char_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class ReadableTextParser(HTMLParser):
    """Extract readable title/headings/paragraph/list/table text from HTML."""

    BLOCK_TAGS = {
        "article",
        "caption",
        "dd",
        "div",
        "dt",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "p",
        "td",
        "th",
        "title",
        "tr",
    }
    SKIP_TAGS = {
        "aside",
        "canvas",
        "footer",
        "form",
        "header",
        "iframe",
        "nav",
        "noscript",
        "script",
        "style",
        "svg",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip_depth = 0
        self._tag_stack: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in self.SKIP_TAGS:
            self._skip_depth += 1
        self._tag_stack.append(tag)
        if self._skip_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self._skip_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")
        if tag in self.SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if self._tag_stack:
            self._tag_stack.pop()

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        text = html.unescape(data).strip()
        if not text:
            return
        if self._tag_stack and self._tag_stack[-1] == "title":
            self.title_parts.append(text)
        self.parts.append(text)
        self.parts.append(" ")

    @property
    def title(self) -> str:
        return normalize_text(" ".join(self.title_parts)).split(" | ")[0].strip()

    @property
    def text(self) -> str:
        return normalize_text("".join(self.parts))


def clean_generic_html_source(
    *,
    url: str,
    source_id: str,
    docs_clean_dir: str | Path,
    raw_dir: str | Path,
) -> GenericHtmlCleanResult:
    raw_html = fetch_url(url)
    parser = ReadableTextParser()
    parser.feed(raw_html)

    title = parser.title or source_id
    clean_text = parser.text
    clean_text = f"{title}\n\n{clean_text}".strip() + "\n"

    clean_path = Path(docs_clean_dir) / f"{source_id}.txt"
    raw_path = Path(raw_dir) / f"{source_id}.html"
    write_text(clean_path, clean_text)
    write_text(raw_path, raw_html)

    return GenericHtmlCleanResult(
        source_id=source_id,
        url=url,
        title=title,
        clean_text_path=str(clean_path),
        raw_html_path=str(raw_path),
        content_hash=hashlib.sha256(clean_text.encode("utf-8")).hexdigest(),
        char_count=len(clean_text),
    )


def fetch_url(url: str, timeout: int = 30) -> str:
    request = Request(url, headers={"User-Agent": DEFAULT_USER_AGENT})
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        return response.read().decode(charset, errors="replace")


def normalize_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")

