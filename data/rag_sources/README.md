# RAG Sources

This folder stores raw and cleaned source materials for the RAG prototype.

Recommended initial questions:
- Q9: Which dwarf planet located in the Kuiper Belt was discovered first?
- Q11: Which planets orbit beyond Earth yet have fewer moons than Jupiter?
- Q15: As of November 2021, how many confirmed moons did Saturn have?

Subfolders:
- `docs_raw/`: copied plain-text versions of your source documents
- `docs_clean/`: cleaned text generated from web/PDF sources
- `web_raw/`: cached raw HTML/API responses used to reproduce cleaned outputs
- `pdf_raw/`: exported PDF versions of the same documents
- `notes/`: optional planning or source notes

Suggested file naming:
- `rag_doc_01_pluto_kuiper`
- `rag_doc_02_planets_moons`
- `rag_doc_03_saturn_moons_2021`
- `rag_doc_04_mixed_astronomy_facts`
- `rag_doc_05_temporal_astronomy_notes`

Wikipedia cleaning:

```powershell
python scripts\clean_wikipedia_source.py --url "https://en.wikipedia.org/wiki/Saturn" --source-id wiki_saturn
```

The script uses `external/Wikipedia_text_extractor` by default. Set
`WIKIPEDIA_TEXT_EXTRACTOR_DIR` or pass `--cleaner-dir` if the cleaner lives in a
different folder.

Q9 RAG+LLM vertical slice:

```powershell
python scripts\ingest_rag_sources.py --sources data\rag_sources\sources_master.json
python scripts\build_rag_index.py
python scripts\preview_rag_context.py --id 9
python scripts\preview_rag_context.py --id 9 --retrieval-mode auto-source
python main_rag.py --ids 9 11 --retrieval-mode auto-source
```
