# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Structured entity extraction from Brazilian investment fund regulatory PDFs (filed with **CVM**, the Brazilian SEC equivalent, via the **B3** exchange's Fundos.NET system), using Google's [LangExtract](https://github.com/google/langextract) library. Extractions are source-grounded (every value links back to an exact text span in the original PDF). Built for investigative journalism use (e.g. red-flag detection in FIDC quarterly reports).

Despite the README describing a `scripts/` subdirectory, all scripts currently live at the repo root.

## Setup and commands

```bash
# Install
pip install -r requirements.txt   # langextract, pdfplumber, python-dotenv

# API key: create .env in repo root
echo 'GOOGLE_API_KEY=your-key-here' > .env

# Verify config
python test_config.py

# Smoke test any script
python -m py_compile extrair_regulamento.py

# Run extraction
python extrair_regulamento.py regulation.pdf [--fast|--no-filter] [--model gemini-2.5-flash] [--workers N] [--max-chars N]
python extrair_demonstrativo.py quarterly_report.pdf
python extrair_lote.py ./documents/ --compare
```

No test suite beyond `test_config.py` (checks API key presence). No linter configured. Always test with a single small PDF before batch runs.

## Architecture

Four files, no packages:

- **`utils.py`** — shared foundation for everything: `.env`/API key loading (`load_config`), model config dispatch (`configure_model` — routes `gemini-*`, `gpt-*`, `ollama:*` model strings to the right LangExtract backend), PDF-to-text extraction with smart section reduction (`extract_pdf_text`), and HTML report generation.
- **`extrair_regulamento.py`** — extracts from fund regulations (Regulamento).
- **`extrair_demonstrativo.py`** — extracts from FIDC quarterly reports (Demonstrativo Trimestral), plus `classify_alerts()` for automatic red-flag detection (backing inconsistencies, lawsuits, financial disruptions, etc.).
- **`extrair_lote.py`** — batch driver: `detect_document_type`/`detect_type_from_filename` route each PDF to the right extractor, then `generate_comparative_report` merges results across a directory.

### The core problem this codebase solves: rate limits

Gemini free tier = 15 requests/minute. Naively chunking a 250K-char PDF into 3K-char pieces means ~46 API calls, which blows through the limit and triggers slow retry/backoff. `utils.extract_pdf_text` avoids this by keeping only regulator-relevant sections before sending text to the LLM — see `SECTION_KEYWORDS` and `_extract_relevant_sections` in `utils.py`. That function splits Brazilian regulations by their standard numbered clause headers (e.g. `7. TAXA DE ADMINISTRAÇÃO`, per CVM Resolution 175/2022 structure), always keeps the first/last sections and any section matching keep-headers or content keywords, and drops/trims the rest. This is the single most load-bearing piece of logic in the repo — don't casually rewrite it without understanding why (see comments in `_extract_relevant_sections`).

CLI presets (`--fast`, default, `--max-chars`, `--no-filter`) all just tune `max_chars`/`chunk_size`/`workers` passed into this pipeline; they don't change the extraction logic itself.

### Multi-group extraction pattern

`extrair_regulamento.py` splits its ~22 entity types into 3 prompt/example groups (A: identification/providers, B: fees/structure, C: policy/risk/events), each run as a separate `lx.extract()` call. This exists because asking for all 22 entities in one call produces JSON output long enough to get truncated by the model (`Unterminated string` errors). Each group has its own `PROMPT_GROUP_*` and a hand-written `lx.data.ExampleData` few-shot example — these are the actual extraction schema definitions; there's no separate schema file. If adding new entity fields, add to the appropriate group's prompt AND its example, and expect this to roughly multiply the API call count for that document.

Extraction calls retry at progressively smaller `max_char_buffer` (chunk size) on JSON parse errors before giving up on a group.

### Output convention

Every script writes `output/<pdf_stem>_report.json` (primary, structured by entity type) and `output/<pdf_stem>_report.html` (visualization). Always call `ensure_output_dir`/`safe_save_documents` from `utils.py` before writing — LangExtract's own I/O does not create parent directories and will raise `FileNotFoundError` otherwise.

### Working in Portuguese

Extraction prompts, keyword lists, and entity class names (`cnpj_fundo`, `taxa_administracao`, etc.) are intentionally in Portuguese because source documents are Portuguese-language Brazilian regulatory filings. Keep this convention when extending entity types — don't translate extraction_class names to English.
