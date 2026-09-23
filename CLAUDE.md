# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Structured entity extraction from Brazilian investment fund regulatory PDFs (filed with **CVM**, the Brazilian SEC equivalent, via the **B3** exchange's Fundos.NET system), using Google's [LangExtract](https://github.com/google/langextract) library. Extractions are source-grounded (every value links back to an exact text span in the original PDF). Built for investigative journalism use (e.g. red-flag detection in FIDC quarterly reports).

Despite the README describing a `scripts/` subdirectory, all scripts currently live at the repo root. `docs/` holds prose only — currently the source of record for upstream issue [#545](https://github.com/google/langextract/issues/545).

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

Five files, no packages:

- **`utils.py`** — shared foundation for everything: `.env`/API key loading (`load_config`), model config dispatch (`configure_model` — routes `gemini-*`, `gpt-*`, `ollama:*` model strings to the right LangExtract backend), PDF-to-text extraction with smart section reduction (`extract_pdf_text`), and HTML report generation.
- **`extrair_regulamento.py`** — extracts from fund regulations (Regulamento).
- **`extrair_demonstrativo.py`** — extracts from FIDC quarterly reports (Demonstrativo Trimestral), plus `classify_alerts()` for automatic red-flag detection (backing inconsistencies, lawsuits, financial disruptions, etc.).
- **`comparar_versoes.py`** — A/B harness for langextract/model swaps; not part of the extraction pipeline (see "Evaluating a model or library swap" below).
- **`extrair_lote.py`** — batch driver: `detect_document_type`/`detect_type_from_filename` route each PDF to the right extractor, then `generate_comparative_report` merges results across a directory.

### The core problem this codebase solves: rate limits

Gemini free tier = 15 requests/minute. Naively chunking a 250K-char PDF into 3K-char pieces means ~46 API calls, which blows through the limit and triggers slow retry/backoff. `utils.extract_pdf_text` avoids this by keeping only regulator-relevant sections before sending text to the LLM — see `SECTION_KEYWORDS` and `_extract_relevant_sections` in `utils.py`. That function splits Brazilian regulations by their standard numbered clause headers (e.g. `7. TAXA DE ADMINISTRAÇÃO`, per CVM Resolution 175/2022 structure), always keeps the first/last sections and any section matching keep-headers or content keywords, and drops/trims the rest. This is the single most load-bearing piece of logic in the repo — don't casually rewrite it without understanding why (see comments in `_extract_relevant_sections`).

**Retry/checkpoint is layered, not redundant** (as of `langextract==1.6.0` installed / `v1.7.0` released, 2026-09-23):
1. **Native per-chunk retry** (`GeminiLanguageModel._process_single_prompt`, upstream commit `3aab86c`) retries individual HTTP calls on 429/503/timeout — but only 3 attempts, 16s delay cap, and when exhausted it kills the *entire* parallel `infer()` batch (all chunks in that `lx.extract()` call), not just the failed one.
2. **`utils.extract_with_backoff()`** wraps a whole `lx.extract()` call (one of the 3 prompt groups) with coarser backoff (5 attempts, 5s·2ⁿ) — this is the fallback for when native retry exhausts under sustained load (e.g. free-tier 15 RPM), which the 16s-cap native retry alone doesn't cover.
3. **`utils.save_checkpoint()`** persists progress between the 3 group-level `lx.extract()` calls (A/B/C) that this repo's own script orchestrates — a layer upstream has no visibility into, since each group is a separate `extract()` invocation, not chunks within one.

Upstream will not close these gaps for you. PR [#520](https://github.com/google/langextract/pull/520) (`max_rpm` throttling + partial-chunk preservation *within* a single `extract()` call) was **closed unmerged** on 2026-09-20 — so layers 2 and 3 above are permanent, not a stopgap waiting on upstream. PR [#509](https://github.com/google/langextract/pull/509) (merged, already in 1.6.0) only forwards `max_output_tokens`/`top_p`/`top_k`; this repo doesn't set those yet. PR [#521](https://github.com/google/langextract/pull/521) (merged 2026-09-20, *after* the v1.7.0 cut, so not in any release yet) reworded the resolver's silent-chunk-drop warning and added a `Note:` to `extract()`'s docstring naming `ResolverParsingError` as the stable contract — that type, not the message text, is what this repo matches on (`_is_parse_error` in both extractors). Issue [#358](https://github.com/google/langextract/issues/358), the original report behind all of the above, was closed 2026-09-20 as too broad; the maintainer asked for narrow follow-up issues naming a concrete example and provider. Relevance-aware chunking — what `_extract_relevant_sections` does here — is the one item from #358 never addressed upstream; it was filed separately as [#545](https://github.com/google/langextract/issues/545) on 2026-09-23 (draft and measured figures in `docs/issue_chunking_draft.md`).

### Evaluating a model or library swap

`comparar_versoes.py` is the A/B harness for changing langextract versions or models. It exists because `output/<stem>_report.json` is a lossy projection — `entities` is `dict[class -> list[str]]`, so `char_interval` and `alignment_status` are discarded and an alignment regression is invisible in it. The harness runs the same production path (`extract_with_backoff` -> `lx.extract`) but keeps the full `Extraction` objects, and pins `temperature=0.0` (production leaves it at the provider default).

```bash
python comparar_versoes.py run <pdf> --label smoke --groups A --max-chars 6000   # ~2 calls, validates the dump
python comparar_versoes.py run <pdf> --label v1_7_0_a                            # ~48 calls
python comparar_versoes.py diff output/ab/v1_7_0_a.json output/ab/v1_7_0_b.json
```

**Always run two arms of the SAME configuration first.** Measured on 2026-09-23, two identical runs on 1.6.0 (same PDF, same reduced-text sha256, `temperature=0.0`) gave 58 vs 25 entities, 14/23 vs 7/23 class coverage, and one of them lost group C entirely to a `ResolverParsingError` that exhausted all three chunk sizes. The run-to-run noise floor is larger than any version effect seen so far, so a single arm per version proves nothing. `diff` refuses to call a comparison trustworthy when an arm has a failed group or a different input hash.

The 1.6.0 -> 1.7.0 evaluation that motivated this: every cross-version delta landed inside the same-version noise band, so `requirements.txt` moved to `>=1.7.0` on "no detectable regression", not on measured improvement. Upstream PR #485's fuzzy->exact alignment shift is visible in the right direction but is not separable from noise at n=2. The predicted PR #534 hazard — a truncation surfacing as `InferenceRuntimeError`, which `_is_parse_error` does not match, silently disabling the chunk-halving retry — did **not** occur in any 1.7.0 arm; every failure observed was a correctly typed `ResolverParsingError`. That is two runs' worth of absence, not proof, so re-check it if groups start failing without the retry chain firing.

**Known data-quality issue this surfaced (unfixed):** group C returns `""` and the literal string `"null"` for classes it cannot find — 19 of 58 extractions in one arm. These flow into `report["entities"]` and into the HTML a journalist reads, as if they were extracted values. `comparar_versoes.PLACEHOLDER_TEXTS` treats them as blanks for measurement purposes, but nothing filters them out of the actual reports.

CLI presets (`--fast`, default, `--max-chars`, `--no-filter`) all just tune `max_chars`/`chunk_size`/`workers` passed into this pipeline; they don't change the extraction logic itself.

### Multi-group extraction pattern

`extrair_regulamento.py` splits its ~22 entity types into 3 prompt/example groups (A: identification/providers, B: fees/structure, C: policy/risk/events), each run as a separate `lx.extract()` call. This exists because asking for all 22 entities in one call produces JSON output long enough to get truncated by the model (`Unterminated string` errors). Each group has its own `PROMPT_GROUP_*` and a hand-written `lx.data.ExampleData` few-shot example — these are the actual extraction schema definitions; there's no separate schema file. If adding new entity fields, add to the appropriate group's prompt AND its example, and expect this to roughly multiply the API call count for that document.

Extraction calls retry at progressively smaller `max_char_buffer` (chunk size) on JSON parse errors before giving up on a group.

### Output convention

Every script writes `output/<pdf_stem>_report.json` (primary, structured by entity type) and `output/<pdf_stem>_report.html` (visualization). Always call `ensure_output_dir`/`safe_save_documents` from `utils.py` before writing — LangExtract's own I/O does not create parent directories and will raise `FileNotFoundError` otherwise.

### Working in Portuguese

Extraction prompts, keyword lists, and entity class names (`cnpj_fundo`, `taxa_administracao`, etc.) are intentionally in Portuguese because source documents are Portuguese-language Brazilian regulatory filings. Keep this convention when extending entity types — don't translate extraction_class names to English.
