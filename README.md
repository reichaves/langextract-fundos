# LangExtract for Brazilian Investment Fund Documents

Structured entity extraction from Brazilian investment fund regulatory PDFs using [LangExtract](https://github.com/google/langextract) (Google, Apache 2.0).

## What is this?

Brazil's securities regulator (**CVM** — similar to the US SEC) requires investment funds to file standardized documents through **B3** (the Brazilian stock exchange). These PDFs contain critical information about fund structure, fees, risks, and service providers — but it's buried in dense legal Portuguese text.

This tool uses LLMs (via Google's LangExtract library) to automatically extract structured data from these documents, with source text grounding (every extracted entity links back to the exact text span in the original PDF).

### Document types supported

| Document | Portuguese name | What it is | Script |
|----------|----------------|-----------|--------|
| **Fund Regulation** | Regulamento | Fund's charter/prospectus — defines rules, fees, eligible assets, risk factors | `extrair_regulamento.py` |
| **Quarterly Report** | Demonstrativo Trimestral | Mandatory FIDC disclosure — backing verification, lawsuits, defaults, originator concentration | `extrair_demonstrativo.py` |
| **Batch processing** | — | Process multiple PDFs with auto type detection | `extrair_lote.py` |

### What gets extracted

**From regulations:**
Fund name, CNPJ (tax ID), fund type (FIDC/FIAGRO/FII/FIP), administrator, manager, custodian, auditor, fees (admin/management/performance/custody), duration, open/closed-end status, target investors, share classes, eligible assets, concentration limits, risk factors, evaluation events, liquidation events, legal forum.

**From quarterly reports:**
Fund ID, reference quarter, backing verification result, lawsuits, policy changes, originator concentration, collateral changes, credit assignment details, prepayment impact, financial flow disruptions. **Plus automatic red flag detection** for investigative journalism.

## Installation

```bash
# Create virtual environment
python -m venv langextract_env
source langextract_env/bin/activate  # Linux/Mac
# langextract_env\Scripts\activate   # Windows

# Install dependencies
pip install langextract pdfplumber python-dotenv

# Set up API key (get one at https://aistudio.google.com/app/apikey)
cp .env.example .env
# then edit .env and add your GOOGLE_API_KEY
```

See [`.env.example`](.env.example) for all supported environment variables (Gemini, optional OpenAI, optional local Ollama URL).

## Quick start

```bash
# Extract from a fund regulation (default: ~2 min)
python scripts/extrair_regulamento.py regulation.pdf

# Fast mode (~1 min, less recall)
python scripts/extrair_regulamento.py regulation.pdf --fast

# Quarterly report with red flag detection
python scripts/extrair_demonstrativo.py quarterly_report.pdf

# Batch processing with comparative report
python scripts/extrair_lote.py ./documents/ --compare
```

## Output files

Each extraction produces 3 files in `output/`:

| File | Description |
|------|-------------|
| `*_extractions.jsonl` | Structured data with character positions (LangExtract native format) |
| `*_visualization.html` | Interactive visualization — open in browser to see highlighted entities |
| `*_report.json` | Consolidated JSON report grouped by entity type |

## Performance

### The rate limit problem

The Gemini free tier allows **15 requests per minute**. A 250K-character PDF split into 3K-char chunks = 46 chunks = 46 API calls. With parallel workers hitting the rate limit, the API returns 429 errors and LangExtract retries with backoff — turning a 2-minute job into a 30+ minute crawl.

### The solution: smart text reduction

Instead of processing the full 250K characters, we extract only the **relevant sections** (~50K chars) before sending to the LLM:

1. **Always keep** the first ~12K chars (fund ID, service providers, structure)
2. **Always keep** the last ~5K chars (legal forum, general provisions)
3. **In the middle**, keep only paragraphs containing regulatory keywords (fees, risks, limits, etc.)
4. **Result**: 250K → ~50K chars = **6 chunks = 6 API calls = ~2 minutes**

### Performance presets

| Preset | API calls | Time | Text processed | Use case |
|--------|-----------|------|----------------|----------|
| `--fast` | ~4 | ~1 min | 15K chars | Quick exploration |
| Default | ~6 | ~2 min | 50K chars | Standard extraction |
| `--max-chars 80000` | ~16 | ~10 min | 80K chars | Higher recall (paid tier) |
| `--no-filter` | ~46 | 30+ min | Full text | Complete extraction |

### For paid API tiers

If you have a paid Gemini API key with higher rate limits:

```bash
# More workers + more text = better recall, still fast
python scripts/extrair_regulamento.py regulation.pdf --workers 10 --max-chars 80000
```

## CLI options

All scripts share these options:

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `gemini-2.0-flash` | LLM model (`gemini-2.0-flash`, `gpt-4o`, `ollama:model`) |
| `--passes` | `1` | Extraction passes (2 = higher recall, 2× slower) |
| `--workers` | `3` | Parallel workers (keep at 3 for free tier, increase for paid) |
| `--output` | `output` | Output directory |
| `--chunk-size` | `3000` | Characters per LLM chunk |
| `--max-chars` | `30000` | Max text after smart filtering (0 = no limit) |
| `--fast` | — | 1 pass, chunk 3K, max 15K chars |
| `--no-filter` | — | Process full text (slow, avoids losing any information) |

## Red flag detection (quarterly reports)

The quarterly report script automatically flags potential issues:

| Severity | Type | What it means |
|----------|------|---------------|
| 🔴 CRITICAL | `BACKING_INCONSISTENCY` | Credit rights may lack proper collateral |
| 🟠 HIGH | `LAWSUIT` | Active lawsuits involving the fund |
| 🟠 HIGH | `FINANCIAL_DISRUPTION` | Defaults or bankruptcies affecting cash flows |
| 🟡 MEDIUM | `POLICY_CHANGE` | Investment policy was modified |
| 🟡 MEDIUM | `ORIGINATOR_CONCENTRATION` | Single originator >10% of portfolio |

## Glossary for international users

| Portuguese term | English equivalent | Description |
|----------------|-------------------|-------------|
| **CVM** | SEC (US) / FCA (UK) | Brazilian securities regulator |
| **B3** | NYSE/Nasdaq | Brazilian stock exchange |
| **FIDC** | CLO / ABS trust | Credit Rights Investment Fund (securitization) |
| **FIAGRO** | Agricultural fund | Agribusiness Investment Fund |
| **FII** | REIT | Real Estate Investment Fund |
| **FIP** | PE fund | Private Equity Investment Fund |
| **CNPJ** | EIN (US) / CRN (UK) | Brazilian tax ID for entities |
| **Regulamento** | Prospectus / Charter | Fund's governing document |
| **Demonstrativo Trimestral** | Quarterly report | Mandatory FIDC disclosure |
| **Administrador** | Trustee / Administrator | Fiduciary administrator |
| **Gestor** | Asset manager | Portfolio manager |
| **Custodiante** | Custodian | Asset custodian |
| **Direitos creditórios** | Credit rights / Receivables | The underlying assets in a FIDC |
| **Lastro** | Backing / Collateral | Proof that credit rights exist |
| **Cotas** | Shares / Units | Fund participation units |

## Where to get the PDFs

Fund documents are publicly available at:

1. **Fundos.NET** (official CVM/B3 system): https://fnet.bmfbovespa.com.br/fnet/publico/abrirGerenciadorDocumentosCVM
2. Search by fund CNPJ or name
3. Download regulation ("Regulamento") or quarterly report ("Informe Periódico") PDFs

## Project structure

```
langextract-fundos/
├── README.md
├── requirements.txt
├── .env                       # API keys (create this, not in git)
└── scripts/
    ├── utils.py               # Shared: config, PDF extraction, text reduction
    ├── extrair_regulamento.py # Fund regulation extraction
    ├── extrair_demonstrativo.py # Quarterly report extraction + alerts
    ├── extrair_lote.py        # Batch processing + comparative reports
    └── test_config.py         # API key verification
```

## References

- [LangExtract](https://github.com/google/langextract) — Google's structured extraction library
- [CVM Resolution 175](https://conteudo.cvm.gov.br/legislacao/resolucoes/resol175.html) — Brazilian fund regulation framework
- [Fundos.NET](https://fnet.bmfbovespa.com.br/fnet/publico/abrirGerenciadorDocumentosCVM) — Public document repository

## License

Scripts: MIT | LangExtract: Apache 2.0 (Google)
