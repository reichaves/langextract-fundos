"""
utils.py — Shared utilities for LangExtract fund document extraction.

Provides:
- API key configuration (with .env support)
- PDF text extraction with smart reduction for regulatory documents
- LLM model configuration (Gemini, OpenAI, Ollama)
- Safe output directory creation (prevents FileNotFoundError)
- Processing time estimation
"""

import os
import re
from pathlib import Path

import pdfplumber
import langextract as lx
from dotenv import load_dotenv


# ============================================================
# Configuration
# ============================================================


def load_config():
    """
    Load environment variables from .env file.

    LangExtract reads LANGEXTRACT_API_KEY for Gemini models.
    This function maps GOOGLE_API_KEY -> LANGEXTRACT_API_KEY automatically.
    """
    load_dotenv()

    google_key = os.getenv("GOOGLE_API_KEY")
    lx_key = os.getenv("LANGEXTRACT_API_KEY")

    if google_key and not lx_key:
        os.environ["LANGEXTRACT_API_KEY"] = google_key

    if not any(os.getenv(k) for k in ["LANGEXTRACT_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"]):
        print("⚠️  No API key found. Set GOOGLE_API_KEY, OPENAI_API_KEY, or LANGEXTRACT_API_KEY.")
        print("   Create a .env file or export the variable in your shell.")


def configure_model(model_str: str) -> dict:
    """
    Return LangExtract keyword arguments for the given model string.

    Supported formats:
        gemini-2.5-flash          -> Google Gemini (recommended for production)
        gemini-3-flash-preview    -> Google Gemini preview (may have lower rate limits)
        gpt-4o                    -> OpenAI
        ollama:gemma2:9b          -> Local Ollama model (no API costs)
    """
    config = {}

    if model_str.startswith("ollama:"):
        config["model_id"] = model_str.replace("ollama:", "")
        config["language_model_type"] = lx.inference.OllamaLanguageModel
        config["model_url"] = os.environ.get("OLLAMA_URL", "http://localhost:11434")
        config["fence_output"] = False
        config["use_schema_constraints"] = False
    elif model_str.startswith("gpt"):
        config["model_id"] = model_str
        config["fence_output"] = True
        config["use_schema_constraints"] = False
    else:
        config["model_id"] = model_str  # Gemini (default)

    return config


# ============================================================
# Safe output directory creation
# ============================================================


def ensure_output_dir(filepath: str):
    """
    Create all parent directories for a file path if they don't exist.

    CRITICAL: lx.io.save_annotated_documents() does NOT create directories.
    It raises FileNotFoundError if the parent dir is missing.
    Always call this before writing any file.
    """
    parent = Path(filepath).parent
    parent.mkdir(parents=True, exist_ok=True)


def safe_save_documents(documents, output_name: str):
    """
    Wrapper around lx.io.save_annotated_documents that guarantees
    the output directory exists before writing.

    LangExtract may internally resolve the path differently, so we
    create the directory both for the path we pass AND for the
    filename alone (in case LangExtract strips the directory).
    """
    # Create the directory for the full path we're passing
    ensure_output_dir(output_name)

    # Also create the directory at cwd level in case LangExtract
    # only uses the basename internally
    abs_path = os.path.abspath(output_name)
    ensure_output_dir(abs_path)

    lx.io.save_annotated_documents(documents, output_name=abs_path)


# ============================================================
# Keywords for identifying relevant sections in fund documents
# ============================================================

# Brazilian regulatory terms that signal sections with extractable entities.
# These are in Portuguese because the source documents are in Portuguese.
SECTION_KEYWORDS = [
    # Fund identification
    "cnpj", "fundo de investimento", "fidc", "fiagro", "fii", "fip",
    # Service providers
    "administrador", "gestor", "custodiante", "auditor", "escriturador",
    "ato declaratório", "prestadores de serviço",
    # Fees
    "taxa de administração", "taxa de gestão", "taxa de performance",
    "taxa de custódia", "remuneração", "encargos",
    # Fund structure
    "prazo de duração", "regime condominial", "público-alvo",
    "investidores profissionais", "investidores qualificados",
    "classe de cotas", "séries", "classe única", "classe sênior",
    "classe subordinada", "classe mezanino",
    # Investment policy
    "direitos creditórios", "limite de concentração",
    "política de investimento", "aplicação mínima", "composição da carteira",
    # Risk factors
    "fatores de risco", "risco de crédito", "risco de mercado",
    "risco de liquidez", "risco operacional",
    # Events
    "evento de avaliação", "evento de liquidação", "liquidação antecipada",
    "amortização", "resgate",
    # Issuance
    "valor da emissão", "oferta", "distribuição", "integralização",
    # Legal forum
    "foro", "comarca", "controvérsia", "arbitragem",
    # Glossary
    "glossário", "definições",
]


# ============================================================
# Text cleaning and smart reduction
# ============================================================


def _clean_basic(text: str) -> str:
    """Remove excessive blank lines and isolated page numbers."""
    lines = text.split("\n")
    result = []
    consecutive_blanks = 0

    for line in lines:
        stripped = line.strip()

        # Skip isolated page numbers (1-3 digits alone on a line)
        if re.match(r"^\d{1,3}$", stripped):
            continue

        if stripped == "":
            consecutive_blanks += 1
            if consecutive_blanks > 1:
                continue
        else:
            consecutive_blanks = 0

        result.append(line)

    return "\n".join(result)


def _remove_headers_footers(text: str, num_pages: int) -> str:
    """
    Remove lines that repeat across many pages (likely headers/footers).
    A line appearing in >25% of pages is considered a header or footer.
    """
    if num_pages < 4:
        return text

    lines = text.split("\n")
    count = {}
    for line in lines:
        stripped = line.strip()
        if 5 < len(stripped) < 200 and not stripped.startswith("---"):
            count[stripped] = count.get(stripped, 0) + 1

    threshold = max(3, num_pages * 0.25)
    repeated = {line for line, n in count.items() if n >= threshold}

    if not repeated:
        return text

    return "\n".join(l for l in lines if l.strip() not in repeated)


def _extract_relevant_sections(text: str, max_chars: int) -> str:
    """
    Keep only sections that contain extractable entities.

    This is the KEY optimization for large documents (200K+ chars).

    CRITICAL FIX: PDF text from pdfplumber uses SINGLE newlines, not double.
    Previous approach split by \\n\\n which produced 1 giant block = no filtering.
    New approach: split by numbered CLAUSE HEADERS (e.g., "7. TAXA DE ADMINISTRAÇÃO")
    which is the standard structure of Brazilian fund regulations.

    Strategy:
    1. Split text by numbered clause headers (regex: line starting with digit+period)
    2. Always keep first section (fund name, CNPJ)
    3. Skip glossary/definitions sections (very long, no extractable entities)
    4. Keep sections whose HEADER or CONTENT matches target keywords
    5. Always keep last section (usually "FORO")
    """
    if len(text) <= max_chars:
        return text

    # Split by major clause headers: lines starting with "N." or "N.N"
    # e.g., "2. CARACTERÍSTICAS GERAIS" or "7. TAXA DE ADMINISTRAÇÃO"
    # Must be at line start (after newline) and followed by text
    section_pattern = re.compile(r'\n(?=\d{1,2}\.\s+[A-ZÁÀÂÃÉÈÊÍÏÓÔÕÚÇ])')
    parts = section_pattern.split(text)

    if len(parts) <= 1:
        # Fallback: couldn't split by sections, use simple head+tail
        half = max_chars // 2
        return (
            text[:half]
            + "\n\n[... intermediate sections omitted ...]\n\n"
            + text[-half:]
        )

    # Classify each section
    SKIP_HEADERS = [
        "glossário", "definições", "glossario", "definicoes",
    ]

    KEEP_HEADERS = [
        # Always keep these section topics
        "características gerais", "caracteristicas gerais",
        "prazo de duração", "prazo de duracao",
        "público-alvo", "publico-alvo", "público alvo",
        "prestadores de serviço", "prestadores de servico",
        "administrador", "gestora", "gestor",
        "custodiante", "custódia", "custodia",
        "auditor",
        "taxa de administração", "taxa de gestão", "taxa de performance",
        "taxa de custódia", "remuneração", "outras taxas",
        "remuneração dos prestadores",
        "classe", "cotas", "subclasses", "séries",
        "política de investimento", "politica de investimento",
        "composição da carteira", "composicao da carteira",
        "direitos creditórios elegíveis", "critérios de elegibilidade",
        "limite", "concentração",
        "fatores de risco", "fator de risco",
        "evento de avaliação", "evento de liquidação",
        "eventos de avaliação", "eventos de liquidação",
        "emissão", "valor da emissão", "oferta",
        "foro", "disposições gerais", "disposições finais",
    ]

    kept_sections = []
    total_kept = 0

    for i, section in enumerate(parts):
        # First 200 chars = header area
        header = section[:200].lower()
        section_lower = section[:2000].lower()  # Check first 2K for content keywords

        is_first = (i == 0)
        is_last = (i == len(parts) - 1)

        # Skip glossary (typically 10-15K chars of definitions)
        is_glossary = any(h in header for h in SKIP_HEADERS)
        if is_glossary and len(section) > 5000:
            # Keep first 3000 chars of glossary (has key definitions like
            # Alocação Mínima, Auditor, etc.)
            kept_sections.append(section[:3000] + "\n[... glossary omitted ...]")
            total_kept += 3030
            continue

        # Always keep first and last
        if is_first or is_last:
            kept_sections.append(section)
            total_kept += len(section)
            continue

        # Keep if header matches target topics
        header_match = any(h in header for h in KEEP_HEADERS)

        # Keep if content contains key data patterns (percentages, CNPJs, fees)
        content_match = any(kw in section_lower for kw in [
            "% ao ano", "por cento) ao ano",
            "valor mínimo", "valor máximo",
            "cnpj", "ato declaratório",
            "regime fechado", "regime aberto",
            "prazo de duração",
            "investidores profissionais", "investidores qualificados",
            "evento de avaliação", "evento de liquidação",
            "foro da comarca", "fica eleito o foro",
            "limite de concentração",
            "alocação mínima", "aplicação mínima",
        ])

        if header_match or content_match:
            kept_sections.append(section)
            total_kept += len(section)
        else:
            # Mark omission
            sec_header = section[:80].replace('\n', ' ').strip()
            kept_sections.append(f"[... section omitted: {sec_header[:60]}... ...]")
            total_kept += 80

    result = "\n".join(kept_sections)

    # If still too large, trim long sections proportionally (keep short ones intact)
    if len(result) > max_chars:
        # Calculate how much needs trimming, only from sections > 2000 chars
        short_total = sum(len(s) for s in kept_sections if len(s) <= 2000 or s.startswith("[..."))
        long_sections_total = sum(len(s) for s in kept_sections if len(s) > 2000 and not s.startswith("[..."))
        target_long = max(0, max_chars - short_total)
        ratio = target_long / max(1, long_sections_total)

        trimmed = []
        for section in kept_sections:
            if section.startswith("[...") or len(section) <= 2000:
                trimmed.append(section)
            else:
                max_sec = max(800, int(len(section) * ratio))
                if len(section) > max_sec:
                    trimmed.append(
                        section[:max_sec]
                        + "\n[... section trimmed ...]"
                    )
                else:
                    trimmed.append(section)
        result = "\n".join(trimmed)

    # Final hard truncate as safety net
    if len(result) > max_chars:
        result = result[:max_chars]

    return result


# ============================================================
# PDF text extraction
# ============================================================


def extract_pdf_text(pdf_path: str, max_chars: int = 50000) -> str:
    """
    Extract text from a PDF and apply smart reduction for long documents.

    For documents exceeding max_chars, selectively keeps only sections
    containing relevant regulatory info (fund ID, fees, risks, etc.),
    dramatically reducing the number of API calls needed.

    Args:
        pdf_path:  Path to the PDF file.
        max_chars: Maximum characters to return (default: 50000).
                   Set to 0 to disable reduction and return full text.
    """
    pages_text = []
    with pdfplumber.open(pdf_path) as pdf:
        num_pages = len(pdf.pages)
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text:
                pages_text.append(f"--- PAGE {i + 1} ---\n{text}")

    raw_text = "\n\n".join(pages_text)
    raw_len = len(raw_text)

    text = _clean_basic(raw_text)
    text = _remove_headers_footers(text, num_pages)
    clean_len = len(text)

    if max_chars > 0 and len(text) > max_chars:
        text = _extract_relevant_sections(text, max_chars)
        pct = ((raw_len - len(text)) / raw_len * 100)
        print(f"   🧹 Text reduced: {raw_len:,} → {clean_len:,} (cleanup) → {len(text):,} chars (relevant sections)")
        print(f"   📉 Total reduction: {pct:.0f}%")
    elif clean_len < raw_len:
        pct = ((raw_len - clean_len) / raw_len * 100)
        if pct > 2:
            print(f"   🧹 Text cleaned: {raw_len:,} → {clean_len:,} chars ({pct:.0f}% reduced)")

    return text


# ============================================================
# Processing estimation
# ============================================================


def estimate_time(num_chars: int, chunk_size: int, passes: int, workers: int) -> str:
    """
    Estimate processing time (conservative, accounts for rate limits).

    Gemini free tier: 15 RPM. With 3 workers, effective throughput is
    ~3 calls every ~15 seconds = ~12 calls/min, safely under the limit.
    """
    num_chunks = max(1, num_chars // chunk_size)
    total_calls = num_chunks * passes
    # With 3 workers and rate limiting: ~20s per batch of 3 calls
    effective = min(workers, 10, num_chunks)
    parallel = (total_calls * 20) / max(1, effective)

    m, s = divmod(int(parallel), 60)
    return f"~{m}min {s}s" if m > 0 else f"~{s}s"


def print_extraction_plan(num_chars: int, chunk_size: int, passes: int, workers: int):
    """Print a summary of the extraction plan before execution."""
    num_chunks = max(1, num_chars // chunk_size)
    total_calls = num_chunks * passes
    effective = min(workers, 10, num_chunks)
    est = estimate_time(num_chars, chunk_size, passes, workers)

    print(f"   📊 Text: {num_chars:,} chars")
    print(f"   📦 Chunks: ~{num_chunks} (buffer: {chunk_size:,} chars)")
    print(f"   🔄 API calls: ~{total_calls} ({passes} pass{'es' if passes > 1 else ''} × {num_chunks} chunks)")
    print(f"   👷 Workers: {effective}")
    print(f"   ⏱️  Estimate: {est}")


def generate_html_report(report: dict, output_path: str):
    """
    Generate a simple HTML visualization of extracted entities.

    Since we merge results from multiple extraction groups, we can't use
    lx.visualize() (which requires a single JSONL file). This generates
    a standalone HTML report from the JSON data instead.
    """
    ensure_output_dir(output_path)

    entities = report.get("entities", {})
    source = report.get("source_file", "unknown")
    total = sum(len(v) for v in entities.values())

    # Color palette for entity types
    colors = [
        "#e3f2fd", "#e8f5e9", "#fff3e0", "#fce4ec", "#f3e5f5",
        "#e0f7fa", "#fff8e1", "#efebe9", "#e8eaf6", "#f1f8e9",
        "#fbe9e7", "#e0f2f1", "#ede7f6", "#fff9c4", "#f9fbe7",
        "#ffebee", "#e1f5fe", "#f0f4c3", "#dcedc8", "#ffe0b2",
        "#b2dfdb", "#c8e6c9",
    ]

    rows = ""
    for i, (cls, vals) in enumerate(sorted(entities.items())):
        color = colors[i % len(colors)]
        values_html = ""
        for v in vals:
            escaped = v.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            escaped = escaped.replace("\n", "<br>")
            values_html += f'<div class="value">{escaped}</div>'
        rows += f"""
        <div class="entity" style="border-left: 4px solid {color}; background: {color}40;">
            <div class="entity-header">
                <span class="entity-name">{cls}</span>
                <span class="entity-count">{len(vals)}</span>
            </div>
            {values_html}
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<title>Extraction Report — {source}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       max-width: 900px; margin: 40px auto; padding: 0 20px; color: #333; }}
h1 {{ color: #1a237e; font-size: 1.4em; }}
.meta {{ color: #666; margin-bottom: 20px; }}
.entity {{ margin: 12px 0; padding: 12px 16px; border-radius: 6px; }}
.entity-header {{ display: flex; justify-content: space-between; align-items: center;
                  margin-bottom: 8px; }}
.entity-name {{ font-weight: 700; font-size: 0.95em; color: #1a237e; }}
.entity-count {{ background: #1a237e; color: white; border-radius: 12px;
                 padding: 2px 10px; font-size: 0.8em; }}
.value {{ padding: 6px 0; border-bottom: 1px solid rgba(0,0,0,0.06);
          font-size: 0.9em; line-height: 1.5; }}
.value:last-child {{ border-bottom: none; }}
</style>
</head>
<body>
<h1>Extraction Report</h1>
<div class="meta">
    <strong>Source:</strong> {source}<br>
    <strong>Entities:</strong> {total} extractions across {len(entities)} types
</div>
{rows}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
