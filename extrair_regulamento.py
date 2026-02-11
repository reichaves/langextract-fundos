#!/usr/bin/env python3
"""
extrair_regulamento.py — Structured extraction from Brazilian fund regulation PDFs.

Uses Google's LangExtract library to extract key entities from investment fund
regulation documents filed with CVM (Comissão de Valores Mobiliários — the
Brazilian SEC equivalent), with source text grounding.

BACKGROUND FOR INTERNATIONAL USERS:
    CVM is Brazil's securities regulator. Investment funds must file regulation
    documents ("regulamentos") via the Fundos.NET system operated by B3 (the
    Brazilian stock exchange). These PDFs follow a standard structure defined
    by CVM Resolution 175/2022.

    A "regulamento" is similar to a fund's prospectus or offering memorandum.
    It defines the fund's rules: who manages it, what it invests in, fees
    charged, investor eligibility, and risk factors.

EXTRACTED ENTITIES:
    - Fund identification (name, CNPJ tax ID, fund type)
    - Service providers (administrator, manager, custodian, auditor)
    - Fees (administration, management, performance, custody)
    - Structure (duration, open/closed-end, target investors, share classes)
    - Investment policy (eligible assets, concentration limits)
    - Risk factors, evaluation events, liquidation events

USAGE:
    python extrair_regulamento.py <pdf_path> [options]

EXAMPLES:
    python extrair_regulamento.py regulation.pdf
    python extrair_regulamento.py regulation.pdf --fast
    python extrair_regulamento.py regulation.pdf --no-filter --passes 2
    python extrair_regulamento.py regulation.pdf --model gpt-4o
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    load_config, extract_pdf_text, configure_model,
    ensure_output_dir, generate_html_report,
)

load_config()

try:
    import langextract as lx
except ImportError:
    print("Error: LangExtract not installed. Run: pip install langextract")
    sys.exit(1)


# ============================================================
# Prompts and examples — SPLIT INTO 2 GROUPS
# ============================================================
#
# WHY: With 22 entity types, the model generates ~22K+ chars of JSON per chunk,
# which gets truncated (Unterminated string error). Splitting into 2 groups
# of ~11 entities each cuts JSON output in half, avoiding truncation.

# --- GROUP A: Identification + Service Providers + Fees ---

PROMPT_GROUP_A = textwrap.dedent("""\
    Extract ONLY the following from a Brazilian fund regulation (CVM Res. 175/2022):

    - nome_fundo: Full fund name
    - cnpj_fundo: Fund CNPJ (XX.XXX.XXX/XXXX-XX)
    - tipo_fundo: Fund type (FIDC, FIAGRO, FII, FIP, etc.)
    - administrador: Name and CNPJ of fiduciary administrator
    - gestor: Name and CNPJ of asset manager
    - custodiante: Custodian name (and CNPJ if available)
    - auditor: Independent auditor name

    Copy the exact text span. Extract ALL occurrences.
""")

EXAMPLE_GROUP_A = lx.data.ExampleData(
    text=textwrap.dedent("""\
        REGULAMENTO DO
        FUNDO ALPHA FUNDO DE INVESTIMENTO EM DIREITOS CREDITÓRIOS
        CNPJ Nº 12.345.678/0001-99

        2.1. ADMINISTRADOR
        BANCO BETA S.A.
        CNPJ: 11.222.333/0001-44

        2.2. GESTOR
        GAMMA GESTÃO DE RECURSOS LTDA.
        CNPJ: 55.666.777/0001-88
    """),
    extractions=[
        lx.data.Extraction(extraction_class="nome_fundo",
            extraction_text="FUNDO ALPHA FUNDO DE INVESTIMENTO EM DIREITOS CREDITÓRIOS"),
        lx.data.Extraction(extraction_class="cnpj_fundo",
            extraction_text="12.345.678/0001-99"),
        lx.data.Extraction(extraction_class="tipo_fundo",
            extraction_text="FUNDO DE INVESTIMENTO EM DIREITOS CREDITÓRIOS"),
        lx.data.Extraction(extraction_class="administrador",
            extraction_text="BANCO BETA S.A.\nCNPJ: 11.222.333/0001-44"),
        lx.data.Extraction(extraction_class="gestor",
            extraction_text="GAMMA GESTÃO DE RECURSOS LTDA.\nCNPJ: 55.666.777/0001-88"),
    ],
)

# --- GROUP B: Fees + Structure + Legal (concise entities) ---

PROMPT_GROUP_B = textwrap.dedent("""\
    Extract ONLY the following from a Brazilian fund regulation (CVM Res. 175/2022):

    - taxa_administracao: Administration fee (% per year AND minimum amount)
    - taxa_gestao: Management fee (% per year AND minimum amount)
    - taxa_performance: Performance fee (if applicable)
    - taxa_custodia: Custody fee (if applicable)
    - prazo_duracao: Fund duration (e.g., "indeterminado" or "5 anos")
    - regime_condominial: Open or closed-end (aberto/fechado)
    - publico_alvo: Target investors (retail/qualified/professional)
    - foro: Legal forum for disputes

    Copy the exact text span. Keep extractions concise.
""")

EXAMPLE_GROUP_B = lx.data.ExampleData(
    text=textwrap.dedent("""\
        3.1. O Fundo terá prazo de duração indeterminado.
        2.3. REGIME CONDOMINIAL: Fechado
        4.1. A Classe é destinada a investidores profissionais.
        7.1. Taxa de Administração equivalente a 0,20% ao ano.
        Valor mínimo: R$10.000,00 mensais.
        7.2. Taxa de Gestão equivalente a 1,00% ao ano.
        Valor mínimo: R$25.000,00 mensais.
        30.1. Fica eleito o foro da cidade de São Paulo.
    """),
    extractions=[
        lx.data.Extraction(extraction_class="taxa_administracao",
            extraction_text="0,20% ao ano.\nValor mínimo: R$10.000,00 mensais."),
        lx.data.Extraction(extraction_class="taxa_gestao",
            extraction_text="1,00% ao ano.\nValor mínimo: R$25.000,00 mensais."),
        lx.data.Extraction(extraction_class="prazo_duracao",
            extraction_text="prazo de duração indeterminado"),
        lx.data.Extraction(extraction_class="regime_condominial",
            extraction_text="Fechado"),
        lx.data.Extraction(extraction_class="publico_alvo",
            extraction_text="investidores profissionais"),
        lx.data.Extraction(extraction_class="foro",
            extraction_text="foro da cidade de São Paulo"),
    ],
)

# --- GROUP C: Policy + Risk + Events (may have long text → instruct brevity) ---

PROMPT_GROUP_C = textwrap.dedent("""\
    Extract ONLY the following from a Brazilian fund regulation (CVM Res. 175/2022).
    IMPORTANT: Keep each extraction SHORT (max 200 chars). Extract only the key value,
    NOT the full enumeration of items.

    - classe_cotas: Share classes/subclasses (e.g., "Sênior, Mezanino, Subordinada")
    - ativo_alvo: Main asset type (e.g., "direitos creditórios", "duplicatas")
    - aplicacao_minima: Minimum allocation % in target assets
    - limite_concentracao: Main concentration limit per debtor (% of PL)
    - fator_risco: List only the NAMES of risk factors, not descriptions
    - evento_avaliacao: List only the clause number and trigger summary
    - evento_liquidacao: List only the clause number and trigger summary
    - valor_emissao: Total issuance value (if mentioned)

    Copy exact text for short items. For long lists, extract only the key clause reference.
""")

EXAMPLE_GROUP_C = lx.data.ExampleData(
    text=textwrap.dedent("""\
        2.2. Classe Única, com subclasses Sênior, Mezanino e Subordinada.
        10.1. Direitos creditórios representados por duplicatas e CPR-F.
        Alocação Mínima de 50% do Patrimônio Líquido.
        10.5. Limite de 20% do Patrimônio Líquido por devedor.
        15. FATORES DE RISCO: risco de crédito, risco de mercado, risco de liquidez.
        26.1. EVENTOS DE AVALIAÇÃO: (a) PL negativo; (b) inadimplência acima de 10%.
        26.2. EVENTOS DE LIQUIDAÇÃO: (a) não recomposição; (b) desenquadramento.
    """),
    extractions=[
        lx.data.Extraction(extraction_class="classe_cotas",
            extraction_text="Classe Única, com subclasses Sênior, Mezanino e Subordinada"),
        lx.data.Extraction(extraction_class="ativo_alvo",
            extraction_text="Direitos creditórios representados por duplicatas e CPR-F"),
        lx.data.Extraction(extraction_class="aplicacao_minima",
            extraction_text="50% do Patrimônio Líquido"),
        lx.data.Extraction(extraction_class="limite_concentracao",
            extraction_text="20% do Patrimônio Líquido por devedor"),
        lx.data.Extraction(extraction_class="fator_risco",
            extraction_text="risco de crédito, risco de mercado, risco de liquidez"),
        lx.data.Extraction(extraction_class="evento_avaliacao",
            extraction_text="26.1. (a) PL negativo; (b) inadimplência acima de 10%"),
        lx.data.Extraction(extraction_class="evento_liquidacao",
            extraction_text="26.2. (a) não recomposição; (b) desenquadramento"),
    ],
)


# ============================================================
# Report generation
# ============================================================


def build_report(result, pdf_path: str) -> dict:
    """Build a consolidated JSON report from LangExtract results."""
    report = {
        "source_file": os.path.basename(pdf_path),
        "full_path": os.path.abspath(pdf_path),
        "entities": {},
    }
    if hasattr(result, "extractions") and result.extractions:
        for ext in result.extractions:
            report["entities"].setdefault(ext.extraction_class, []).append(
                ext.extraction_text
            )
    return report


# ============================================================
# Main extraction function
# ============================================================


def extract_regulation(
    pdf_path: str,
    model: str = "gemini-2.0-flash",
    passes: int = 1,
    workers: int = 3,
    output_dir: str = "output",
    chunk_size: int = 3000,
    max_chars: int = 50000,
):
    """
    Extract structured entities from a fund regulation PDF.

    Default parameters are tuned for Gemini free-tier (15 RPM rate limit).
    For paid API tiers, increase workers to 10-15.

    Args:
        pdf_path:   Path to the regulation PDF.
        model:      LLM model identifier.
        passes:     Extraction passes (1=fast, 2=higher recall).
        workers:    Parallel workers. Default 3 avoids free-tier rate limits.
        output_dir: Directory for output files (created automatically).
        chunk_size: Characters per chunk (3000-5000 recommended).
        max_chars:  Max text to process after smart filtering.
                    Default 30K ≈ 6 chunks ≈ 6 API calls ≈ ~2 min.
                    Set to 0 to process full text (slow, 30+ min for large docs).
    """
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    # Create output directory tree immediately (defense-in-depth)
    os.makedirs(os.path.abspath(output_dir), exist_ok=True)

    base_name = Path(pdf_path).stem

    print(f"📄 Processing: {pdf_path}")
    print(f"🤖 Model: {model}")

    # 1. Extract and reduce text
    print("\n1️⃣  Extracting text from PDF...")
    text = extract_pdf_text(pdf_path, max_chars=max_chars)
    print(f"   ✅ Text for processing: {len(text):,} characters")

    if len(text) < 100:
        print("   ⚠️  Text too short. PDF may be scanned (image-based).")
        print("   Consider running OCR first (e.g., pytesseract or ocrmypdf).")
        sys.exit(1)

    # 2. Configure model
    config = configure_model(model)

    # 3. Show extraction plan (3 groups = 3× the API calls)
    print(f"\n2️⃣  Extraction plan (3 entity groups):")
    num_chunks = max(1, len(text) // chunk_size)
    total_calls = num_chunks * passes * 3  # ×3 for 3 groups
    effective = min(workers, 10, num_chunks)
    secs = (total_calls * 20) / max(1, effective)
    m, s = divmod(int(secs), 60)
    est = f"~{m}min {s}s" if m > 0 else f"~{s}s"
    print(f"   📊 Text: {len(text):,} chars")
    print(f"   📦 Chunks: ~{num_chunks} per group (buffer: {chunk_size:,} chars)")
    print(f"   🔄 API calls: ~{total_calls} (3 groups × {num_chunks} chunks × {passes} pass{'es' if passes > 1 else ''})")
    print(f"   👷 Workers: {effective}")
    print(f"   ⏱️  Estimate: {est}")

    # 4. Run extraction in 2 groups (smaller JSON = no truncation)
    print(f"\n3️⃣  Running extraction (2 groups)...\n")

    groups = [
        {"name": "A (ID + Providers)", "prompt": PROMPT_GROUP_A, "example": EXAMPLE_GROUP_A},
        {"name": "B (Fees + Structure)", "prompt": PROMPT_GROUP_B, "example": EXAMPLE_GROUP_B},
        {"name": "C (Policy + Risk + Events)", "prompt": PROMPT_GROUP_C, "example": EXAMPLE_GROUP_C},
    ]

    all_extractions = []

    for group in groups:
        print(f"   📋 Group {group['name']}...")
        group_result = None

        for attempt_chunk in [chunk_size, max(1000, chunk_size // 2), 1000]:
            try:
                group_result = lx.extract(
                    text_or_documents=text,
                    prompt_description=group["prompt"],
                    examples=[group["example"]],
                    extraction_passes=passes,
                    max_workers=workers,
                    max_char_buffer=attempt_chunk,
                    **config,
                )
                break
            except Exception as e:
                err_str = str(e)
                is_json_error = any(t in err_str for t in [
                    "Failed to parse JSON", "JSONDecodeError", "Expecting",
                    "Unterminated string", "Invalid control character",
                ])
                if is_json_error and attempt_chunk > 1000:
                    print(f"      ⚠️  JSON error (chunk={attempt_chunk}), retrying smaller...")
                    continue
                else:
                    print(f"\n   ❌ Extraction error in group {group['name']}: {e}")
                    print("\n   Check: API key set? Model exists? Internet connected?")
                    sys.exit(1)

        if group_result is None:
            print(f"   ⚠️  Group {group['name']} failed all attempts, skipping.")
            continue

        # Collect extractions from this group
        if hasattr(group_result, "extractions") and group_result.extractions:
            count = len(group_result.extractions)
            all_extractions.extend(group_result.extractions)
            print(f"      ✅ {count} entities found")
        else:
            print(f"      ⚠️  No entities found")

    if not all_extractions:
        print("\n   ❌ No entities extracted from any group.")
        sys.exit(1)

    # 5. Save outputs
    print(f"\n4️⃣  Saving results to {output_dir}/")

    # Build report from merged extractions
    report = {
        "source_file": os.path.basename(pdf_path),
        "full_path": os.path.abspath(pdf_path),
        "entities": {},
    }
    for ext in all_extractions:
        report["entities"].setdefault(ext.extraction_class, []).append(
            ext.extraction_text
        )

    # Save JSON report (the primary output — does not depend on LangExtract I/O)
    report_path = os.path.join(output_dir, f"{base_name}_report.json")
    ensure_output_dir(report_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Generate HTML visualization
    html_path = os.path.join(output_dir, f"{base_name}_report.html")
    try:
        generate_html_report(report, html_path)
    except Exception as e:
        print(f"   ⚠️  HTML generation: {e}")

    # 6. Print summary
    total = sum(len(v) for v in report["entities"].values())
    print(f"\n{'='*60}")
    print(f"✅ EXTRACTION COMPLETE — {total} entities found")
    print(f"{'='*60}")
    for cls, vals in sorted(report["entities"].items()):
        print(f"   • {cls}: {len(vals)}")
        for v in vals[:2]:
            preview = v.replace("\n", " ")[:80]
            print(f"     → {preview}{'...' if len(v) > 80 else ''}")
    print(f"\n📁 Output files:")
    print(f"   📄 {report_path}")
    print(f"   🌐 {html_path}")
    return report


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract structured data from Brazilian fund regulation PDFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python extrair_regulamento.py regulation.pdf
              python extrair_regulamento.py regulation.pdf --fast
              python extrair_regulamento.py regulation.pdf --no-filter --passes 2
              python extrair_regulamento.py regulation.pdf --model gpt-4o
              python extrair_regulamento.py regulation.pdf --model ollama:gemma2:9b

            Performance tips:
              Default settings are tuned for Gemini free tier (15 RPM).
              - Default: ~6 API calls, ~2 min for a 230K-char document.
              - --fast:  ~4 API calls, ~1 min.
              - --no-filter: processes full text, ~45 API calls, 30+ min.
              For paid API: use --workers 10 --max-chars 80000 for more recall.
        """),
    )
    parser.add_argument("pdf", help="Path to the regulation PDF")
    parser.add_argument("--model", default="gemini-2.0-flash",
                        help="LLM model (default: gemini-2.0-flash)")
    parser.add_argument("--passes", type=int, default=1,
                        help="Extraction passes (default: 1)")
    parser.add_argument("--workers", type=int, default=3,
                        help="Parallel workers (default: 3, safe for free tier)")
    parser.add_argument("--output", default="output",
                        help="Output directory (default: output/)")
    parser.add_argument("--chunk-size", type=int, default=3000,
                        help="Chunk size in chars (default: 3000)")
    parser.add_argument("--max-chars", type=int, default=50000,
                        help="Max text to process (default: 50000, 0 = no limit)")
    parser.add_argument("--no-filter", action="store_true",
                        help="Process full text without section filtering (slow)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: 1 pass, chunk 3K, max 15K chars")

    args = parser.parse_args()

    if args.fast:
        args.passes, args.chunk_size, args.max_chars = 1, 3000, 15000
        print("⚡ Fast mode: 1 pass, chunk 3K, max 15K chars\n")
    if args.no_filter:
        args.max_chars = 0
        print("📄 No filter: processing full text (may be slow)\n")

    extract_regulation(
        args.pdf, args.model, args.passes, args.workers,
        args.output, args.chunk_size, args.max_chars,
    )
