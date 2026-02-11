#!/usr/bin/env python3
"""
extrair_demonstrativo.py — Structured extraction from FIDC quarterly reports.

BACKGROUND FOR INTERNATIONAL USERS:
    FIDC (Fundo de Investimento em Direitos Creditórios) are Brazilian
    securitization vehicles — similar to CLOs or ABS trusts. They buy
    credit rights (receivables) from originators.

    CVM Resolution 175 (Article 27, Annex II) requires FIDCs to publish
    quarterly reports ("Demonstrativos Trimestrais") disclosing:
    - Whether credit rights have proper backing (lastro verification)
    - Ongoing lawsuits involving the fund
    - Changes to investment policy
    - Originator concentration (>10% of portfolio)
    - Events affecting financial flows (defaults, bankruptcies)

    These reports are filed via Fundos.NET and are public documents.

RED FLAG DETECTION:
    This script automatically scans extracted entities for investigative
    journalism red flags: lawsuits, backing inconsistencies, defaults,
    and concentration risks.

USAGE:
    python extrair_demonstrativo.py <pdf_path> [options]
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
    print_extraction_plan, ensure_output_dir, safe_save_documents,
    generate_html_report,
)

load_config()

try:
    import langextract as lx
except ImportError:
    print("Error: LangExtract not installed. Run: pip install langextract")
    sys.exit(1)


# ============================================================
# Prompt and few-shot example
# ============================================================

PROMPT_QUARTERLY = textwrap.dedent("""\
    Extract the following from a Brazilian FIDC quarterly report
    (Demonstrativo Trimestral), per Article 27 of CVM Resolution 175 Annex II:

    - nome_fundo: Full fund/class name
    - cnpj_fundo: Fund CNPJ (tax ID)
    - periodo_referencia: Reference quarter and year (e.g., 2nd quarter 2025)
    - data_documento: Document date
    - gestor: Name of the manager responsible for the quarterly report
    - verificacao_lastro: Result of credit rights backing verification
    - registro_direitos: Result on registration, origin, existence and enforceability
    - processos_judiciais: Information about lawsuits involving the fund
    - precatorios: Information about federal court-ordered debts (precatórios)
    - alteracao_politica: Whether investment policy changed and its effects
    - originador_relevante: Originator with 10%+ of portfolio and credit criteria
    - alteracao_garantias: Whether collateral/guarantees were changed
    - cessao_direitos: How credit rights were assigned, contract details
    - pre_pagamento: Impact of prepayment events
    - alienacao_direitos: Conditions of credit rights disposal
    - descontinuidade_originacao: Impact of potential origination discontinuity
    - fatos_fluxos_financeiros: Events that affected financial flow regularity
    - inconsistencia: Any inconsistency, caveat, or relevant observation found

    Copy the exact text span for each extraction. Extract ALL occurrences.
    Capture both affirmative and negative statements (e.g., "there were no lawsuits").
""")

EXAMPLE_QUARTERLY = lx.data.ExampleData(
    text=textwrap.dedent("""\
        DEMONSTRATIVO TRIMESTRAL DE FIDC
        1º TRIMESTRE 2025

        Comissão de Valores Mobiliários – CVM

        Em atendimento ao disposto no art. 27 do Anexo Normativo II da Resolução CVM 175
        apresentamos as informações referentes ao Demonstrativo Trimestral do
        FUNDO DELTA FIDC, CNPJ 99.888.777/0001-66 relativo ao 1º TRIMESTRE 2025:

        1. Resultado da verificação de lastro dos direitos creditórios realizados
        pelo custodiante, nos termos do art. 38 do Anexo Normativo II da Resolução 175.

        Os procedimentos de verificação foram concluídos satisfatoriamente e não foram
        identificadas inconsistências significativas no trimestre em referência.

        2. Resultado do registro dos direitos creditórios no que se refere à origem,
        existência e exigibilidade desses ativos.

        Foram observados os requisitos de registro dos direitos creditórios em
        atendimento à norma vigente.

        3. Existência de processos judiciais envolvendo o FIDC.

        Existe 1 (uma) ação judicial em curso referente a cobrança de direitos
        creditórios inadimplidos no valor de R$ 2.500.000,00.

        4. Informações referentes a aquisição de precatórios federais.

        O Fundo não adquire precatórios federais.

        DEMONSTRATIVO TRIMESTRAL FIDC
        GESTORA SIGMA | 33.444.555/0001-22 | 1º trimestre de 2025

        Os efeitos de eventual alteração na política de investimento sobre a
        rentabilidade da carteira de ativos

        Houve alteração na política de investimento: o limite de concentração por
        devedor foi reduzido de 20% para 15% do patrimônio líquido.

        Há algum originador que represente individualmente 10% ou mais da carteira?

        Sim. A empresa Originadora ABC Ltda. representou 35% da carteira no trimestre.

        Eventuais alterações nas garantias existentes

        Não houve alterações de garantias.

        Forma como se operou a cessão dos direitos creditórios

        Foram firmados 3 contratos de cessão no valor total de R$ 15.000.000,00.
        As cessões tiveram caráter definitivo.

        Fatos ocorridos que afetaram a regularidade dos fluxos financeiros

        Durante o período de referência, 2 devedores entraram em recuperação judicial,
        afetando R$ 4.800.000,00 em direitos creditórios.
    """),
    extractions=[
        lx.data.Extraction(extraction_class="nome_fundo", extraction_text="FUNDO DELTA FIDC"),
        lx.data.Extraction(extraction_class="cnpj_fundo", extraction_text="99.888.777/0001-66"),
        lx.data.Extraction(extraction_class="periodo_referencia", extraction_text="1º TRIMESTRE 2025"),
        lx.data.Extraction(extraction_class="gestor", extraction_text="GESTORA SIGMA | 33.444.555/0001-22"),
        lx.data.Extraction(extraction_class="verificacao_lastro", extraction_text="Os procedimentos de verificação foram concluídos satisfatoriamente e não foram identificadas inconsistências significativas no trimestre em referência."),
        lx.data.Extraction(extraction_class="registro_direitos", extraction_text="Foram observados os requisitos de registro dos direitos creditórios em atendimento à norma vigente."),
        lx.data.Extraction(extraction_class="processos_judiciais", extraction_text="Existe 1 (uma) ação judicial em curso referente a cobrança de direitos creditórios inadimplidos no valor de R$ 2.500.000,00."),
        lx.data.Extraction(extraction_class="precatorios", extraction_text="O Fundo não adquire precatórios federais."),
        lx.data.Extraction(extraction_class="alteracao_politica", extraction_text="Houve alteração na política de investimento: o limite de concentração por devedor foi reduzido de 20% para 15% do patrimônio líquido."),
        lx.data.Extraction(extraction_class="originador_relevante", extraction_text="Sim. A empresa Originadora ABC Ltda. representou 35% da carteira no trimestre."),
        lx.data.Extraction(extraction_class="alteracao_garantias", extraction_text="Não houve alterações de garantias."),
        lx.data.Extraction(extraction_class="cessao_direitos", extraction_text="Foram firmados 3 contratos de cessão no valor total de R$ 15.000.000,00. As cessões tiveram caráter definitivo."),
        lx.data.Extraction(extraction_class="fatos_fluxos_financeiros", extraction_text="Durante o período de referência, 2 devedores entraram em recuperação judicial, afetando R$ 4.800.000,00 em direitos creditórios."),
    ],
)


# ============================================================
# Red flag detection for investigative journalism
# ============================================================


def classify_alerts(report: dict) -> list:
    """
    Scan extracted entities for red flags.

    Severity levels:
    - CRITICAL: Backing inconsistencies (fund may lack proper collateral)
    - HIGH: Active lawsuits, financial flow disruptions (defaults, bankruptcies)
    - MEDIUM: Policy changes, originator concentration (>10% of portfolio)
    """
    alerts = []
    ent = report.get("entities", {})

    # Check for active lawsuits
    for text in ent.get("processos_judiciais", []):
        tl = text.lower()
        has_lawsuit = any(t in tl for t in ["ação judicial", "processo", "execução", "cobrança"])
        is_negative = any(t in tl for t in ["não tem conhecimento", "não há", "inexistente"])
        if has_lawsuit and not is_negative:
            alerts.append({"type": "LAWSUIT", "severity": "HIGH", "detail": text[:200]})

    # Check for policy changes
    for text in ent.get("alteracao_politica", []):
        if "houve alteração" in text.lower() or "foi alterad" in text.lower():
            alerts.append({"type": "POLICY_CHANGE", "severity": "MEDIUM", "detail": text[:200]})

    # Check for financial flow disruptions
    for text in ent.get("fatos_fluxos_financeiros", []):
        tl = text.lower()
        has_issue = any(t in tl for t in ["recuperação judicial", "inadimplência", "default", "atraso"])
        if has_issue and "não houve" not in tl:
            alerts.append({"type": "FINANCIAL_DISRUPTION", "severity": "HIGH", "detail": text[:200]})

    # Check for originator concentration
    for text in ent.get("originador_relevante", []):
        if "sim" in text.lower() or "%" in text:
            alerts.append({"type": "ORIGINATOR_CONCENTRATION", "severity": "MEDIUM", "detail": text[:200]})

    # Check for backing inconsistencies (most critical)
    for text in ent.get("verificacao_lastro", []):
        if "inconsistência" in text.lower() and "não foram identificadas" not in text.lower():
            alerts.append({"type": "BACKING_INCONSISTENCY", "severity": "CRITICAL", "detail": text[:200]})

    return alerts


def build_report(result, pdf_path: str) -> dict:
    """Build consolidated JSON report with automatic red flag alerts."""
    report = {
        "source_file": os.path.basename(pdf_path),
        "full_path": os.path.abspath(pdf_path),
        "document_type": "quarterly_report_fidc",
        "entities": {},
    }
    if hasattr(result, "extractions") and result.extractions:
        for ext in result.extractions:
            report["entities"].setdefault(ext.extraction_class, []).append(ext.extraction_text)
    report["alerts"] = classify_alerts(report)
    return report


# ============================================================
# Main extraction
# ============================================================


def extract_quarterly(
    pdf_path, model="gemini-2.0-flash", passes=1, workers=3,
    output_dir="output", chunk_size=3000, max_chars=50000,
):
    """Extract structured data from a FIDC quarterly report PDF."""
    if not os.path.exists(pdf_path):
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)

    os.makedirs(os.path.abspath(output_dir), exist_ok=True)
    base_name = Path(pdf_path).stem

    print(f"📄 Processing: {pdf_path}")
    print(f"🤖 Model: {model}")

    print("\n1️⃣  Extracting text from PDF...")
    text = extract_pdf_text(pdf_path, max_chars=max_chars)
    print(f"   ✅ Text for processing: {len(text):,} characters")

    if len(text) < 50:
        print("   ⚠️  Text too short. PDF may be scanned.")
        sys.exit(1)

    config = configure_model(model)

    print(f"\n2️⃣  Extraction plan:")
    print_extraction_plan(len(text), chunk_size, passes, workers)

    print(f"\n3️⃣  Running extraction...\n")

    result = None
    attempts = [
        {"chunk": chunk_size, "label": "default"},
        {"chunk": max(1000, chunk_size // 2), "label": "reduced chunk"},
        {"chunk": 1000, "label": "minimal chunk"},
    ]

    for attempt in attempts:
        try:
            result = lx.extract(
                text_or_documents=text,
                prompt_description=PROMPT_QUARTERLY,
                examples=[EXAMPLE_QUARTERLY],
                extraction_passes=passes,
                max_workers=workers,
                max_char_buffer=attempt["chunk"],
                **config,
            )
            break
        except Exception as e:
            err_str = str(e)
            is_json_error = any(t in err_str for t in [
                "Failed to parse JSON", "JSONDecodeError", "Expecting",
                "Unterminated string", "Invalid control character",
            ])
            if is_json_error and attempt != attempts[-1]:
                next_chunk = attempts[attempts.index(attempt) + 1]["chunk"]
                print(f"   ⚠️  JSON parse error with chunk={attempt['chunk']}. "
                      f"Retrying with chunk={next_chunk}...\n")
                continue
            else:
                print(f"\n   ❌ Extraction error: {e}")
                sys.exit(1)

    if result is None:
        print("   ❌ All extraction attempts failed.")
        sys.exit(1)

    # Save outputs — safe_save_documents handles directory creation
    print(f"\n4️⃣  Saving results to {output_dir}/")

    jsonl_path = os.path.join(output_dir, f"{base_name}_extractions.jsonl")
    safe_save_documents([result], jsonl_path)

    html_path = os.path.join(output_dir, f"{base_name}_visualization.html")
    ensure_output_dir(html_path)
    try:
        abs_jsonl = os.path.abspath(jsonl_path)
        html = lx.visualize(abs_jsonl)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)
    except Exception:
        pass

    report = build_report(result, pdf_path)
    report_path = os.path.join(output_dir, f"{base_name}_report.json")
    ensure_output_dir(report_path)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # Summary
    total = sum(len(v) for v in report["entities"].values())
    print(f"\n{'='*60}")
    print(f"✅ EXTRACTION COMPLETE — {total} entities found")
    print(f"{'='*60}")

    for cls, values in sorted(report["entities"].items()):
        print(f"   • {cls}: {len(values)}")
        for v in values[:1]:
            print(f"     → {v.replace(chr(10), ' ')[:100]}{'...' if len(v) > 100 else ''}")

    alerts = report.get("alerts", [])
    if alerts:
        print(f"\n🚨 ALERTS ({len(alerts)}):")
        for a in alerts:
            icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡"}.get(a["severity"], "⚪")
            print(f"   {icon} [{a['severity']}] {a['type']}: {a['detail'][:100]}")
    else:
        print(f"\n✅ No alerts (regular operation)")

    print(f"\n📁 Output: {output_dir}/")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract data from FIDC quarterly reports with red flag detection",
    )
    parser.add_argument("pdf", help="Path to quarterly report PDF")
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", default="output")
    parser.add_argument("--chunk-size", type=int, default=3000)
    parser.add_argument("--max-chars", type=int, default=50000)
    parser.add_argument("--no-filter", action="store_true")
    parser.add_argument("--fast", action="store_true")

    args = parser.parse_args()
    if args.fast:
        args.passes, args.chunk_size, args.max_chars = 1, 3000, 15000
    if args.no_filter:
        args.max_chars = 0

    extract_quarterly(
        args.pdf, args.model, args.passes, args.workers,
        args.output, args.chunk_size, args.max_chars,
    )
