#!/usr/bin/env python3
"""
extrair_lote.py — Batch processing of Brazilian fund documents.

Processes multiple PDFs in a folder, auto-detecting document type
(regulation vs quarterly report) and generating a comparative report.

USAGE:
    python extrair_lote.py <pdf_folder_or_files> [options]

EXAMPLES:
    python extrair_lote.py ./documents/
    python extrair_lote.py ./documents/ --compare
    python extrair_lote.py doc1.pdf doc2.pdf --compare
    python extrair_lote.py ./documents/ --fast --compare
"""

import argparse
import csv
import glob
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_config, extract_pdf_text, configure_model, ensure_output_dir, generate_html_report

load_config()

from extrair_regulamento import (
    PROMPT_GROUP_A, EXAMPLE_GROUP_A,
    PROMPT_GROUP_B, EXAMPLE_GROUP_B,
    PROMPT_GROUP_C, EXAMPLE_GROUP_C,
)
from extrair_demonstrativo import (
    PROMPT_QUARTERLY, EXAMPLE_QUARTERLY,
    classify_alerts,
)

try:
    import langextract as lx
except ImportError:
    print("Error: LangExtract not installed. Run: pip install langextract")
    sys.exit(1)


# ============================================================
# Document type auto-detection
# ============================================================

# Keywords that appear more in regulations vs quarterly reports.
REGULATION_KEYWORDS = [
    "regulamento", "capítulo", "glossário", "taxa de administração",
    "público-alvo", "fatores de risco", "assembleia geral",
    "política de investimento", "limites de concentração",
]
QUARTERLY_KEYWORDS = [
    "demonstrativo trimestral", "art. 27", "anexo normativo ii",
    "verificação de lastro", "precatórios", "originador",
    "cessão", "pré-pagamento",
]


def detect_document_type(text: str) -> str:
    """Detect document type from content (first 10K chars)."""
    sample = text[:10000].lower()
    reg_score = sum(1 for kw in REGULATION_KEYWORDS if kw in sample)
    qtr_score = sum(1 for kw in QUARTERLY_KEYWORDS if kw in sample)
    if reg_score > qtr_score:
        return "regulation"
    elif qtr_score > reg_score:
        return "quarterly"
    return "unknown"


def detect_type_from_filename(name: str) -> str:
    """Detect document type from filename conventions used by Fundos.NET."""
    name = name.upper()
    if "REG" in name:
        return "regulation"
    elif "IFP" in name or "DEM" in name or "TRIM" in name:
        return "quarterly"
    return "unknown"


# ============================================================
# Single document processing
# ============================================================


def process_document(
    pdf_path, forced_type=None, model="gemini-2.0-flash",
    passes=1, workers=3, output_dir="output", chunk_size=3000, max_chars=50000,
):
    """Process a single document and return metadata dict."""
    base_name = Path(pdf_path).stem
    meta = {
        "file": os.path.basename(pdf_path),
        "path": os.path.abspath(pdf_path),
        "start": datetime.now().isoformat(),
        "status": "pending", "type": None, "entity_count": 0,
        "alerts": [], "error": None,
    }

    try:
        text = extract_pdf_text(pdf_path, max_chars=max_chars)
        if len(text) < 50:
            meta["status"] = "error"
            meta["error"] = "Text too short (scanned PDF?)"
            return meta

        # Detect document type
        if forced_type and forced_type != "auto":
            doc_type = forced_type
        else:
            doc_type = detect_type_from_filename(os.path.basename(pdf_path))
            if doc_type == "unknown":
                doc_type = detect_document_type(text)

        meta["type"] = doc_type
        if doc_type == "unknown":
            doc_type = "regulation"
            meta["type"] = "regulation (inferred)"

        # Select prompts based on document type
        if doc_type.startswith("regulation"):
            # Regulations use 3 groups to avoid JSON truncation (22 entity types)
            extraction_groups = [
                {"prompt": PROMPT_GROUP_A, "example": EXAMPLE_GROUP_A},
                {"prompt": PROMPT_GROUP_B, "example": EXAMPLE_GROUP_B},
                {"prompt": PROMPT_GROUP_C, "example": EXAMPLE_GROUP_C},
            ]
        else:
            # Quarterly reports use single group (fewer entity types)
            extraction_groups = [
                {"prompt": PROMPT_QUARTERLY, "example": EXAMPLE_QUARTERLY},
            ]

        config = configure_model(model)
        all_extractions = []

        for group in extraction_groups:
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
                        continue
                    raise

            if group_result and hasattr(group_result, "extractions") and group_result.extractions:
                all_extractions.extend(group_result.extractions)

        if not all_extractions:
            raise RuntimeError("No entities extracted from any group")

        # Build report from merged extractions
        sub_dir = os.path.join(output_dir, base_name)

        report = {
            "source_file": os.path.basename(pdf_path),
            "full_path": os.path.abspath(pdf_path),
            "document_type": doc_type,
            "entities": {},
        }
        for ext in all_extractions:
            report["entities"].setdefault(ext.extraction_class, []).append(ext.extraction_text)

        # Add alerts for quarterly reports
        if not doc_type.startswith("regulation"):
            report["alerts"] = classify_alerts(report)

        report_path = os.path.join(sub_dir, f"{base_name}_report.json")
        ensure_output_dir(report_path)
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        # Generate HTML visualization
        html_path = os.path.join(sub_dir, f"{base_name}_report.html")
        try:
            generate_html_report(report, html_path)
        except Exception:
            pass

        meta["status"] = "success"
        meta["entity_count"] = sum(len(v) for v in report.get("entities", {}).values())
        meta["alerts"] = report.get("alerts", [])
        meta["output_dir"] = sub_dir

    except Exception as e:
        meta["status"] = "error"
        meta["error"] = str(e)

    meta["end"] = datetime.now().isoformat()
    return meta


# ============================================================
# Comparative report
# ============================================================


def generate_comparative_report(results, output_dir):
    """Generate a comparative report across multiple documents."""
    fund_data = []
    for r in results:
        if r["status"] != "success":
            continue
        base_name = Path(r["file"]).stem
        report_path = os.path.join(r.get("output_dir", ""), f"{base_name}_report.json")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                fund_data.append(json.load(f))

    if not fund_data:
        print("No data available for comparison.")
        return

    comparative = {
        "generated_at": datetime.now().isoformat(),
        "document_count": len(fund_data),
        "funds": [], "consolidated_alerts": [], "service_providers": {},
    }

    for data in fund_data:
        ent = data.get("entities", {})
        info = {
            "file": data.get("source_file", ""),
            "type": data.get("document_type", ""),
            "name": (ent.get("nome_fundo", ["N/A"]) or ["N/A"])[0],
            "cnpj": (ent.get("cnpj_fundo", ["N/A"]) or ["N/A"])[0],
            "entity_count": sum(len(v) for v in ent.values()),
        }
        # Collect fee information
        for fee in ["taxa_administracao", "taxa_gestao", "taxa_performance", "taxa_custodia"]:
            if fee in ent:
                info[fee] = ent[fee]
        # Map service providers
        for role in ["administrador", "gestor", "custodiante", "auditor"]:
            if role in ent:
                for name in ent[role]:
                    comparative["service_providers"].setdefault(role, {}).setdefault(
                        name.strip()[:100], []
                    ).append(info["name"])
        comparative["funds"].append(info)
        for a in data.get("alerts", []):
            a["fund"] = info["name"]
            comparative["consolidated_alerts"].append(a)

    # Save comparative JSON
    comp_path = os.path.join(output_dir, "comparative_report.json")
    ensure_output_dir(comp_path)
    with open(comp_path, "w", encoding="utf-8") as f:
        json.dump(comparative, f, ensure_ascii=False, indent=2)

    # Save comparative CSV
    csv_path = os.path.join(output_dir, "comparative_report.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f, fieldnames=["file", "type", "name", "cnpj", "entity_count"],
            extrasaction="ignore",
        )
        w.writeheader()
        for fund in comparative["funds"]:
            w.writerow(fund)

    # Print summary
    print(f"\n{'='*70}")
    print(f"📊 COMPARATIVE REPORT — {len(fund_data)} documents")
    print(f"{'='*70}")
    for f_info in comparative["funds"]:
        print(f"   • {f_info['name']} ({f_info['cnpj']}) — {f_info['entity_count']} entities")
    if comparative["consolidated_alerts"]:
        print(f"\n🚨 CONSOLIDATED ALERTS ({len(comparative['consolidated_alerts'])}):")
        for a in comparative["consolidated_alerts"]:
            print(f"   [{a.get('fund', '?')}] {a.get('type', a.get('tipo', '?'))}")
    print(f"\n📁 Report: {output_dir}/comparative_report.json")
    return comparative


# ============================================================
# Batch processing
# ============================================================


def collect_pdfs(paths):
    """Collect PDF file paths from given paths (files or directories)."""
    pdfs = []
    for p in paths:
        if os.path.isdir(p):
            for ext in ("*.pdf", "*.PDF"):
                pdfs.extend(glob.glob(os.path.join(p, "**", ext), recursive=True))
        elif os.path.isfile(p) and p.lower().endswith(".pdf"):
            pdfs.append(p)
        else:
            pdfs.extend(f for f in glob.glob(p) if f.lower().endswith(".pdf"))
    return sorted(set(pdfs))


def process_batch(
    paths, doc_type="auto", model="gemini-2.0-flash",
    passes=1, workers=3, output_dir="output_batch",
    compare=False, fast=False, chunk_size=3000, max_chars=50000,
):
    """Process multiple PDFs in batch."""
    if fast:
        passes, workers, chunk_size, max_chars = 1, 3, 3000, 15000
        print(f"⚡ Fast mode: 1 pass, chunk 3K, max 15K chars\n")

    pdfs = collect_pdfs(paths)
    if not pdfs:
        print("❌ No PDF files found.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    print(f"{'='*70}")
    print(f"📂 BATCH PROCESSING — {len(pdfs)} PDFs")
    print(f"🤖 {model} | passes={passes} | workers={workers} | chunk={chunk_size} | max_chars={max_chars}")
    print(f"{'='*70}\n")

    results = []
    start = time.time()

    for i, pdf in enumerate(pdfs, 1):
        print(f"\n[{i}/{len(pdfs)}] 📄 {os.path.basename(pdf)}")
        t0 = time.time()
        r = process_document(pdf, doc_type, model, passes, workers, output_dir, chunk_size, max_chars)
        dt = time.time() - t0
        r["processing_time_sec"] = round(dt, 1)

        if r["status"] == "success":
            print(f"   ✅ {r['type']} | {r['entity_count']} entities | {dt:.1f}s")
            if r.get("alerts"):
                print(f"   🚨 {len(r['alerts'])} alert(s)")
        else:
            print(f"   ❌ {r['error']}")
        results.append(r)

        # Delay between documents to avoid rate limiting
        if i < len(pdfs):
            time.sleep(3)

    # Save processing log
    log = {
        "timestamp": datetime.now().isoformat(), "model": model,
        "total": len(pdfs),
        "success": sum(1 for r in results if r["status"] == "success"),
        "errors": sum(1 for r in results if r["status"] == "error"),
        "total_time_sec": round(time.time() - start, 1),
        "results": results,
    }
    log_path = os.path.join(output_dir, "batch_log.json")
    ensure_output_dir(log_path)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"✅ DONE: {log['success']}/{log['total']} succeeded | {log['total_time_sec']}s total")
    print(f"{'='*70}")

    if compare and log["success"] > 1:
        generate_comparative_report(results, output_dir)

    return log


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Batch process Brazilian fund documents with auto type detection",
    )
    parser.add_argument("paths", nargs="+", help="PDF files or folders")
    parser.add_argument("--type", choices=["auto", "regulation", "quarterly"], default="auto")
    parser.add_argument("--model", default="gemini-2.0-flash")
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--output", default="output_batch")
    parser.add_argument("--chunk-size", type=int, default=3000)
    parser.add_argument("--max-chars", type=int, default=50000)
    parser.add_argument("--compare", action="store_true", help="Generate comparative report")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--no-filter", action="store_true")

    args = parser.parse_args()
    if args.no_filter:
        args.max_chars = 0

    process_batch(
        args.paths, args.type, args.model, args.passes, args.workers,
        args.output, args.compare, args.fast, args.chunk_size, args.max_chars,
    )
