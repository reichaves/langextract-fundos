#!/usr/bin/env python3
"""
comparar_versoes.py — A/B comparison harness for langextract version/model swaps.

PURPOSE:
    The reports this repo normally writes (output/<stem>_report.json) are a lossy
    projection: `entities` is dict[extraction_class -> list[str]], so char_interval
    and alignment_status are discarded. That makes them useless for judging a
    langextract upgrade, because source grounding is exactly what can regress
    (see upstream PR #485, which relocates repeated-mention spans).

    This harness runs the same extraction through the same production code path
    (utils.extract_with_backoff -> lx.extract) but keeps the full Extraction
    objects, so two runs can be diffed on what actually matters:
    class coverage, blank rate and alignment — not just entity counts.

    Written for the 1.6.0 -> 1.7.0 evaluation, but the same two modes work for
    any model swap (--model), which is a recurring need in this project.

AUTHOR:  Reinaldo Chaves (reichaves@gmail.com)
DATE:    2026-09-23
DEPS:    langextract, pdfplumber, python-dotenv (see requirements.txt)

USAGE:
    python comparar_versoes.py run <pdf> --label v1_6_0_a
    python comparar_versoes.py run <pdf> --label smoke --groups A --max-chars 6000
    python comparar_versoes.py diff output/ab/v1_6_0_a.json output/ab/v1_7_0.json
"""

import argparse
import hashlib
import json
import os
import re
import sys
import time
from importlib import metadata
from typing import Any, Optional

from extrair_regulamento import (
    EXAMPLE_GROUP_A,
    EXAMPLE_GROUP_B,
    EXAMPLE_GROUP_C,
    PROMPT_GROUP_A,
    PROMPT_GROUP_B,
    PROMPT_GROUP_C,
    _is_parse_error,
)
from utils import configure_model, ensure_output_dir, extract_pdf_text, extract_with_backoff

AB_DIR = os.path.join("output", "ab")

# Placeholder strings the model emits when it finds nothing for a class. These
# are blanks in substance: they carry no information, cannot be aligned to the
# source (alignment_status comes back None), and would otherwise flow into the
# report a journalist reads as if they were extracted values.
PLACEHOLDER_TEXTS = frozenset({"null", "none", "n/a", "na", "-", "--", "nan", "nenhum", "não informado"})

# Same three groups as extrair_regulamento.extract_regulation, keyed by a short
# id so --groups can select a subset for a cheap smoke test.
GROUPS = {
    "A": {"name": "A (ID + Providers)", "prompt": PROMPT_GROUP_A, "example": EXAMPLE_GROUP_A},
    "B": {"name": "B (Fees + Structure)", "prompt": PROMPT_GROUP_B, "example": EXAMPLE_GROUP_B},
    "C": {"name": "C (Policy + Risk + Events)", "prompt": PROMPT_GROUP_C, "example": EXAMPLE_GROUP_C},
}

# Chunk-size escalation, identical to extrair_regulamento.py, so a parse error
# degrades the same way here as in production.
def _chunk_attempts(chunk_size: int) -> list:
    """Return the chunk sizes to try in order, mirroring the production retry chain."""
    return [chunk_size, max(1000, chunk_size // 2), 1000]


def expected_classes() -> dict:
    """
    Enumerate the entity classes each prompt group asks for.

    The prompts are the only complete source: every class appears as a bullet
    line "    - nome_fundo: description". The few-shot EXAMPLE_GROUP_* objects
    cover only 18 of the 22 classes, so they must not be used as the authority.

    Returns:
        Mapping of group id -> list of extraction_class names, in prompt order.
    """
    return {gid: re.findall(r"^\s*-\s(\w+):", g["prompt"], re.M) for gid, g in GROUPS.items()}


def _sha256(text: str) -> str:
    """Return the hex SHA-256 of the reduced input text, used as an A/B sanity lock."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _langextract_version() -> str:
    """Return the installed langextract version, or 'unknown' if metadata is missing."""
    try:
        return metadata.version("langextract")
    except metadata.PackageNotFoundError:
        return "unknown"


def _serialize(ext: Any, group_id: str) -> dict:
    """
    Flatten one langextract Extraction, preserving the grounding fields.

    Unlike the production report builders, this keeps char_interval and
    alignment_status — the two fields an alignment A/B depends on.
    """
    interval = getattr(ext, "char_interval", None)
    status = getattr(ext, "alignment_status", None)
    return {
        "group": group_id,
        "extraction_class": ext.extraction_class,
        "extraction_text": ext.extraction_text,
        "char_interval": (
            {"start_pos": interval.start_pos, "end_pos": interval.end_pos} if interval else None
        ),
        # AlignmentStatus is an enum; store its value so the dump stays plain JSON.
        "alignment_status": getattr(status, "value", status),
    }


def run_arm(
    pdf_path: str,
    label: str,
    model: str = "gemini-2.5-flash",
    passes: int = 1,
    workers: int = 3,
    chunk_size: int = 3000,
    max_chars: int = 50000,
    temperature: float = 0.0,
    group_ids: Optional[list] = None,
) -> dict:
    """
    Run one arm of the comparison and write its dump to output/ab/<label>.json.

    Mirrors extrair_regulamento.extract_regulation's group loop, with two
    deliberate differences: temperature is pinned (production leaves it at the
    provider default, which would make the A/B non-deterministic), and the full
    Extraction objects are kept instead of only extraction_text.

    Args:
        pdf_path:    Regulation PDF to extract from.
        label:       Arm name; also the output filename stem.
        temperature: Pinned to 0.0 to reduce run-to-run variance.
        group_ids:   Subset of GROUPS keys to run; None means all three.

    Returns:
        The dump dict that was written to disk.
    """
    group_ids = group_ids or list(GROUPS)
    print(f"🔬 Arm: {label}")
    print(f"   langextract {_langextract_version()} | model {model} | temperature {temperature}")

    text = extract_pdf_text(pdf_path, max_chars=max_chars)
    print(f"   📄 Text for processing: {len(text):,} chars")
    if len(text) < 100:
        print("   ❌ Text too short — PDF may be scanned. Aborting.")
        sys.exit(1)

    config = configure_model(model)
    dump = {
        "label": label,
        "langextract_version": _langextract_version(),
        "model": model,
        "temperature": temperature,
        "params": {
            "passes": passes,
            "workers": workers,
            "chunk_size": chunk_size,
            "max_chars": max_chars,
            "groups": group_ids,
        },
        "input": {
            "pdf": os.path.basename(pdf_path),
            "text_len": len(text),
            "text_sha256": _sha256(text),
        },
        "groups": [],
        "extractions": [],
    }

    for gid in group_ids:
        group = GROUPS[gid]
        print(f"   📋 Group {group['name']}...")
        started = time.time()
        result = None
        error = None
        attempts = []

        for attempt_chunk in _chunk_attempts(chunk_size):
            try:
                result = extract_with_backoff(
                    text_or_documents=text,
                    prompt_description=group["prompt"],
                    examples=[group["example"]],
                    extraction_passes=passes,
                    max_workers=workers,
                    max_char_buffer=attempt_chunk,
                    temperature=temperature,
                    resolver_params={"suppress_parse_errors": False},
                    **config,
                )
                attempts.append({"chunk_size": attempt_chunk, "outcome": "ok"})
                break
            except Exception as e:
                # Record the exception TYPE, not just the message: upstream PR
                # #534 can turn a truncated response into InferenceRuntimeError,
                # which _is_parse_error() does not match, so the chunk-halving
                # chain below would silently stop firing. This field is how that
                # regression becomes visible in the diff.
                attempts.append(
                    {
                        "chunk_size": attempt_chunk,
                        "outcome": "error",
                        "exception_type": type(e).__name__,
                        "is_parse_error": bool(_is_parse_error(e)),
                        "message": str(e)[:300],
                    }
                )
                if _is_parse_error(e) and attempt_chunk > 1000:
                    print(f"      ⚠️  Parse error (chunk={attempt_chunk}), retrying smaller...")
                    continue
                error = str(e)
                print(f"      ❌ Group {gid} failed: {e}")
                break

        extractions = list(getattr(result, "extractions", None) or []) if result else []
        for ext in extractions:
            dump["extractions"].append(_serialize(ext, gid))

        dump["groups"].append(
            {
                "id": gid,
                "name": group["name"],
                "winning_chunk_size": attempts[-1]["chunk_size"] if result else None,
                "attempts": attempts,
                "error": error,
                "elapsed_s": round(time.time() - started, 1),
                "n_extractions": len(extractions),
            }
        )
        print(f"      {'✅' if extractions else '⚠️ '} {len(extractions)} extractions")

    out_path = os.path.join(AB_DIR, f"{label}.json")
    ensure_output_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(dump, fh, ensure_ascii=False, indent=2)
    print(f"   💾 Saved {out_path}")
    return dump


# ============================================================
# Diff
# ============================================================


def _is_blank(text: Optional[str]) -> bool:
    """Whether an extraction carries no usable value (empty or a placeholder token)."""
    stripped = (text or "").strip()
    return not stripped or stripped.lower() in PLACEHOLDER_TEXTS


def _metrics(dump: dict) -> dict:
    """Compute coverage, blank-rate and alignment metrics for one arm's dump."""
    by_class: dict = {}
    empties = 0
    placeholders = 0
    alignment: dict = {}

    for ext in dump["extractions"]:
        cls = ext["extraction_class"]
        text = ext["extraction_text"] or ""
        by_class.setdefault(cls, []).append(ext)
        if not text.strip():
            empties += 1
        elif text.strip().lower() in PLACEHOLDER_TEXTS:
            placeholders += 1
        status = ext["alignment_status"] or "none"
        alignment[status] = alignment.get(status, 0) + 1

    total = len(dump["extractions"])
    blanks = empties + placeholders
    return {
        "total": total,
        "empties": empties,
        "placeholders": placeholders,
        "blanks": blanks,
        "blank_rate": (blanks / total) if total else 0.0,
        "by_class": by_class,
        "alignment": alignment,
    }


def _blank_rate_by_class(by_class: dict) -> dict:
    """Return class -> blank fraction, for spotting a class that degraded alone."""
    out = {}
    for cls, items in by_class.items():
        blank = sum(1 for e in items if _is_blank(e["extraction_text"]))
        out[cls] = blank / len(items) if items else 0.0
    return out


def _span_index(dump: dict) -> dict:
    """Map (class, text) -> set of (start_pos, end_pos), for comparing grounding."""
    index: dict = {}
    for ext in dump["extractions"]:
        interval = ext["char_interval"]
        if not interval:
            continue
        key = (ext["extraction_class"], (ext["extraction_text"] or "").strip())
        index.setdefault(key, set()).add((interval["start_pos"], interval["end_pos"]))
    return index


def diff_arms(path_a: str, path_b: str) -> int:
    """
    Compare two arm dumps and print the evaluation.

    Ordered by how much each metric actually tells you: class coverage first
    (a missing class silently breaks downstream consumers), then blank rate,
    then alignment, with raw entity count last because it is the weakest signal.

    Returns:
        Process exit code: 0 if the comparison is valid, 1 if it is not
        trustworthy (input mismatch or a failed group in either arm).
    """
    with open(path_a, encoding="utf-8") as fh:
        a = json.load(fh)
    with open(path_b, encoding="utf-8") as fh:
        b = json.load(fh)

    la, lb = a["label"], b["label"]
    print("=" * 70)
    print(f"A = {la}  (langextract {a['langextract_version']}, {a['model']})")
    print(f"B = {lb}  (langextract {b['langextract_version']}, {b['model']})")
    print("=" * 70)

    # --- 0. Integrity: is this comparison even valid? ---
    print("\n0️⃣  Comparison integrity")
    valid = True
    if a["input"]["text_sha256"] != b["input"]["text_sha256"]:
        print("   ❌ Input text differs between arms — the comparison is invalid.")
        print(f"      A: {a['input']['text_len']:,} chars / B: {b['input']['text_len']:,} chars")
        valid = False
    else:
        print(f"   ✅ Identical input text ({a['input']['text_len']:,} chars, sha256 match)")

    for dump in (a, b):
        failed = [g["id"] for g in dump["groups"] if g["error"]]
        if failed:
            print(f"   ❌ {dump['label']}: group(s) {', '.join(failed)} failed — arm incomplete.")
            valid = False

    chunks_a = {g["id"]: g["winning_chunk_size"] for g in a["groups"]}
    chunks_b = {g["id"]: g["winning_chunk_size"] for g in b["groups"]}
    if chunks_a != chunks_b:
        print(f"   ⚠️  Chunk sizes differ: A={chunks_a} B={chunks_b}")
        print("      Alignment differences below are confounded with chunking differences.")
    else:
        print(f"   ✅ Same winning chunk size per group: {chunks_a}")

    # Surface non-parse exceptions, which are the PR #534 regression signature.
    for dump in (a, b):
        for g in dump["groups"]:
            for att in g["attempts"]:
                if att["outcome"] == "error" and not att["is_parse_error"]:
                    print(
                        f"   ⚠️  {dump['label']} group {g['id']}: {att['exception_type']} "
                        f"is not a parse error — chunk-halving retry did NOT fire."
                    )

    ma, mb = _metrics(a), _metrics(b)
    expected = expected_classes()
    ran = set(a["params"]["groups"]) & set(b["params"]["groups"])
    wanted = [c for gid in sorted(ran) for c in expected[gid]]

    # --- 1. Class coverage ---
    print(f"\n1️⃣  Class coverage  ({len(wanted)} classes expected across groups {sorted(ran)})")
    missing_a = [c for c in wanted if c not in ma["by_class"]]
    missing_b = [c for c in wanted if c not in mb["by_class"]]
    print(f"   {la}: {len(wanted) - len(missing_a)}/{len(wanted)} present")
    print(f"   {lb}: {len(wanted) - len(missing_b)}/{len(wanted)} present")
    lost = [c for c in missing_b if c not in missing_a]
    gained = [c for c in missing_a if c not in missing_b]
    if lost:
        print(f"   ❌ Classes lost in {lb}: {', '.join(lost)}")
    if gained:
        print(f"   ✅ Classes recovered in {lb}: {', '.join(gained)}")
    if not lost and not gained:
        print("   ✅ No change in class coverage")
    unexpected = sorted(set(ma["by_class"]) | set(mb["by_class"]) - set(wanted))
    unexpected = [c for c in unexpected if c not in wanted]
    if unexpected:
        print(f"   ℹ️  Classes returned but never requested: {', '.join(unexpected)}")

    # --- 2. Blank rate ---
    print("\n2️⃣  Blank rate (empty, whitespace-only, or a placeholder like 'null')")
    for lbl, m in ((la, ma), (lb, mb)):
        print(
            f"   {lbl}: {m['blanks']}/{m['total']} ({m['blank_rate']:.1%})"
            f"  [{m['empties']} empty + {m['placeholders']} placeholder]"
        )
    ba, bb = _blank_rate_by_class(ma["by_class"]), _blank_rate_by_class(mb["by_class"])
    worse = [(c, ba.get(c, 0.0), bb[c]) for c in bb if bb[c] - ba.get(c, 0.0) > 0.1]
    for cls, ra, rb in sorted(worse, key=lambda x: x[2] - x[1], reverse=True):
        print(f"   ⚠️  {cls}: {ra:.0%} → {rb:.0%} blank")

    # --- 3. Alignment (upstream PR #485) ---
    print("\n3️⃣  Alignment status distribution")
    for status in sorted(set(ma["alignment"]) | set(mb["alignment"])):
        va, vb = ma["alignment"].get(status, 0), mb["alignment"].get(status, 0)
        arrow = "" if va == vb else f"  ({vb - va:+d})"
        print(f"   {status:<16} {la}: {va:<5} {lb}: {vb}{arrow}")

    # --- 4. Span shifts ---
    print("\n4️⃣  Source grounding shifts (same class + text, different offsets)")
    ia, ib = _span_index(a), _span_index(b)
    shared = set(ia) & set(ib)
    shifted = [k for k in shared if ia[k] != ib[k]]
    print(f"   {len(shifted)}/{len(shared)} shared extractions changed span")
    for cls, text in sorted(shifted)[:5]:
        print(f"      {cls}: {sorted(ia[(cls, text)])} → {sorted(ib[(cls, text)])}  {text[:45]!r}")
    if len(shifted) > 5:
        print(f"      ... and {len(shifted) - 5} more")

    # --- 5. Entity count (weakest signal, reported last on purpose) ---
    print("\n5️⃣  Entity count (weakest signal — do not decide on this alone)")
    delta = mb["total"] - ma["total"]
    print(f"   {la}: {ma['total']}   {lb}: {mb['total']}   ({delta:+d})")

    print("\n" + "=" * 70)
    if not valid:
        print("❌ Comparison NOT trustworthy — see integrity section above.")
        return 1
    print("✅ Comparison valid. Judge A/B deltas against the same-version noise floor.")
    return 0


def main() -> int:
    """Parse arguments and dispatch to the run or diff mode."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="mode", required=True)

    run_p = sub.add_parser("run", help="Run one arm and dump grounded extractions")
    run_p.add_argument("pdf", help="Path to the regulation PDF")
    run_p.add_argument("--label", required=True, help="Arm name (also the output filename)")
    run_p.add_argument("--model", default="gemini-2.5-flash")
    run_p.add_argument("--passes", type=int, default=1)
    run_p.add_argument("--workers", type=int, default=3)
    run_p.add_argument("--chunk-size", type=int, default=3000)
    run_p.add_argument("--max-chars", type=int, default=50000)
    run_p.add_argument("--temperature", type=float, default=0.0)
    run_p.add_argument("--groups", default="ABC", help="Subset of groups to run, e.g. 'A' or 'AC'")

    diff_p = sub.add_parser("diff", help="Compare two arm dumps")
    diff_p.add_argument("arm_a")
    diff_p.add_argument("arm_b")

    args = parser.parse_args()

    if args.mode == "run":
        bad = [g for g in args.groups.upper() if g not in GROUPS]
        if bad:
            print(f"❌ Unknown group(s): {', '.join(bad)}. Valid: {', '.join(GROUPS)}")
            return 1
        if not os.path.exists(args.pdf):
            print(f"❌ PDF not found: {args.pdf}")
            return 1
        run_arm(
            pdf_path=args.pdf,
            label=args.label,
            model=args.model,
            passes=args.passes,
            workers=args.workers,
            chunk_size=args.chunk_size,
            max_chars=args.max_chars,
            temperature=args.temperature,
            group_ids=list(args.groups.upper()),
        )
        return 0

    return diff_arms(args.arm_a, args.arm_b)


if __name__ == "__main__":
    sys.exit(main())
