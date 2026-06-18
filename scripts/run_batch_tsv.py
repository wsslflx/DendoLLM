#!/usr/bin/env python3
"""
Batch runner: execute the GraphRAG pipeline for every row in a TSV file.

Creates one timestamped batch root folder, then one sub-bundle per gene inside it.
Species already indexed/enriched in Neo4j and already ingested in Chroma are
automatically reused — no re-processing happens unless --force-reindex /
--force-reenrich are passed explicitly.

Usage:
    python scripts/run_batch_tsv.py --tsv candidate_sets_v1_le8.tsv
    python scripts/run_batch_tsv.py --tsv candidate_sets_v1_le8.tsv --dry-run
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys
import time
from datetime import datetime

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))


def _species_json(species_raw: str) -> list[dict]:
    """Convert a comma-separated species string (underscores OK) to pipeline JSON."""
    entries = []
    seen: set[str] = set()
    for raw in species_raw.split(","):
        raw = raw.strip()
        if not raw:
            continue
        canonical = raw.replace("_", " ").strip().lower()
        # deduplicate
        if canonical in seen:
            continue
        seen.add(canonical)
        # include original capitalised form as alias for broader search
        alias = raw.replace("_", " ").strip()
        aliases = [alias] if alias.lower() != canonical else []
        entries.append({"canonical": canonical, "aliases": aliases})
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GraphRAG pipeline for every gene in a TSV file."
    )
    parser.add_argument("--tsv", required=True, help="Path to input TSV file.")
    parser.add_argument(
        "--batch-root",
        default="logs_graph",
        help="Parent directory for all output (default: logs_graph).",
    )
    parser.add_argument(
        "--shared-chroma-dir",
        default="shared_chroma",
        help="Shared Chroma vectorstore reused across all genes (default: shared_chroma).",
    )
    parser.add_argument(
        "--shared-pdf-dir",
        default="shared_pdfs",
        help="Shared PDF cache reused across all genes (default: shared_pdfs).",
    )
    parser.add_argument("--model", default=None, help="Override synthesis LLM model.")
    parser.add_argument("--index-model", default=None, help="Override entity extraction model.")
    parser.add_argument("--mapping-model", default="granite4.1:8b", help="LLM model for uPheno entity normalization and Stage 3 verification (default: granite4.1:8b).")
    parser.add_argument(
        "--index-workers", type=int, default=4,
        help="Parallel threads for chunk entity extraction (default: 4).",
    )
    parser.add_argument(
        "--norm-batch-size", type=int, default=10,
        help="Traits per normalization LLM call (default: 10).",
    )
    parser.add_argument(
        "--map-workers", type=int, default=4,
        help="Parallel threads for uPheno mapping (default: 4).",
    )
    parser.add_argument(
        "--force-reindex", action="store_true",
        help="Re-index all chunks even if already in Neo4j.",
    )
    parser.add_argument(
        "--force-reenrich", action="store_true",
        help="Re-enrich all entities even if already enriched.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print commands without executing them.",
    )
    parser.add_argument(
        "--skip-from", default=None,
        help="Gene name to resume from (skips all genes before it alphabetically).",
    )
    args = parser.parse_args()

    tsv_path = pathlib.Path(args.tsv)
    if not tsv_path.exists():
        raise SystemExit(f"TSV file not found: {tsv_path}")

    # Read TSV
    import csv
    rows = []
    with open(tsv_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            rows.append(row)

    print(f"[BatchRunner] Loaded {len(rows)} testcases from {tsv_path}")

    # Create batch root folder
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    batch_label = tsv_path.stem
    batch_dir = pathlib.Path(args.batch_root) / f"{timestamp}-batch-{batch_label}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    print(f"[BatchRunner] Batch output root: {batch_dir}")

    # Shared dirs
    chroma_dir = pathlib.Path(args.shared_chroma_dir)
    pdf_dir = pathlib.Path(args.shared_pdf_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    # Species JSON dir inside the batch folder
    species_dir = batch_dir / "species_files"
    species_dir.mkdir(parents=True, exist_ok=True)

    pipeline_script = pathlib.Path(__file__).parents[1] / "pipeline" / "run_graph_pipeline.py"

    failed: list[str] = []
    gene_timings: dict[str, float] = {}
    batch_start = time.monotonic()
    skipping = bool(args.skip_from)

    for i, row in enumerate(rows):
        gene = row.get("Gene", "").strip()
        species_raw = row.get("Species", "").strip()

        if not gene or not species_raw:
            print(f"[BatchRunner] Skipping row {i+1}: missing gene or species.")
            continue

        # --skip-from support
        if skipping:
            if gene == args.skip_from:
                skipping = False
            else:
                print(f"[BatchRunner] [{i+1}/{len(rows)}] Skipping {gene} (before --skip-from).")
                continue

        # Write species JSON
        entries = _species_json(species_raw)
        species_file = species_dir / f"{gene}.json"
        species_file.write_text(
            json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        print(f"\n[BatchRunner] [{i+1}/{len(rows)}] Gene: {gene} — {len(entries)} species")

        cmd = [
            sys.executable, str(pipeline_script),
            "--species-file", str(species_file),
            "--runs", "1",
            "--bundle-root", str(batch_dir),
            "--run-label", gene,
            "--shared-chroma-dir", str(chroma_dir),
            "--pdf-dir", str(pdf_dir),
            "--index-workers", str(args.index_workers),
            "--norm-batch-size", str(args.norm_batch_size),
            "--map-workers", str(args.map_workers),
        ]

        if args.model:
            cmd += ["--model", args.model]
        if args.index_model:
            cmd += ["--index-model", args.index_model]
        cmd += ["--mapping-model", args.mapping_model]
        if args.force_reindex:
            cmd.append("--force-reindex")
        if args.force_reenrich:
            cmd.append("--force-reenrich")

        if args.dry_run:
            print(f"[BatchRunner] DRY RUN: {' '.join(cmd)}")
            continue

        t0 = time.monotonic()
        result = subprocess.run(cmd)
        elapsed = time.monotonic() - t0
        gene_timings[gene] = round(elapsed, 1)
        elapsed_str = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        if result.returncode != 0:
            print(f"[BatchRunner] WARNING: gene {gene} exited with code {result.returncode} in {elapsed_str}.")
            failed.append(gene)
        else:
            print(f"[BatchRunner] [{i+1}/{len(rows)}] {gene} done in {elapsed_str}.")

    # Summary
    total_elapsed = time.monotonic() - batch_start
    total_str = f"{int(total_elapsed // 3600)}h {int((total_elapsed % 3600) // 60)}m {int(total_elapsed % 60)}s"
    print(f"\n[BatchRunner] ============================")
    print(f"[BatchRunner] Batch complete in {total_str}.")
    print(f"[BatchRunner] Output: {batch_dir}")
    if gene_timings:
        slowest = max(gene_timings, key=gene_timings.get)
        fastest = min(gene_timings, key=gene_timings.get)
        avg = sum(gene_timings.values()) / len(gene_timings)
        print(f"[BatchRunner] Timing — avg: {avg/60:.1f}m  fastest: {fastest} ({gene_timings[fastest]/60:.1f}m)  slowest: {slowest} ({gene_timings[slowest]/60:.1f}m)")
    if failed:
        print(f"[BatchRunner] Failed genes ({len(failed)}): {', '.join(failed)}")
    else:
        print(f"[BatchRunner] All genes completed successfully.")

    # Write a simple manifest
    manifest = {
        "tsv": str(tsv_path),
        "batch_dir": str(batch_dir),
        "n_testcases": len(rows),
        "failed": failed,
        "shared_chroma_dir": str(chroma_dir),
        "shared_pdf_dir": str(pdf_dir),
        "total_elapsed_seconds": round(total_elapsed, 1),
        "gene_timings_seconds": gene_timings,
    }
    (batch_dir / "batch_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
