#!/usr/bin/env python3
"""
GraphRAG full pipeline orchestrator — parallel to run_full_pipeline_v4.py.

Pipeline: ingest → graph index → uPheno enrich → three-tier synthesize.
Graph indexing is only done on run 1; subsequent runs skip index (--skip-index).
Old v4 pipeline is untouched.

Usage:
    python pipeline/run_graph_pipeline.py --species-file testcase1.json --runs 1
    python pipeline/run_graph_pipeline.py --species-list "talpa europaea,chrysochloris asiatica"
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parents[1]))
sys.path.insert(0, str(_Path(__file__).parent))

import argparse
import json
import pathlib
import subprocess
from datetime import datetime
from uuid import uuid4

from scripts.build_testcase_json import build_entries, parse_species_arg
from core.llm_backend import DEFAULT_CHAT_MODEL


def slugify(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def run_inventory(
    species_file: pathlib.Path,
    bundle_dir: pathlib.Path,
    runs: int,
    skip_ingest_after_first: bool,
    pdf_dir: pathlib.Path,
    chroma_dir: pathlib.Path,
    ingest_lock_file: pathlib.Path,
    llm_model: str,
) -> list[pathlib.Path]:
    run_dirs: list[pathlib.Path] = []
    runs_root = bundle_dir / "runs"
    runs_root.mkdir(parents=True, exist_ok=True)

    for i in range(runs):
        run_dir = runs_root / f"run_{i + 1:02d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(pathlib.Path(__file__).parent / "graph_inventory_single.py"),
            "--species-file", str(species_file),
            "--log-run",
            "--log-dir", str(run_dir),
            "--pdf-dir", str(pdf_dir),
            "--chroma-dir", str(chroma_dir),
            "--ingest-lock-file", str(ingest_lock_file),
            "--model", llm_model,
        ]

        if skip_ingest_after_first and i > 0:
            cmd.append("--skip-ingest")
            # Graph is already built after run 1 — skip indexing on subsequent runs
            cmd.append("--skip-index")

        print(f"\n[GraphPipeline] Run {i + 1}/{runs}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        run_dirs.append(run_dir)

    return run_dirs


def write_run_list(run_dirs: list[pathlib.Path], out_path: pathlib.Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [str(p) for p in run_dirs]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")




def resolve_species_file(
    species_file_arg: str | None,
    species_list_arg: str | None,
    generated_species_file_arg: str | None,
    default_generated_out: pathlib.Path,
) -> tuple[pathlib.Path, str]:
    if species_file_arg:
        species_file = pathlib.Path(species_file_arg)
        if not species_file.exists():
            raise SystemExit(f"Species file not found: {species_file}")
        return species_file, "file"

    species_list = parse_species_arg(species_list_arg or "")
    if not species_list:
        raise SystemExit("No species provided in --species-list.")

    out_path = pathlib.Path(generated_species_file_arg) if generated_species_file_arg else default_generated_out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    entries = build_entries(species_list)
    out_path.write_text(json.dumps(entries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Generated species file: {out_path}")
    return out_path, "list"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run GraphRAG pipeline: ingest → graph index → uPheno enrich → synthesize."
    )
    species_group = parser.add_mutually_exclusive_group(required=True)
    species_group.add_argument("--species-file", help="Path to species JSON file.")
    species_group.add_argument(
        "--species-list",
        help="Comma-separated scientific names.",
    )
    parser.add_argument("--generated-species-file", help="Output path for generated species JSON.")
    parser.add_argument("--runs", type=int, default=1, help="Number of inventory runs.")
    parser.add_argument(
        "--skip-ingest-after-first",
        action="store_true",
        default=True,
        help="Only ingest on first run (default: on).",
    )
    parser.add_argument("--run-list-out", help="Where to write run list.")
    parser.add_argument(
        "--bundle-root",
        default="logs_graph",
        help="Base directory for output bundles (default: logs_graph).",
    )
    parser.add_argument("--run-label", help="Optional label for bundle directory name.")
    parser.add_argument("--pdf-dir")
    parser.add_argument("--chroma-dir")
    parser.add_argument("--ingest-lock-file", default=".ingest.lock")
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--skip-enrich",
        action="store_true",
        default=False,
        help="Skip uPheno enrichment + DOC_SIMILAR_TO step (graph already enriched).",
    )
    parser.add_argument(
        "--force-reenrich",
        action="store_true",
        default=False,
        help="Re-enrich all DocEntity nodes even if upheno_enriched=true.",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.78,
        help="Cosine similarity threshold for DOC_SIMILAR_TO edges (default: 0.78).",
    )
    parser.add_argument(
        "--skip-synthesis",
        action="store_true",
        default=False,
        help="Skip three-tier cross-species synthesis step.",
    )
    parser.add_argument(
        "--min-species",
        type=int,
        default=None,
        help="Minimum species count for synthesis communities (default: max(2, n_species//2)).",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_label:
        label = slugify(args.run_label)
    elif args.species_file:
        label = slugify(pathlib.Path(args.species_file).stem)
    else:
        label = "generated_species"

    bundle_dir = pathlib.Path(args.bundle_root) / f"{timestamp}-{label}-{uuid4().hex[:8]}"
    summary_dir = bundle_dir / "summary"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    pdf_dir = pathlib.Path(args.pdf_dir) if args.pdf_dir else bundle_dir / "cache" / "pdfs"
    chroma_dir = (
        pathlib.Path(args.chroma_dir)
        if args.chroma_dir
        else bundle_dir / "cache" / "chroma_store_ollama"
    )
    ingest_lock_file = pathlib.Path(args.ingest_lock_file)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    ingest_lock_file.parent.mkdir(parents=True, exist_ok=True)

    species_file, species_input_mode = resolve_species_file(
        args.species_file,
        args.species_list,
        args.generated_species_file,
        bundle_dir / "species_input.json",
    )
    species_snapshot = bundle_dir / "species_input.json"
    species_snapshot.write_text(species_file.read_text(encoding="utf-8"), encoding="utf-8")

    print(f"\n[GraphPipeline] Bundle: {bundle_dir}")
    print(f"[GraphPipeline] Species file: {species_file}")
    print(f"[GraphPipeline] Runs: {args.runs}")

    run_dirs = run_inventory(
        species_file=species_file,
        bundle_dir=bundle_dir,
        runs=args.runs,
        skip_ingest_after_first=args.skip_ingest_after_first,
        pdf_dir=pdf_dir,
        chroma_dir=chroma_dir,
        ingest_lock_file=ingest_lock_file,
        llm_model=args.model,
    )

    # Parse species info from the species file for enrichment + synthesis
    try:
        species_groups = json.loads(species_file.read_text(encoding="utf-8"))
        species_norms = [e["canonical"].lower().strip() for e in species_groups if e.get("canonical")]
        species_display = [e["canonical"] for e in species_groups if e.get("canonical")]
    except Exception as exc:
        print(f"[GraphPipeline] Could not parse species file for enrichment/synthesis: {exc}")
        species_norms = []
        species_display = []

    min_species = args.min_species if args.min_species is not None else max(2, len(species_norms) // 2)

    # Enrichment step: map DocEntity → uPheno + DOC_SIMILAR_TO edges
    if not args.skip_enrich and species_norms:
        print(f"\n[GraphPipeline] ========================================")
        print(f"[GraphPipeline] Running enrichment for {len(species_norms)} species...")
        print(f"[GraphPipeline] ========================================")
        try:
            from kg.graph_enricher import enrich_doc_entities
            enrich_summary = enrich_doc_entities(
                species_norms=species_norms,
                similarity_threshold=args.similarity_threshold,
                model=args.model,
                force_reenrich=args.force_reenrich,
                log_dir=bundle_dir,
            )
            (bundle_dir / "graph_enricher_summary.json").write_text(
                json.dumps(enrich_summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"[GraphPipeline] Enrichment complete: {enrich_summary.get('entities_mapped', 0)} mapped, "
                  f"{enrich_summary.get('similarity_pairs_added', 0)} similarity pairs.")
        except Exception as exc:
            print(f"[GraphPipeline] Enrichment failed (non-fatal): {exc}")
    elif args.skip_enrich:
        print(f"[GraphPipeline] Skipping enrichment step (--skip-enrich).")

    # Three-tier synthesis step
    if not args.skip_synthesis and species_norms:
        print(f"\n[GraphPipeline] ========================================")
        print(f"[GraphPipeline] Running three-tier synthesis (min_species={min_species})...")
        print(f"[GraphPipeline] ========================================")
        try:
            from kg.graph_synthesizer import run_synthesis
            synthesis = run_synthesis(
                species_norms=species_norms,
                species_display_names=species_display,
                min_species=min_species,
                model=args.model,
                log_dir=bundle_dir,
            )
            (bundle_dir / "graph_synthesis.json").write_text(
                json.dumps(synthesis, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            n_communities = len(synthesis.get("communities", []))
            print(f"[GraphPipeline] Synthesis complete: {n_communities} communities found.")
            if n_communities:
                for comm in synthesis["communities"]:
                    print(f"[GraphPipeline]   [{comm.get('tier','?')}] {comm.get('label','?')} "
                          f"({comm.get('species_count',0)} species)")
        except Exception as exc:
            print(f"[GraphPipeline] Synthesis failed (non-fatal): {exc}")
    elif args.skip_synthesis:
        print(f"[GraphPipeline] Skipping synthesis step (--skip-synthesis).")

    run_list_path = (
        pathlib.Path(args.run_list_out) if args.run_list_out else bundle_dir / "run_list.txt"
    )
    write_run_list(run_dirs, run_list_path)
    print(f"\n[GraphPipeline] Run list written to: {run_list_path}")

    meta = {
        "pipeline": "graph_rag",
        "bundle_dir": str(bundle_dir),
        "run_label": args.run_label if args.run_label else label,
        "species_file": str(species_file),
        "species_snapshot": str(species_snapshot),
        "species_input_mode": species_input_mode,
        "species_list": args.species_list if args.species_list else None,
        "runs": args.runs,
        "skip_ingest_after_first": args.skip_ingest_after_first,
        "pdf_dir": str(pdf_dir),
        "chroma_dir": str(chroma_dir),
        "ingest_lock_file": str(ingest_lock_file),
        "run_list": str(run_list_path),
        "model": args.model,
        "temperature": args.temperature,
        "timestamp": timestamp,
        "run_dirs": [str(p) for p in run_dirs],
        "skip_enrich": args.skip_enrich,
        "force_reenrich": args.force_reenrich,
        "similarity_threshold": args.similarity_threshold,
        "skip_synthesis": args.skip_synthesis,
        "min_species": min_species,
        "species_norms": species_norms,
    }
    (summary_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[GraphPipeline] Done. Bundle: {bundle_dir}")


if __name__ == "__main__":
    main()
