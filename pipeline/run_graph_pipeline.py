#!/usr/bin/env python3
"""
GraphRAG full pipeline orchestrator — parallel to run_full_pipeline_v4.py.

Runs graph_inventory_single.py per species (per run), then the KG step.
On multi-run mode, graph indexing is only done on run 1 (--skip-index on subsequent runs).
Old v4 pipeline is untouched.

Usage:
    python pipeline/run_graph_pipeline.py --species-file testcase1.json --runs 2
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
    reuse_traits: bool,
    skip_ingest_after_first: bool,
    traits_dir: pathlib.Path,
    pdf_dir: pathlib.Path,
    chroma_dir: pathlib.Path,
    ingest_lock_file: pathlib.Path,
    hybrid_sim_threshold: float,
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
            "--traits-dir", str(traits_dir),
            "--pdf-dir", str(pdf_dir),
            "--chroma-dir", str(chroma_dir),
            "--ingest-lock-file", str(ingest_lock_file),
            "--hybrid-sim-threshold", str(hybrid_sim_threshold),
            "--model", llm_model,
        ]

        if reuse_traits:
            cmd.append("--reuse-traits")

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


def _is_wiki_source(value: str | None) -> bool:
    if not value:
        return False
    return value.startswith("wiki:") or value.startswith("wikipedia:")


def aggregate_source_stats(run_dirs: list[pathlib.Path]) -> dict:
    per_run: list[dict] = []
    total_papers_fetched: set[str] = set()
    total_papers_used: set[str] = set()
    total_wiki_fetched: set[str] = set()
    total_wiki_used: set[str] = set()

    for run_dir in run_dirs:
        papers_fetched: set[str] = set()
        papers_used: set[str] = set()
        wiki_fetched: set[str] = set()
        wiki_used: set[str] = set()

        for species_dir in run_dir.iterdir():
            if not species_dir.is_dir() or species_dir.name == "synthesis":
                continue

            ingested_path = species_dir / "ingested_docs.json"
            if ingested_path.exists():
                try:
                    ingested_docs = json.loads(ingested_path.read_text(encoding="utf-8"))
                except Exception:
                    ingested_docs = []
                if isinstance(ingested_docs, list):
                    for entry in ingested_docs:
                        if not isinstance(entry, str):
                            continue
                        if _is_wiki_source(entry):
                            wiki_fetched.add(entry)
                        else:
                            papers_fetched.add(entry)

            # New pipeline writes graph_context.txt instead of used_chunks.json
            # but source_chunk_ids in traits JSON can be parsed for stats
            traits_path = species_dir.parent.parent.parent / "traits" / f"{species_dir.name}.json"
            if traits_path.exists():
                try:
                    traits = json.loads(traits_path.read_text(encoding="utf-8"))
                    for tr in traits:
                        for src in tr.get("sources", []):
                            if _is_wiki_source(src):
                                wiki_used.add(src)
                            elif src:
                                papers_used.add(src)
                except Exception:
                    pass

        total_papers_fetched.update(papers_fetched)
        total_papers_used.update(papers_used)
        total_wiki_fetched.update(wiki_fetched)
        total_wiki_used.update(wiki_used)

        per_run.append({
            "run_dir": str(run_dir),
            "papers_fetched": len(papers_fetched),
            "papers_used": len(papers_used),
            "wiki_fetched": len(wiki_fetched),
            "wiki_used": len(wiki_used),
        })

    return {
        "overall": {
            "papers_fetched": len(total_papers_fetched),
            "papers_used": len(total_papers_used),
            "wiki_fetched": len(total_wiki_fetched),
            "wiki_used": len(total_wiki_used),
        },
        "per_run": per_run,
    }


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
        description="Run GraphRAG pipeline: graph-indexed retrieval + trait extraction + KG step."
    )
    species_group = parser.add_mutually_exclusive_group(required=True)
    species_group.add_argument("--species-file", help="Path to species JSON file.")
    species_group.add_argument(
        "--species-list",
        help="Comma-separated scientific names.",
    )
    parser.add_argument("--generated-species-file", help="Output path for generated species JSON.")
    parser.add_argument("--runs", type=int, default=1, help="Number of inventory runs.")
    parser.add_argument("--reuse-traits", action="store_true")
    parser.add_argument(
        "--skip-ingest-after-first",
        action="store_true",
        default=True,
        help="Only ingest on first run (default: on).",
    )
    parser.add_argument(
        "--hybrid-sim-threshold",
        type=float,
        default=0.82,
    )
    parser.add_argument("--run-list-out", help="Where to write run list.")
    parser.add_argument(
        "--bundle-root",
        default="logs_graph",
        help="Base directory for output bundles (default: logs_graph).",
    )
    parser.add_argument("--run-label", help="Optional label for bundle directory name.")
    parser.add_argument("--traits-dir")
    parser.add_argument("--pdf-dir")
    parser.add_argument("--chroma-dir")
    parser.add_argument("--ingest-lock-file", default=".ingest.lock")
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--kg-hypotheses",
        action="store_true",
        default=False,
        help="Enable LLM hypothesis generation at the KG step.",
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

    traits_dir = pathlib.Path(args.traits_dir) if args.traits_dir else bundle_dir / "traits"
    pdf_dir = pathlib.Path(args.pdf_dir) if args.pdf_dir else bundle_dir / "cache" / "pdfs"
    chroma_dir = (
        pathlib.Path(args.chroma_dir)
        if args.chroma_dir
        else bundle_dir / "cache" / "chroma_store_ollama"
    )
    ingest_lock_file = pathlib.Path(args.ingest_lock_file)
    traits_dir.mkdir(parents=True, exist_ok=True)
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
        reuse_traits=args.reuse_traits,
        skip_ingest_after_first=args.skip_ingest_after_first,
        traits_dir=traits_dir,
        pdf_dir=pdf_dir,
        chroma_dir=chroma_dir,
        ingest_lock_file=ingest_lock_file,
        hybrid_sim_threshold=args.hybrid_sim_threshold,
        llm_model=args.model,
    )
    source_stats = aggregate_source_stats(run_dirs)

    # KG step (reused unchanged from v4)
    try:
        from kg.kg_pipeline_step import run_kg_step
        run_kg_step(str(bundle_dir), model=args.model, generate_hypotheses=args.kg_hypotheses)
    except Exception as _kg_exc:
        print(f"[KG] KG step failed (non-fatal) — pipeline continues: {_kg_exc}")

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
        "reuse_traits": args.reuse_traits,
        "skip_ingest_after_first": args.skip_ingest_after_first,
        "hybrid_sim_threshold": args.hybrid_sim_threshold,
        "traits_dir": str(traits_dir),
        "pdf_dir": str(pdf_dir),
        "chroma_dir": str(chroma_dir),
        "ingest_lock_file": str(ingest_lock_file),
        "run_list": str(run_list_path),
        "model": args.model,
        "temperature": args.temperature,
        "timestamp": timestamp,
        "source_stats": source_stats,
        "run_dirs": [str(p) for p in run_dirs],
    }
    (summary_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[GraphPipeline] Done. Bundle: {bundle_dir}")


if __name__ == "__main__":
    main()
