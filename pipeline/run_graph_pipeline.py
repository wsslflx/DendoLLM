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
import time
from datetime import datetime
from uuid import uuid4

import requests

from scripts.build_testcase_json import build_entries, parse_species_arg
from core.llm_backend import DEFAULT_CHAT_MODEL


# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

def _preflight_check(args) -> None:
    """
    Verify Neo4j and LLM server are reachable before any work begins.
    Raises SystemExit immediately if either is down, so the operator sees a
    clear error instead of a silent zero-output run hours later.
    """
    from core.llm_backend import ollama_base_url, ollama_headers

    errors: list[str] = []

    # Neo4j — needed by indexer, enricher, synthesizer, fan-in
    needs_neo4j = not (
        getattr(args, "skip_index", False)
        and getattr(args, "skip_enrich", False)
        and getattr(args, "skip_synthesis", False)
    )
    if needs_neo4j:
        print("[GraphPipeline] Preflight: checking Neo4j...")
        try:
            from kg.neo4j_client import _get_driver
            driver = _get_driver()
            if driver is None:
                errors.append(
                    "Neo4j is not reachable at bolt://localhost:7687\n"
                    "    → Start the Neo4j Docker container before running the pipeline."
                )
            else:
                driver.close()
                print("[GraphPipeline] Preflight: Neo4j        ✓")
        except Exception as exc:
            errors.append(f"Neo4j check raised an exception: {exc}")

    # LLM server — always needed (indexer, enricher, synthesizer all call it)
    print("[GraphPipeline] Preflight: checking LLM server...")
    base = ollama_base_url()
    try:
        headers = ollama_headers(require_api_key=False)
        resp = requests.get(f"{base}/api/tags", headers=headers, timeout=10)
        if resp.ok:
            print(f"[GraphPipeline] Preflight: LLM server     ✓  ({base})")
        else:
            errors.append(
                f"LLM server returned HTTP {resp.status_code} ({base})\n"
                "    → Check OLLAMA_API_KEY and OLLAMA_BASE_URL."
            )
    except Exception as exc:
        errors.append(
            f"LLM server not reachable ({base}): {exc}\n"
            "    → Check OLLAMA_BASE_URL and network connectivity."
        )

    # Emit API key warning (non-fatal) if key is missing
    import os
    if not os.getenv("OLLAMA_API_KEY", "").strip():
        print("[GraphPipeline] Preflight: WARNING — OLLAMA_API_KEY not set; authenticated endpoints will fail.")

    if errors:
        lines = "\n".join(f"  ✗ {e}" for e in errors)
        raise SystemExit(
            f"\n[GraphPipeline] PREFLIGHT FAILED — fix the issues below and re-run:\n{lines}\n"
        )


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
    embed_backend: str | None = None,
    index_workers: int = 1,
    index_model: str | None = None,
    force_reindex: bool = False,
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

        if embed_backend:
            cmd.extend(["--embed-backend", embed_backend])

        cmd.extend(["--index-workers", str(index_workers)])
        if index_model:
            cmd.extend(["--index-model", index_model])
        if force_reindex:
            cmd.append("--force-reindex")

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
    parser.add_argument(
        "--shared-chroma-dir",
        default=None,
        help=(
            "Path to a persistent shared Chroma store reused across bundles. "
            "Documents already present are skipped (no re-download, no re-embedding). "
            "If omitted, a fresh per-bundle store is created (old behavior). "
            "Use a separate path per embedding backend to avoid mixing models."
        ),
    )
    parser.add_argument("--ingest-lock-file", default=".ingest.lock")
    parser.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--embed-backend",
        default=None,
        choices=["ollama", "openai"],
        help="Embedding backend for Chroma + uPheno + similarity (default: from EMBED_BACKEND env var or ollama).",
    )
    parser.add_argument(
        "--map-workers",
        type=int,
        default=1,
        help="Parallel threads for uPheno entity mapping (default: 1). Increase if LLM server handles concurrency well.",
    )
    parser.add_argument(
        "--index-workers",
        type=int,
        default=1,
        help="Parallel threads for chunk entity extraction during graph indexing (default: 1).",
    )
    parser.add_argument(
        "--index-model",
        default=None,
        help="LLM model for chunk entity extraction (default: qwen2.5:7b). "
             "Smaller/faster than the main synthesis model.",
    )
    parser.add_argument(
        "--mapping-model",
        default="granite4.1:8b",
        help="LLM model for uPheno entity normalization and Stage 3 verification "
             "(default: granite4.1:8b). Separate from the synthesis model so a "
             "smaller/faster model can be used without affecting output quality.",
    )
    parser.add_argument(
        "--norm-batch-size",
        type=int,
        default=1,
        help="Traits per Stage 1 normalization LLM call (default: 1 = one-by-one). "
             "Values > 1 batch multiple traits into a single call, e.g. --norm-batch-size 10.",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        default=False,
        help="Re-index all chunks even if already in Neo4j (use when fixing entity extraction).",
    )
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
    parser.add_argument(
        "--tier1-max-entity-forms",
        type=int,
        default=200,
        help="Hard safety-net cap: Tier 1 ancestors with more distinct entity forms than this "
             "are excluded (default: 200). IC scoring handles specificity below this threshold.",
    )
    parser.add_argument(
        "--no-summarize-subgraphs",
        action="store_true",
        default=False,
        help="Disable per-species subgraph summarization before synthesis (passes raw triples "
             "to synthesis LLM; may exceed context window for large runs).",
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

    # Shared Chroma store: reused across bundles to avoid re-downloading/re-embedding.
    # Priority: --shared-chroma-dir > --chroma-dir > per-bundle default.
    if args.shared_chroma_dir:
        chroma_dir = pathlib.Path(args.shared_chroma_dir)
        print(f"[GraphPipeline] Using shared Chroma store: {chroma_dir}")
    elif args.chroma_dir:
        chroma_dir = pathlib.Path(args.chroma_dir)
    else:
        chroma_dir = bundle_dir / "cache" / "chroma_store_ollama"

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

    _preflight_check(args)

    pipeline_start = time.monotonic()
    stage_timings: dict[str, float] = {}

    t0 = time.monotonic()
    run_dirs = run_inventory(
        species_file=species_file,
        bundle_dir=bundle_dir,
        runs=args.runs,
        skip_ingest_after_first=args.skip_ingest_after_first,
        pdf_dir=pdf_dir,
        chroma_dir=chroma_dir,
        ingest_lock_file=ingest_lock_file,
        llm_model=args.model,
        embed_backend=args.embed_backend,
        index_workers=args.index_workers,
        index_model=args.index_model,
        force_reindex=args.force_reindex,
    )
    stage_timings["indexing_s"] = round(time.monotonic() - t0, 1)

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
        t0 = time.monotonic()
        try:
            from kg.graph_enricher import enrich_doc_entities
            enrich_summary = enrich_doc_entities(
                species_norms=species_norms,
                similarity_threshold=args.similarity_threshold,
                model=args.model,
                mapping_model=args.mapping_model,
                force_reenrich=args.force_reenrich,
                log_dir=bundle_dir,
                embed_backend=args.embed_backend,
                max_workers=args.map_workers,
                norm_batch_size=args.norm_batch_size,
            )
            (bundle_dir / "graph_enricher_summary.json").write_text(
                json.dumps(enrich_summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            stage_timings["enrichment_s"] = round(time.monotonic() - t0, 1)
            print(f"[GraphPipeline] Enrichment complete: {enrich_summary.get('entities_mapped', 0)} mapped, "
                  f"{enrich_summary.get('similarity_pairs_added', 0)} similarity pairs. "
                  f"({stage_timings['enrichment_s']}s)")
        except Exception as exc:
            stage_timings["enrichment_s"] = round(time.monotonic() - t0, 1)
            print(f"[GraphPipeline] Enrichment failed (non-fatal): {exc}")
    elif args.skip_enrich:
        print(f"[GraphPipeline] Skipping enrichment step (--skip-enrich).")

    # Precompute global_fan_in on OntologyTerm nodes so Tier 1 IC-based ranking
    # has up-to-date scores before synthesis queries the graph.
    if not args.skip_enrich and not args.skip_synthesis and species_norms:
        print(f"\n[GraphPipeline] Updating global_fan_in on OntologyTerm nodes...")
        t0 = time.monotonic()
        try:
            from kg.precompute_fan_in import run as _run_fan_in
            fan_in_stats = _run_fan_in()
            stage_timings["fan_in_s"] = round(time.monotonic() - t0, 1)
            print(f"[GraphPipeline] Fan-in precompute complete. ({stage_timings['fan_in_s']}s)")
            (bundle_dir / "fan_in_precompute.json").write_text(
                json.dumps(fan_in_stats, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            stage_timings["fan_in_s"] = round(time.monotonic() - t0, 1)
            print(f"[GraphPipeline] precompute_fan_in failed (non-fatal, synthesis will use fallback ranking): {exc}")
            (bundle_dir / "fan_in_precompute.json").write_text(
                json.dumps({"status": "exception", "error": str(exc)}, indent=2), encoding="utf-8"
            )

    # Three-tier synthesis step
    if not args.skip_synthesis and species_norms:
        print(f"\n[GraphPipeline] ========================================")
        print(f"[GraphPipeline] Running three-tier synthesis (min_species={min_species})...")
        print(f"[GraphPipeline] ========================================")
        t0 = time.monotonic()
        try:
            from kg.graph_synthesizer import run_synthesis
            synthesis = run_synthesis(
                species_norms=species_norms,
                species_display_names=species_display,
                min_species=min_species,
                model=args.model,
                log_dir=bundle_dir,
                tier2_score_threshold=args.similarity_threshold,
                tier1_max_entity_forms=args.tier1_max_entity_forms,
                summarize_subgraphs=not args.no_summarize_subgraphs,
            )
            (bundle_dir / "graph_synthesis.json").write_text(
                json.dumps(synthesis, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            stage_timings["synthesis_s"] = round(time.monotonic() - t0, 1)
            n_communities = len(synthesis.get("communities", []))
            print(f"[GraphPipeline] Synthesis complete: {n_communities} communities found. "
                  f"({stage_timings['synthesis_s']}s)")
            if n_communities:
                for comm in synthesis["communities"]:
                    print(f"[GraphPipeline]   [{comm.get('tier','?')}] {comm.get('label','?')} "
                          f"({comm.get('species_count',0)} species)")
        except Exception as exc:
            stage_timings["synthesis_s"] = round(time.monotonic() - t0, 1)
            print(f"[GraphPipeline] Synthesis failed (non-fatal): {exc}")
    elif args.skip_synthesis:
        print(f"[GraphPipeline] Skipping synthesis step (--skip-synthesis).")

    stage_timings["total_s"] = round(time.monotonic() - pipeline_start, 1)

    print(f"\n[GraphPipeline] ══════════════════════════════════════════")
    print(f"[GraphPipeline]  Stage timing summary")
    print(f"[GraphPipeline] ══════════════════════════════════════════")
    for stage, secs in stage_timings.items():
        label = stage.replace("_s", "").replace("_", " ").ljust(16)
        bar = "█" * max(1, int(secs / max(stage_timings.values()) * 30))
        print(f"[GraphPipeline]  {label}  {bar}  {secs}s")
    print(f"[GraphPipeline] ══════════════════════════════════════════")

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
        "embed_backend": args.embed_backend or "ollama",
        "map_workers": args.map_workers,
        "norm_batch_size": args.norm_batch_size,
        "index_workers": args.index_workers,
        "index_model": args.index_model or "granite4.1:8b",
        "mapping_model": args.mapping_model,
        "shared_chroma_dir": str(chroma_dir) if args.shared_chroma_dir else None,
        "stage_timings": stage_timings,
    }
    (summary_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    total_str = f"{int(stage_timings['total_s'] // 60)}m {int(stage_timings['total_s'] % 60)}s"
    print(f"[GraphPipeline] Done in {total_str}. Bundle: {bundle_dir}")


if __name__ == "__main__":
    main()
