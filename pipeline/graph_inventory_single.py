#!/usr/bin/env python3
"""
GraphRAG per-species inventory — ingest + graph index only.

Flow per species:
  1. Ingest docs into Chroma (reuses inventory_single_2.ingest_species)
  2. Run LLM entity extraction on all chunks → push DocGraph to Neo4j (graph_indexer)

Trait extraction and hybrid normalization have been moved downstream:
the three-tier synthesis step (graph_synthesizer) reads directly from the
enriched Neo4j document graph, not from per-species trait files.

Usage:
    python pipeline/graph_inventory_single.py --species-file testcase1.json --log-run
    python pipeline/graph_inventory_single.py --species "talpa europaea" --log-run
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parents[1]))
sys.path.insert(0, str(_Path(__file__).parent))

import argparse
import json
import pathlib
from contextlib import nullcontext
from datetime import datetime
from typing import Any

import inventory_single_2 as v1
from kg.graph_indexer import index_species


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def init_log_dir(
    log_runs: bool,
    log_root: pathlib.Path | None,
    subdir: str | None = None,
) -> pathlib.Path | None:
    if not log_runs:
        return None
    base = pathlib.Path(log_root) if log_root else pathlib.Path("logs_graph")
    log_dir = base / subdir if subdir else base
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def init_run_log_dir(
    log_runs: bool,
    run_label: str,
    explicit_log_dir: pathlib.Path | None = None,
) -> pathlib.Path | None:
    if not log_runs:
        return None
    if explicit_log_dir is not None:
        explicit_log_dir.mkdir(parents=True, exist_ok=True)
        return explicit_log_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pathlib.Path("logs_graph") / f"{timestamp}-{v1.slugify(run_label)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


# ---------------------------------------------------------------------------
# Core per-species function
# ---------------------------------------------------------------------------

def run_graph_inventory(
    specie: str,
    aliases: list[str],
    log_runs: bool = False,
    log_root: pathlib.Path | None = None,
    skip_ingest: bool = False,
    skip_index: bool = False,
    pdf_dir: pathlib.Path | None = None,
    chroma_dir: pathlib.Path | None = None,
    ingest_lock_file: pathlib.Path | None = None,
    llm_model: str | None = None,
    index_model: str | None = None,
    index_workers: int = 1,
    temperature: float = 0.0,
    force_reindex: bool = False,
    embed_backend: str | None = None,
) -> dict:
    """
    Ingest documents and index them into the Neo4j document graph.
    Returns index_summary dict from index_species().
    """
    species_norm = specie.lower().strip()

    # --- Step 1: Ingest ---
    from core.rag_cli import RAG
    rag = RAG(
        log_runs=log_runs,
        persist_dir=str(chroma_dir) if chroma_dir else "./chroma_store_ollama",
        embed_backend=embed_backend,
    )
    if not skip_ingest:
        lock_ctx = v1.ingest_lock(ingest_lock_file) if ingest_lock_file else nullcontext()
        with lock_ctx:
            print(f"[GraphRAG] Ingesting docs for '{specie}'...")
            v1.ingest_species(rag, canonical=specie, aliases=aliases, pdf_dir=pdf_dir)

    log_dir = init_log_dir(log_runs, log_root=log_root, subdir=v1.slugify(specie))

    ingested_docs = sorted(rag.ingested_per_species.get(species_norm, set()))
    if log_dir:
        (log_dir / "ingested_docs.json").write_text(
            json.dumps(ingested_docs, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # --- Step 2: Graph indexing ---
    index_summary: dict[str, Any] = {}
    if not skip_index:
        print(f"[GraphRAG] Indexing chunks for '{specie}' into Neo4j document graph...")
        index_summary = index_species(
            species_norm=species_norm,
            chroma_vectorstore=rag.vectorstore,
            llm_model=llm_model,
            index_llm_model=index_model,
            temperature=temperature,
            force_reindex=force_reindex,
            log_dir=log_dir,
            max_workers=index_workers,
        )
        if log_dir and index_summary:
            (log_dir / "graph_indexer_summary.json").write_text(
                json.dumps(index_summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
    else:
        print(f"[GraphRAG] Skipping graph indexing for '{specie}' (--skip-index).")

    return index_summary


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GraphRAG per-species inventory (ingest + graph indexing)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--species", help="Canonical species name to process.")
    group.add_argument("--species-file", help="Path to JSON file with canonical/aliases mappings.")
    parser.add_argument("--aliases", help="Comma-separated aliases (single species mode only).")
    parser.add_argument("--log-run", action="store_true", help="Log artifacts to logs_graph/.")
    parser.add_argument("--log-dir", help="Explicit directory for run logs.")
    parser.add_argument("--pdf-dir", help="Directory for downloaded PDFs.")
    parser.add_argument("--chroma-dir", help="Directory for persisted Chroma vectorstore.")
    parser.add_argument("--ingest-lock-file", default=".ingest.lock")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-index", action="store_true", help="Skip graph indexing step.")
    parser.add_argument("--force-reindex", action="store_true",
                        help="Re-index all chunks even if already indexed.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--index-model", default=None,
                        help="LLM model for chunk entity extraction (default: qwen2.5:7b).")
    parser.add_argument("--index-workers", type=int, default=1,
                        help="Parallel threads for chunk indexing (default: 1).")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--embed-backend",
        default=None,
        choices=["ollama", "openai"],
        help="Embedding backend to use (default: from EMBED_BACKEND env var or ollama).",
    )
    args = parser.parse_args()

    explicit_log_dir = pathlib.Path(args.log_dir) if args.log_dir else None
    pdf_dir = pathlib.Path(args.pdf_dir) if args.pdf_dir else None
    chroma_dir = pathlib.Path(args.chroma_dir) if args.chroma_dir else None
    ingest_lock_file = pathlib.Path(args.ingest_lock_file) if args.ingest_lock_file else None

    run_log_dir = None
    if args.log_run:
        run_label = pathlib.Path(args.species_file).stem if args.species_file else (args.species or "").strip()
        run_log_dir = init_run_log_dir(True, run_label, explicit_log_dir=explicit_log_dir)

    if args.species_file:
        species_groups = v1.load_species_file(args.species_file)
        summaries: dict[str, dict] = {}
        for entry in species_groups:
            canonical = (entry.get("canonical") or "").strip()
            aliases = [a.strip() for a in entry.get("aliases", []) if a.strip()]
            if not canonical:
                continue
            summary = run_graph_inventory(
                canonical,
                aliases,
                log_runs=args.log_run,
                log_root=run_log_dir,
                skip_ingest=args.skip_ingest,
                skip_index=args.skip_index,
                pdf_dir=pdf_dir,
                chroma_dir=chroma_dir,
                ingest_lock_file=ingest_lock_file,
                llm_model=args.model,
                index_model=args.index_model,
                index_workers=args.index_workers,
                temperature=args.temperature,
                force_reindex=args.force_reindex,
                embed_backend=args.embed_backend,
            )
            summaries[canonical] = summary

        print(json.dumps(summaries, ensure_ascii=False, indent=2))
        return

    aliases = [a.strip() for a in (args.aliases or "").split(",") if a.strip()]
    summary = run_graph_inventory(
        args.species.strip(),
        aliases,
        log_runs=args.log_run,
        log_root=run_log_dir,
        skip_ingest=args.skip_ingest,
        skip_index=args.skip_index,
        pdf_dir=pdf_dir,
        chroma_dir=chroma_dir,
        ingest_lock_file=ingest_lock_file,
        llm_model=args.model,
        index_model=args.index_model,
        index_workers=args.index_workers,
        temperature=args.temperature,
        force_reindex=args.force_reindex,
        embed_backend=args.embed_backend,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
