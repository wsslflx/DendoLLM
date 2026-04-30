#!/usr/bin/env python3
"""
GraphRAG per-species inventory — replaces inventory_single_2/4 in the graph pipeline.

Flow per species:
  1. Ingest docs into Chroma (reuses inventory_single_2.ingest_species)
  2. Run LLM entity extraction on all chunks → push DocGraph to Neo4j (graph_indexer)
  3. Traverse DocGraph → collect subgraph + chunk texts (graph_retriever)
  4. LLM reads structured subgraph context → extracts traits (prompt_graph_inventory.txt)
  5. Hybrid normalization (build_hybrid_species_profile, unchanged)
  6. Write hybrid_profile.json and trait cache

Output is backwards-compatible with the old pipeline:
  Each trait has: trait, sources, confidence (old fields) +
                  supporting_triples, source_chunk_ids (new fields, ignored by old steps)

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
import re
from contextlib import nullcontext
from datetime import datetime
from typing import Any

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

import inventory_single_2 as v1
from core.hybrid_normalization import build_hybrid_species_profile
from core.llm_backend import make_chat_llm
from kg.graph_indexer import index_species
from kg.graph_retriever import retrieve_subgraph, serialize_subgraph_to_context

GRAPH_INVENTORY_PROMPT_FILE = "Prompts/prompt_graph_inventory.txt"
MAX_TRAIT_RETRIES = 2


# ---------------------------------------------------------------------------
# Helpers reused from v1 / v4
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


def _extract_json_text(payload: object) -> str:
    text = str(payload).strip()
    if text.startswith("```"):
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
        if m:
            text = m.group(1).strip()
    return text


# ---------------------------------------------------------------------------
# Trait validation
# ---------------------------------------------------------------------------

def _validate_graph_traits(data: object) -> list[dict]:
    """
    Validate and clean the LLM's trait extraction output.
    Accepts the richer format (supporting_triples, source_chunk_ids) while
    remaining compatible with the old format (trait, sources, confidence).
    """
    if not isinstance(data, list):
        raise ValueError("Graph inventory output must be a JSON array")

    cleaned: list[dict] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        trait = item.get("trait")
        if not isinstance(trait, str) or not trait.strip():
            continue

        sources = item.get("sources", [])
        if not isinstance(sources, list):
            sources = []
        clean_sources = [str(s) for s in sources if isinstance(s, str) and s.strip()]

        try:
            conf = float(item.get("confidence", 0.7))
            conf = max(0.0, min(1.0, conf))
        except (TypeError, ValueError):
            conf = 0.7

        # New fields — passthrough with type checking
        supporting_triples = item.get("supporting_triples", [])
        if not isinstance(supporting_triples, list):
            supporting_triples = []
        clean_triples = []
        for t in supporting_triples:
            if isinstance(t, list) and len(t) == 3:
                clean_triples.append([str(t[0]), str(t[1]), str(t[2])])

        source_chunk_ids = item.get("source_chunk_ids", [])
        if not isinstance(source_chunk_ids, list):
            source_chunk_ids = []
        clean_chunk_ids = [str(c) for c in source_chunk_ids if c]

        cleaned.append({
            "trait": " ".join(trait.split()).strip(),
            "sources": sorted(set(clean_sources)),
            "confidence": conf,
            "supporting_triples": clean_triples,
            "source_chunk_ids": clean_chunk_ids,
        })

    return cleaned


# ---------------------------------------------------------------------------
# Core per-species function
# ---------------------------------------------------------------------------

def run_graph_inventory(
    specie: str,
    aliases: list[str],
    log_runs: bool = False,
    reuse_traits: bool = False,
    log_root: pathlib.Path | None = None,
    skip_ingest: bool = False,
    skip_index: bool = False,
    traits_dir: pathlib.Path | None = None,
    pdf_dir: pathlib.Path | None = None,
    chroma_dir: pathlib.Path | None = None,
    ingest_lock_file: pathlib.Path | None = None,
    llm_model: str | None = None,
    temperature: float = 0.0,
    force_reindex: bool = False,
) -> list[dict]:
    """
    Run the GraphRAG inventory for one species.
    Returns list of trait dicts (backwards-compatible + richer fields).
    """
    traits_root = traits_dir if traits_dir else pathlib.Path("traits")
    traits_root.mkdir(parents=True, exist_ok=True)
    out_path = traits_root / f"{v1.slugify(specie)}.json"

    # Trait cache shortcut
    if reuse_traits and out_path.exists():
        traits = json.loads(out_path.read_text(encoding="utf-8"))
        print(json.dumps(traits, ensure_ascii=False, indent=2))
        return traits if isinstance(traits, list) else []

    species_norm = specie.lower().strip()

    # --- Step 1: Ingest ---
    from core.rag_cli import RAG
    rag = RAG(
        log_runs=log_runs,
        persist_dir=str(chroma_dir) if chroma_dir else "./chroma_store_ollama",
    )
    if not skip_ingest:
        lock_ctx = v1.ingest_lock(ingest_lock_file) if ingest_lock_file else nullcontext()
        with lock_ctx:
            print(f"[GraphRAG] Ingesting docs for '{specie}'...")
            v1.ingest_species(rag, canonical=specie, aliases=aliases, pdf_dir=pdf_dir)

    log_dir = init_log_dir(log_runs, log_root=log_root, subdir=v1.slugify(specie))

    # Log ingested docs
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
            temperature=temperature,
            force_reindex=force_reindex,
            log_dir=log_dir,
        )
    else:
        print(f"[GraphRAG] Skipping graph indexing for '{specie}' (--skip-index).")

    # --- Step 3: Graph retrieval ---
    print(f"[GraphRAG] Retrieving subgraph for '{specie}'...")
    subgraph = retrieve_subgraph(species_norm)

    if subgraph.n_chunks == 0:
        print(f"[GraphRAG] No indexed chunks found for '{specie}' — returning empty traits.")
        out_path.write_text("[]", encoding="utf-8")
        return []

    graph_context = serialize_subgraph_to_context(subgraph)

    if log_dir:
        (log_dir / "graph_context.txt").write_text(graph_context, encoding="utf-8")
        if index_summary:
            (log_dir / "graph_indexer_summary.json").write_text(
                json.dumps(index_summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    # --- Step 4: LLM trait extraction ---
    print(f"[GraphRAG] Running LLM trait extraction for '{specie}'...")
    try:
        prompt_template_text = pathlib.Path(GRAPH_INVENTORY_PROMPT_FILE).read_text(encoding="utf-8")
    except Exception as exc:
        print(f"[GraphRAG] Cannot load prompt {GRAPH_INVENTORY_PROMPT_FILE}: {exc}")
        out_path.write_text("[]", encoding="utf-8")
        return []

    prompt = PromptTemplate(
        input_variables=["species_name", "graph_context"],
        template=prompt_template_text,
    )
    formatted_prompt = prompt.format(species_name=specie, graph_context=graph_context)

    if log_dir:
        (log_dir / "prompt.txt").write_text(formatted_prompt, encoding="utf-8")

    llm = make_chat_llm(model=llm_model, temperature=temperature)
    chain = (
        {
            "species_name": RunnableLambda(lambda _: specie),
            "graph_context": RunnableLambda(lambda _: graph_context),
        }
        | prompt
        | llm
    )

    traits: list[dict] = []
    raw_answer = ""
    for attempt in range(1, MAX_TRAIT_RETRIES + 2):
        raw = chain.invoke({})
        payload = raw.content if hasattr(raw, "content") else raw
        raw_answer = str(payload)
        try:
            parsed = json.loads(_extract_json_text(payload))
            traits = _validate_graph_traits(parsed)
            break
        except Exception as exc:
            print(f"[GraphRAG] Trait parse attempt {attempt} failed: {exc}")
            if attempt == MAX_TRAIT_RETRIES + 1:
                print(f"[GraphRAG] All attempts failed — returning empty traits for '{specie}'.")
                traits = []

    if log_dir:
        (log_dir / "answer.txt").write_text(raw_answer, encoding="utf-8")

    # Write trait cache
    out_path.write_text(json.dumps(traits, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(traits, ensure_ascii=False, indent=2))
    return traits


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GraphRAG per-species inventory (graph-indexed retrieval + trait extraction)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--species", help="Canonical species name to process.")
    group.add_argument("--species-file", help="Path to JSON file with canonical/aliases mappings.")
    parser.add_argument("--aliases", help="Comma-separated aliases.")
    parser.add_argument("--log-run", action="store_true", help="Log artifacts to logs_graph/.")
    parser.add_argument("--log-dir", help="Explicit directory for run logs.")
    parser.add_argument("--traits-dir", help="Directory for per-species trait cache files.")
    parser.add_argument("--pdf-dir", help="Directory for downloaded PDFs.")
    parser.add_argument("--chroma-dir", help="Directory for persisted Chroma vectorstore.")
    parser.add_argument("--ingest-lock-file", default=".ingest.lock")
    parser.add_argument("--reuse-traits", action="store_true")
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-index", action="store_true", help="Skip graph indexing step.")
    parser.add_argument("--force-reindex", action="store_true", help="Re-index all chunks even if already indexed.")
    parser.add_argument("--model", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--hybrid-sim-threshold",
        type=float,
        default=0.82,
        help="Cosine similarity threshold for hybrid normalization (default: 0.82).",
    )
    args = parser.parse_args()

    explicit_log_dir = pathlib.Path(args.log_dir) if args.log_dir else None
    traits_dir = pathlib.Path(args.traits_dir) if args.traits_dir else None
    pdf_dir = pathlib.Path(args.pdf_dir) if args.pdf_dir else None
    chroma_dir = pathlib.Path(args.chroma_dir) if args.chroma_dir else None
    ingest_lock_file = pathlib.Path(args.ingest_lock_file) if args.ingest_lock_file else None

    run_log_dir = None
    if args.log_run:
        run_label = pathlib.Path(args.species_file).stem if args.species_file else (args.species or "").strip()
        run_log_dir = init_run_log_dir(True, run_label, explicit_log_dir=explicit_log_dir)

    if args.species_file:
        species_groups = v1.load_species_file(args.species_file)
        species_profiles: dict[str, dict[str, Any]] = {}
        for entry in species_groups:
            canonical = (entry.get("canonical") or "").strip()
            aliases = [a.strip() for a in entry.get("aliases", []) if a.strip()]
            if not canonical:
                continue
            open_traits = run_graph_inventory(
                canonical,
                aliases,
                log_runs=args.log_run,
                reuse_traits=args.reuse_traits,
                log_root=run_log_dir,
                skip_ingest=args.skip_ingest,
                skip_index=args.skip_index,
                traits_dir=traits_dir,
                pdf_dir=pdf_dir,
                chroma_dir=chroma_dir,
                ingest_lock_file=ingest_lock_file,
                llm_model=args.model,
                temperature=args.temperature,
                force_reindex=args.force_reindex,
            )
            profile = build_hybrid_species_profile(open_traits, similarity_threshold=args.hybrid_sim_threshold)
            species_profiles[canonical] = profile

            if run_log_dir is not None:
                species_log_dir = run_log_dir / v1.slugify(canonical)
                species_log_dir.mkdir(parents=True, exist_ok=True)
                (species_log_dir / "hybrid_profile.json").write_text(
                    json.dumps(profile, ensure_ascii=False, indent=2), encoding="utf-8"
                )

        print(json.dumps(species_profiles, ensure_ascii=False, indent=2))
        return

    aliases = [a.strip() for a in (args.aliases or "").split(",") if a.strip()]
    open_traits = run_graph_inventory(
        args.species.strip(),
        aliases,
        log_runs=args.log_run,
        reuse_traits=args.reuse_traits,
        log_root=run_log_dir,
        skip_ingest=args.skip_ingest,
        skip_index=args.skip_index,
        traits_dir=traits_dir,
        pdf_dir=pdf_dir,
        chroma_dir=chroma_dir,
        ingest_lock_file=ingest_lock_file,
        llm_model=args.model,
        temperature=args.temperature,
        force_reindex=args.force_reindex,
    )
    profile = build_hybrid_species_profile(open_traits, similarity_threshold=args.hybrid_sim_threshold)
    print(json.dumps(profile, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
