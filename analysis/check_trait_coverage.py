#!/usr/bin/env python3
"""
Evidence collector: did a concept make it into the LLM context for a species?

For a given bundle, species, and concept phrase, this script:
  1. Searches the Chroma store (species-filtered) for the concept
  2. Cross-references results against used_chunks.json
  3. Prints each hit with its distance score, relevance flag, and text snippet

You read the output and judge whether the chunks are actually relevant.

Usage:
    python analysis/check_trait_coverage.py \\
        --bundle-dir logs_v4/<bundle> \\
        --species "condylura cristata" \\
        --concept "subterranean fossorial tunnel burrow" \\
        [--k 10] \\
        [--run run_01]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


def slugify(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def load_used_chunks(bundle_dir: Path, species: str, run: str) -> set[tuple[str, int]]:
    slug = slugify(species)
    path = bundle_dir / "runs" / run / slug / "used_chunks.json"
    if not path.exists():
        print(f"[!] used_chunks.json not found: {path}")
        return set()
    try:
        chunks = json.loads(path.read_text(encoding="utf-8"))
        return {(c["doc_id"], c["chunk_index"]) for c in chunks if "doc_id" in c and "chunk_index" in c}
    except Exception as exc:
        print(f"[!] Failed to load used_chunks.json: {exc}")
        return set()


def find_runs(bundle_dir: Path) -> list[str]:
    runs_dir = bundle_dir / "runs"
    if not runs_dir.exists():
        return []
    return sorted(p.name for p in runs_dir.iterdir() if p.is_dir())


def main() -> None:
    parser = argparse.ArgumentParser(description="Evidence collector for trait coverage diagnostics.")
    parser.add_argument("--bundle-dir", required=True, help="Path to pipeline bundle directory.")
    parser.add_argument("--species", required=True, help="Species name as stored in Chroma (lowercase).")
    parser.add_argument("--concept", required=True, help="Concept phrase to search for (e.g. 'subterranean fossorial burrow').")
    parser.add_argument("--k", type=int, default=10, help="Number of nearest chunks to retrieve (default: 10).")
    parser.add_argument("--run", default=None, help="Run directory name (e.g. run_01). Defaults to first run found.")
    args = parser.parse_args()

    bundle_dir = Path(args.bundle_dir)
    chroma_dir = bundle_dir / "cache" / "chroma_store_ollama"

    if not chroma_dir.exists():
        print(f"[!] Chroma store not found: {chroma_dir}")
        sys.exit(1)

    # Resolve run
    run = args.run
    if run is None:
        runs = find_runs(bundle_dir)
        if not runs:
            print(f"[!] No runs found in {bundle_dir / 'runs'}")
            sys.exit(1)
        run = runs[0]
        print(f"[i] Using run: {run}")

    # Load used chunks for this species/run
    used = load_used_chunks(bundle_dir, args.species, run)
    print(f"[i] used_chunks.json: {len(used)} chunks reached the LLM")

    # Load Chroma and search
    from core.rag_cli import RAG
    rag = RAG(persist_dir=str(chroma_dir))

    print(f"\n[i] Searching for: '{args.concept}'")
    print(f"[i] Species filter: '{args.species}'")
    print(f"[i] k={args.k}")
    print()

    results = rag.vectorstore.similarity_search_with_score(
        args.concept,
        k=args.k,
        filter={"specie": args.species},
    )

    if not results:
        print("NO CHUNKS FOUND for this species in Chroma.")
        print("=> Verdict hint: ingestion gap — source material not in store.")
        return

    print(f"{'#':<3}  {'SCORE':<7}  {'IN LLM':<8}  {'DOC_ID':<55}  SNIPPET")
    print("-" * 130)
    for i, (doc, score) in enumerate(results, 1):
        meta = doc.metadata or {}
        doc_id = meta.get("doc_id", "?")
        chunk_idx = meta.get("chunk_index", -1)
        in_llm = (doc_id, chunk_idx) in used
        snippet = doc.page_content.replace("\n", " ").strip()[:120]
        flag = "YES" if in_llm else "no"
        print(f"{i:<3}  {score:<7.4f}  {flag:<8}  {doc_id:<55}  {snippet}")

    print()
    hits_in_llm = [(doc, score) for doc, score in results if (doc.metadata.get("doc_id"), doc.metadata.get("chunk_index")) in used]
    print(f"Summary: {len(results)} chunks retrieved, {len(hits_in_llm)} reached the LLM.")
    print()
    print("Verdict guide (YOU decide based on snippet relevance above):")
    print("  - Snippets irrelevant        => ingestion gap / wrong papers fetched")
    print("  - Snippets relevant, IN LLM  => LLM extraction failure")
    print("  - Snippets relevant, not LLM => filtered out before LLM (MMR/percentile)")


if __name__ == "__main__":
    main()
