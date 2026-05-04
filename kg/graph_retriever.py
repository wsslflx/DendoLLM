#!/usr/bin/env python3
"""
Graph Retriever — Step 2 of the GraphRAG pipeline.

Given a species name, traverses the Neo4j document graph (DocChunk + DocEntity)
built by graph_indexer.py and returns a structured SubgraphResult containing:
  - All entities mentioned in the species' chunks
  - All RELATED_TO edges between those entities
  - Source chunk IDs and text snippets

serialize_subgraph_to_context() converts the result into a structured string
ready to pass as context to the LLM trait extraction prompt.

Usage (standalone):
    python kg/graph_retriever.py --species "talpa europaea"
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class SubgraphResult:
    species_norm: str
    entities: list[dict]               # [{entity_id, surface_form, entity_type, mention_count}]
    triples: list[tuple[str, str, str]]  # [(subj_surface, relation_type, obj_surface)]
    source_chunk_ids: list[str]
    chunk_texts: dict[str, str]        # chunk_id → text_snippet
    chunk_sources: dict[str, str]      # chunk_id → source_path
    n_chunks: int
    n_entities: int
    n_triples: int


_EMPTY = SubgraphResult(
    species_norm="",
    entities=[],
    triples=[],
    source_chunk_ids=[],
    chunk_texts={},
    chunk_sources={},
    n_chunks=0,
    n_entities=0,
    n_triples=0,
)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def retrieve_subgraph(
    species_norm: str,
    max_chunks: int = 200,
    entity_types: list[str] | None = None,
) -> SubgraphResult:
    """
    Traverse Neo4j document graph for species_norm.
    Returns SubgraphResult (empty if Neo4j unreachable or no indexed data).
    """
    from kg.neo4j_client import run_query

    # Q1: all DocChunk nodes for this species
    q1 = (
        "MATCH (c:DocChunk {species_norm: $sn}) "
        "RETURN c.chunk_id AS chunk_id, c.text_snippet AS snippet, "
        "       c.source_path AS source_path, c.doc_id AS doc_id "
        "LIMIT $max_c"
    )
    chunk_rows = run_query(q1, {"sn": species_norm, "max_c": max_chunks})

    if not chunk_rows:
        print(f"[GraphRetriever] No DocChunk nodes found for '{species_norm}'.")
        result = dataclasses.replace(_EMPTY, species_norm=species_norm)
        return result

    chunk_texts: dict[str, str] = {}
    chunk_sources: dict[str, str] = {}
    for row in chunk_rows:
        cid = row.get("chunk_id", "")
        if cid:
            chunk_texts[cid] = row.get("snippet", "") or ""
            chunk_sources[cid] = row.get("source_path", "") or ""

    # Q2: all entities mentioned in those chunks
    entity_filter = ""
    q2_params: dict = {"sn": species_norm}
    if entity_types:
        entity_filter = "WHERE e.entity_type IN $etypes "
        q2_params["etypes"] = entity_types

    q2 = (
        "MATCH (c:DocChunk {species_norm: $sn})-[:MENTIONS]->(e:DocEntity) "
        + entity_filter +
        "RETURN e.entity_id AS eid, e.surface_form AS sf, e.entity_type AS etype, "
        "       count(c) AS mention_count "
        "ORDER BY mention_count DESC"
    )
    entity_rows = run_query(q2, q2_params)

    entities: list[dict] = []
    entity_surface: dict[str, str] = {}  # entity_id → surface_form
    for row in entity_rows:
        eid = row.get("eid", "")
        sf = row.get("sf", "")
        if eid and sf:
            entities.append({
                "entity_id": eid,
                "surface_form": sf,
                "entity_type": row.get("etype", ""),
                "mention_count": row.get("mention_count", 1),
            })
            entity_surface[eid] = sf

    # Q3: all RELATED_TO edges between entities in those chunks
    q3 = (
        "MATCH (c:DocChunk {species_norm: $sn})-[:MENTIONS]->(e1:DocEntity) "
        "      -[r:RELATED_TO]->(e2:DocEntity) "
        "RETURN DISTINCT e1.surface_form AS subj, r.relation_type AS rel, "
        "       e2.surface_form AS obj "
        "LIMIT 5000"
    )
    triple_rows = run_query(q3, {"sn": species_norm})

    seen_triples: set[tuple[str, str, str]] = set()
    triples: list[tuple[str, str, str]] = []
    for row in triple_rows:
        subj = row.get("subj", "")
        rel = row.get("rel", "")
        obj = row.get("obj", "")
        if subj and rel and obj:
            t = (subj, rel, obj)
            if t not in seen_triples:
                seen_triples.add(t)
                triples.append(t)

    source_chunk_ids = list(chunk_texts.keys())

    print(
        f"[GraphRetriever] '{species_norm}': "
        f"{len(source_chunk_ids)} chunks, {len(entities)} entities, {len(triples)} triples."
    )

    return SubgraphResult(
        species_norm=species_norm,
        entities=entities,
        triples=triples,
        source_chunk_ids=source_chunk_ids,
        chunk_texts=chunk_texts,
        chunk_sources=chunk_sources,
        n_chunks=len(source_chunk_ids),
        n_entities=len(entities),
        n_triples=len(triples),
    )


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------

def serialize_subgraph_to_context(
    result: SubgraphResult,
    max_chunks_in_context: int = 40,
    include_source_chunks: bool = True,
    max_triples_in_context: int = 0,
) -> str:
    """
    Convert SubgraphResult to a structured string for the LLM prompt.
    Produces up to three sections: ENTITY SUMMARY, KNOWLEDGE GRAPH TRIPLES, SOURCE CHUNKS.

    include_source_chunks: if False, the SOURCE CHUNKS section is omitted entirely.
                           Set to False for synthesis context to save tokens.
    max_triples_in_context: if > 0, truncate triples to this many. 0 = no limit.
    """
    lines: list[str] = []

    # --- Entity Summary ---
    lines.append("=== ENTITY SUMMARY ===")
    if not result.entities:
        lines.append("(no entities extracted)")
    else:
        # Group by type for readability
        by_type: dict[str, list[dict]] = defaultdict(list)
        for e in result.entities:
            by_type[e["entity_type"]].append(e)

        type_order = ["Species", "Phenotype", "Anatomy", "Process", "Habitat", "Gene"]
        shown_types = type_order + [t for t in by_type if t not in type_order]
        for etype in shown_types:
            if etype not in by_type:
                continue
            for e in by_type[etype]:
                mc = e.get("mention_count", 1)
                lines.append(f"  [{etype}] {e['surface_form']}  (×{mc} chunks)")

    lines.append("")

    # --- Knowledge Graph Triples ---
    lines.append("=== KNOWLEDGE GRAPH TRIPLES ===")
    if not result.triples:
        lines.append("(no triples extracted)")
    else:
        triples = result.triples
        if max_triples_in_context > 0:
            triples = triples[:max_triples_in_context]
        for subj, rel, obj in triples:
            lines.append(f"  ({subj}) -[{rel}]-> ({obj})")
    lines.append("")

    # --- Source Chunks ---
    if include_source_chunks:
        lines.append("=== SOURCE CHUNKS ===")
        chunk_ids_to_show = result.source_chunk_ids[:max_chunks_in_context]
        if not chunk_ids_to_show:
            lines.append("(no source chunks)")
        else:
            for cid in chunk_ids_to_show:
                source = result.chunk_sources.get(cid, "unknown")
                snippet = result.chunk_texts.get(cid, "")
                lines.append(f"[chunk:{cid} | source:{source}]")
                if snippet:
                    lines.append(snippet)
                lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve and display document subgraph for a species.")
    parser.add_argument("--species", required=True, help="Species name (e.g. 'talpa europaea').")
    parser.add_argument("--max-chunks", type=int, default=200)
    parser.add_argument("--show-context", action="store_true", help="Print serialized context string.")
    args = parser.parse_args()

    result = retrieve_subgraph(args.species.lower().strip(), max_chunks=args.max_chunks)
    print(f"Chunks:   {result.n_chunks}")
    print(f"Entities: {result.n_entities}")
    print(f"Triples:  {result.n_triples}")

    if args.show_context:
        print("\n" + "=" * 60)
        print(serialize_subgraph_to_context(result))
