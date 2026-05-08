#!/usr/bin/env python3
"""
Graph Enricher — Step 2 of the GraphRAG pipeline.

After graph_indexer.py has built DocChunk/DocEntity nodes, this step:
1. Maps every DocEntity node to uPheno/HP/MP/ENVO ontology terms
   → adds (DocEntity)-[:DOC_MAPPED_TO]->(OntologyTerm) edges
   → adds (OntologyTerm)-[:IS_A]->(OntologyTerm) ancestor edges (up to depth 3)
2. Computes embedding cosine similarity between all DocEntity nodes across species
   → adds (DocEntity)-[:DOC_SIMILAR_TO {score}]->(DocEntity) edges for pairs above threshold

Run once per bundle after all species have been indexed. Idempotent:
- uPheno mapping: skips entities with upheno_enriched=True unless --force-reenrich
- Similarity: MERGE prevents duplicate edges

Usage (standalone):
    python kg/graph_enricher.py --species "talpa europaea,nannospalax ehrenbergi"
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_ENRICHER_SCHEMA = [
    "CREATE INDEX IF NOT EXISTS FOR (e:DocEntity) ON (e.upheno_enriched)",
]


def _init_enricher_schema(driver) -> None:
    with driver.session() as session:
        for q in _ENRICHER_SCHEMA:
            try:
                session.run(q)
            except Exception as exc:
                print(f"[GraphEnricher] Schema init warning (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Sub-step 1: uPheno mapping
# ---------------------------------------------------------------------------

def _pull_entities_to_map(driver, species_norms: list[str], force: bool) -> list[dict]:
    """Pull DocEntity nodes that need uPheno mapping."""
    if force:
        condition = ""
    else:
        condition = "AND (e.upheno_enriched IS NULL OR e.upheno_enriched = false) "
    cypher = (
        "MATCH (c:DocChunk)-[:MENTIONS]->(e:DocEntity) "
        f"WHERE c.species_norm IN $sn {condition}"
        "RETURN DISTINCT e.entity_id AS eid, e.surface_form AS sf, e.entity_type AS etype"
    )
    try:
        from kg.neo4j_client import run_query
        rows = run_query(cypher, {"sn": species_norms})
        return rows
    except Exception as exc:
        print(f"[GraphEnricher] Pull entities failed: {exc}")
        return []


def _push_mapping(driver, entity_id: str, map_result: dict, world, embed_model: str = "") -> int:
    """
    Push DOC_MAPPED_TO edge + IS_A ancestor edges for one entity.
    Returns number of ancestor terms added.
    """
    ancestors_added = 0
    term_id = map_result.get("term_id")
    term_name = map_result.get("term_name", "")
    cosine = map_result.get("cosine_score") or 0.0
    confidence = map_result.get("confidence", "no_match")
    broadened = map_result.get("broadened", False)

    if not term_id:
        # Mark as enriched (no match) to skip on reruns
        try:
            with driver.session() as session:
                session.run(
                    "MATCH (e:DocEntity {entity_id: $eid}) "
                    "SET e.upheno_enriched = true, e.upheno_mapped = false",
                    eid=entity_id,
                )
        except Exception as exc:
            print(f"[GraphEnricher] Mark no-match failed for {entity_id}: {exc}")
        return 0

    ontology_source = term_id.split(":")[0] if ":" in term_id else "UNKNOWN"

    try:
        with driver.session() as session:
            # MERGE OntologyTerm node
            session.run(
                "MERGE (t:OntologyTerm {term_id: $tid}) "
                "SET t.term_name = $tname, t.ontology_source = $src",
                tid=term_id, tname=term_name, src=ontology_source,
            )
            # MERGE DOC_MAPPED_TO edge
            session.run(
                "MATCH (e:DocEntity {entity_id: $eid}), (t:OntologyTerm {term_id: $tid}) "
                "MERGE (e)-[r:DOC_MAPPED_TO]->(t) "
                "SET r.cosine_score = $score, r.confidence = $conf, r.broadened = $broad, r.embed_model = $em",
                eid=entity_id, tid=term_id,
                score=cosine, conf=confidence, broad=broadened, em=embed_model,
            )
            # Mark entity as enriched
            session.run(
                "MATCH (e:DocEntity {entity_id: $eid}) "
                "SET e.upheno_enriched = true, e.upheno_mapped = true",
                eid=entity_id,
            )
    except Exception as exc:
        print(f"[GraphEnricher] Push DOC_MAPPED_TO failed for {entity_id} → {term_id}: {exc}")
        return 0

    # IS_A ancestor traversal
    if world is not None:
        from kg.kg_builder import _load_ancestors, _is_allowed
        ancestors = _load_ancestors(term_id, world, depth=3)
        seen: set[str] = set()
        for anc_id, rel_type in ancestors:
            if anc_id in seen:
                continue
            seen.add(anc_id)
            anc_src = anc_id.split(":")[0] if ":" in anc_id else "UNKNOWN"
            try:
                with driver.session() as session:
                    session.run(
                        "MERGE (t:OntologyTerm {term_id: $tid}) "
                        "SET t.ontology_source = $src",
                        tid=anc_id, src=anc_src,
                    )
                    session.run(
                        "MATCH (a:OntologyTerm {term_id: $from_id}), "
                        "      (b:OntologyTerm {term_id: $to_id}) "
                        f"MERGE (a)-[:{rel_type}]->(b)",
                        from_id=term_id, to_id=anc_id,
                    )
                    ancestors_added += 1
            except Exception as exc:
                print(f"[GraphEnricher] IS_A edge failed {term_id} → {anc_id}: {exc}")

    return ancestors_added


def _load_owl_world():
    """Load owlready2 world for ancestor traversal. Returns world or None."""
    try:
        import owlready2
        from kg.ontology_index import CACHE_DIR, OWL_PATH
        quadstore = CACHE_DIR / "owlready2_quadstore.db"
        if not quadstore.exists():
            print("[GraphEnricher] owlready2 quadstore not found — ancestor traversal disabled.")
            return None
        world = owlready2.World()
        world.set_backend(filename=str(quadstore), exclusive=False)
        if OWL_PATH.exists():
            try:
                world.get_ontology(OWL_PATH.absolute().as_uri()).load()
                print("[GraphEnricher] owlready2 world loaded.")
            except Exception as exc:
                print(f"[GraphEnricher] owlready2 load failed: {exc} — ancestor traversal disabled.")
                return None
        else:
            print("[GraphEnricher] upheno.owl not found — ancestor traversal disabled.")
            return None
        return world
    except Exception as exc:
        print(f"[GraphEnricher] owlready2 unavailable: {exc}")
        return None


def _run_upheno_mapping(
    driver,
    species_norms: list[str],
    model: str | None,
    force: bool,
    embed_backend: str | None = None,
    max_workers: int = 1,
    norm_batch_size: int = 1,
) -> tuple[int, int, int]:
    """
    Map DocEntity nodes to uPheno. Returns (mapped, no_match, ancestors_added).
    """
    entities = _pull_entities_to_map(driver, species_norms, force)
    if not entities:
        print("[GraphEnricher] No entities to map (all already enriched or none found).")
        return 0, 0, 0

    # Deduplicate by surface_form — mapping is text-only, so same sf always gives same result
    unique_sfs = list(dict.fromkeys(e["sf"] for e in entities))
    print(
        f"[GraphEnricher] {len(entities)} entities to map "
        f"({len(unique_sfs)} unique surface forms after deduplication, "
        f"{len(entities) - len(unique_sfs)} duplicates skipped)."
    )

    # Load ontology index
    try:
        from kg.ontology_index import build_index, load_index
        build_index(force=False, embed_backend=embed_backend)
        _embeddings, _terms = load_index(embed_backend=embed_backend)
    except Exception as exc:
        print(f"[GraphEnricher] Cannot load ontology index: {exc} — skipping uPheno mapping.")
        return 0, len(entities), 0

    # Load owlready2 world for ancestor traversal
    world = _load_owl_world()

    # Map unique surface forms only
    try:
        from kg.trait_mapper import map_traits_batch
        unique_results = map_traits_batch(
            unique_sfs,
            no_match_out_path=None,
            model=model,
            embed_backend=embed_backend,
            max_workers=max_workers,
            norm_batch_size=norm_batch_size,
        )
    except Exception as exc:
        print(f"[GraphEnricher] map_traits_batch failed: {exc}")
        return 0, len(entities), 0

    # Build sf → result lookup
    sf_to_result = {sf: result for sf, result in zip(unique_sfs, unique_results)}

    from core.llm_backend import resolve_embed_model_name
    em_name = resolve_embed_model_name(embed_backend)

    mapped_count = 0
    no_match_count = 0
    ancestors_total = 0

    for entity_row in entities:
        eid = entity_row["eid"]
        map_result = sf_to_result[entity_row["sf"]]
        if map_result.get("mapped"):
            anc = _push_mapping(driver, eid, map_result, world, embed_model=em_name)
            mapped_count += 1
            ancestors_total += anc
        else:
            _push_mapping(driver, eid, map_result, world, embed_model=em_name)  # marks upheno_enriched=true
            no_match_count += 1

    print(
        f"[GraphEnricher] uPheno mapping done: "
        f"{mapped_count} mapped, {no_match_count} no-match, "
        f"{ancestors_total} ancestor IS_A edges added."
    )
    return mapped_count, no_match_count, ancestors_total


# ---------------------------------------------------------------------------
# Sub-step 2: DocEntity similarity edges
# ---------------------------------------------------------------------------

def _run_similarity_edges(
    driver,
    species_norms: list[str],
    threshold: float,
    embed_backend: str | None = None,
) -> int:
    """
    Compute cosine similarity between all DocEntity nodes (grouped by species),
    add DOC_SIMILAR_TO edges for cross-species pairs above threshold.
    Returns number of pairs added.
    """
    # Pull entities with their species sets
    cypher = (
        "MATCH (c:DocChunk)-[:MENTIONS]->(e:DocEntity) "
        "WHERE c.species_norm IN $sn "
        "RETURN e.entity_id AS eid, e.surface_form AS sf, "
        "       collect(DISTINCT c.species_norm) AS species_set"
    )
    try:
        from kg.neo4j_client import run_query
        rows = run_query(cypher, {"sn": species_norms})
    except Exception as exc:
        print(f"[GraphEnricher] Pull entities for similarity failed: {exc}")
        return 0

    if len(rows) < 2:
        print("[GraphEnricher] Not enough entities for similarity computation.")
        return 0

    eids = [r["eid"] for r in rows]
    surface_forms = [r["sf"] for r in rows]
    species_sets = [set(r["species_set"]) for r in rows]

    print(f"[GraphEnricher] Computing similarity for {len(eids)} entities...")
    try:
        from core.llm_backend import make_embeddings, resolve_embed_model_name
        embedder = make_embeddings(embed_backend=embed_backend)
        em_name = resolve_embed_model_name(embed_backend)
        vecs = np.array(embedder.embed_documents(surface_forms), dtype="float32")
    except Exception as exc:
        print(f"[GraphEnricher] Embedding failed: {exc}")
        return 0

    # L2 normalise
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms

    pairs_added = 0
    n = len(eids)

    try:
        with driver.session() as session:
            for i in range(n):
                for j in range(i + 1, n):
                    # Only cross-species: species sets must not be identical
                    if species_sets[i] == species_sets[j]:
                        continue
                    score = float(np.dot(vecs[i], vecs[j]))
                    if score < threshold:
                        continue
                    score_r = round(score, 4)
                    try:
                        session.run(
                            "MATCH (a:DocEntity {entity_id: $eid_a}), "
                            "      (b:DocEntity {entity_id: $eid_b}) "
                            "MERGE (a)-[:DOC_SIMILAR_TO {score: $sc, embed_model: $em}]->(b) "
                            "MERGE (b)-[:DOC_SIMILAR_TO {score: $sc, embed_model: $em}]->(a)",
                            eid_a=eids[i], eid_b=eids[j], sc=score_r, em=em_name,
                        )
                        pairs_added += 1
                    except Exception as exc:
                        print(f"[GraphEnricher] DOC_SIMILAR_TO merge failed: {exc}")
    except Exception as exc:
        print(f"[GraphEnricher] Similarity session failed: {exc}")

    print(f"[GraphEnricher] DOC_SIMILAR_TO edges added: {pairs_added} cross-species pairs "
          f"above threshold {threshold}.")
    return pairs_added


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def enrich_doc_entities(
    species_norms: list[str],
    similarity_threshold: float = 0.78,
    model: str | None = None,
    force_reenrich: bool = False,
    log_dir: pathlib.Path | None = None,
    embed_backend: str | None = None,
    max_workers: int = 1,
    norm_batch_size: int = 1,
) -> dict:
    """
    Enrich DocEntity nodes with uPheno ontology mapping and cross-species
    embedding similarity edges.

    Returns summary dict.
    """
    from kg.neo4j_client import _get_driver

    print(f"\n[GraphEnricher] ================================================")
    print(f"[GraphEnricher] Starting enrichment for {len(species_norms)} species.")
    print(f"[GraphEnricher] Species: {species_norms}")
    print(f"[GraphEnricher] ================================================")

    summary: dict[str, Any] = {
        "species_norms": species_norms,
        "entities_seen": 0,
        "entities_mapped": 0,
        "entities_no_match": 0,
        "ancestor_terms_added": 0,
        "similarity_pairs_added": 0,
    }

    driver = _get_driver()
    if driver is None:
        print("[GraphEnricher] Neo4j not reachable — skipping enrichment.")
        return summary

    try:
        _init_enricher_schema(driver)
    except Exception as exc:
        print(f"[GraphEnricher] Schema init failed (non-fatal): {exc}")

    # Sub-step 1: uPheno mapping
    mapped, no_match, ancestors = _run_upheno_mapping(driver, species_norms, model, force_reenrich, embed_backend=embed_backend, max_workers=max_workers, norm_batch_size=norm_batch_size)
    summary["entities_seen"] = mapped + no_match
    summary["entities_mapped"] = mapped
    summary["entities_no_match"] = no_match
    summary["ancestor_terms_added"] = ancestors

    # Sub-step 2: similarity edges
    pairs = _run_similarity_edges(driver, species_norms, similarity_threshold, embed_backend=embed_backend)
    summary["similarity_pairs_added"] = pairs

    driver.close()

    print(f"\n[GraphEnricher] ================================================")
    print(f"[GraphEnricher] Enrichment complete.")
    print(f"[GraphEnricher]   Entities mapped:        {mapped}")
    print(f"[GraphEnricher]   Entities no-match:      {no_match}")
    print(f"[GraphEnricher]   Ancestor terms added:   {ancestors}")
    print(f"[GraphEnricher]   Similarity pairs added: {pairs}")
    print(f"[GraphEnricher] ================================================\n")

    if log_dir:
        try:
            (log_dir / "graph_enricher_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception as exc:
            print(f"[GraphEnricher] Could not write summary: {exc}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enrich DocEntity nodes with uPheno + similarity.")
    parser.add_argument("--species", required=True,
                        help="Comma-separated species norms (e.g. 'talpa europaea,nannospalax ehrenbergi').")
    parser.add_argument("--similarity-threshold", type=float, default=0.78)
    parser.add_argument("--model", default=None)
    parser.add_argument("--force-reenrich", action="store_true")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument(
        "--embed-backend",
        default=None,
        choices=["ollama", "openai"],
        help="Embedding backend to use (default: from EMBED_BACKEND env var or ollama).",
    )
    parser.add_argument(
        "--map-workers",
        type=int,
        default=1,
        help="Number of parallel threads for uPheno mapping (default: 1).",
    )
    parser.add_argument(
        "--norm-batch-size",
        type=int,
        default=1,
        help="Traits per Stage 1 normalization LLM call (default: 1 = one-by-one). "
             "Values > 1 batch multiple traits into a single call.",
    )
    args = parser.parse_args()

    norms = [s.strip().lower() for s in args.species.split(",") if s.strip()]
    log_dir = pathlib.Path(args.log_dir) if args.log_dir else None

    result = enrich_doc_entities(
        species_norms=norms,
        similarity_threshold=args.similarity_threshold,
        model=args.model,
        force_reenrich=args.force_reenrich,
        log_dir=log_dir,
        embed_backend=args.embed_backend,
        max_workers=args.map_workers,
        norm_batch_size=args.norm_batch_size,
    )
    print(json.dumps(result, indent=2))
