#!/usr/bin/env python3
"""
Build the in-memory graph (node/edge lists) from mapped traits,
push to Neo4j, and serialize to GraphML.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parents[1]))

ALLOWED_PREFIXES = ("UPHENO:", "HP:", "MP:", "GO:", "ENVO:", "UBERON:", "ZP:", "XPO:")
MAX_ANCESTOR_DEPTH = 3
SIMILAR_TO_THRESHOLD = 0.78


def _is_allowed(term_id: str) -> bool:
    return isinstance(term_id, str) and any(term_id.startswith(p) for p in ALLOWED_PREFIXES)


# ---------------------------------------------------------------------------
# OAK ancestor fetching
# ---------------------------------------------------------------------------

def _curie_to_iri_fragment(curie: str) -> str:
    """Convert UPHENO:0001234 → UPHENO_0001234 for IRI matching."""
    return curie.replace(":", "_")


def _load_ancestors(term_id: str, world, depth: int = MAX_ANCESTOR_DEPTH) -> list[tuple[str, str]]:
    """
    Return list of (ancestor_term_id, relation_type) up to `depth` hops.
    Uses owlready2 world for lookup. Filters to allowed prefixes.
    """
    results: list[tuple[str, str]] = []
    seen: set[str] = {term_id}

    # Find the class in the world by IRI fragment
    frag = _curie_to_iri_fragment(term_id)
    cls = None
    for c in world.classes():
        if c.iri and frag in c.iri:
            cls = c
            break
    if cls is None:
        return results

    frontier = [cls]
    for d in range(depth):
        next_frontier = []
        for c in frontier:
            try:
                for parent in c.is_a:
                    parent_iri = getattr(parent, "iri", None)
                    if not parent_iri:
                        continue
                    # Convert IRI to CURIE
                    for sep in ("_", "/"):
                        if sep in parent_iri:
                            last = parent_iri.rsplit(sep, 1)[-1]
                            prefix = parent_iri.rsplit(sep, 1)[0].rsplit("/", 1)[-1]
                            pid = f"{prefix}:{last}"
                            if _is_allowed(pid) and pid not in seen:
                                seen.add(pid)
                                results.append((pid, "IS_A"))
                                next_frontier.append(parent)
                            break
            except Exception:
                continue
        frontier = next_frontier
        if not frontier:
            break

    return results


# ---------------------------------------------------------------------------
# Semantic similarity edges
# ---------------------------------------------------------------------------

def _add_similar_to_edges(
    species_trait_map: dict[str, list[dict]],
    run_id: str,
    threshold: float = SIMILAR_TO_THRESHOLD,
) -> list[dict]:
    """
    Embed all normalized trait strings and add SIMILAR_TO edges between
    traits from *different* species whose cosine similarity >= threshold.
    Returns a list of edge dicts (same format as build_graph edges).
    """
    import numpy as np
    from core.llm_backend import make_embeddings

    # Collect (species, raw_trait, normalized_trait) tuples
    entries = []
    for species_name, trait_results in species_trait_map.items():
        for tr in trait_results:
            raw = tr.get("raw_trait", "")
            norm = tr.get("normalized_trait", "") or raw
            if raw:
                entries.append((species_name, raw, norm))

    if len(entries) < 2:
        return []

    texts = [e[2] for e in entries]
    print(f"[KG] Embedding {len(texts)} traits for SIMILAR_TO edge computation...")
    try:
        embedder = make_embeddings()
        vecs = np.array(embedder.embed_documents(texts), dtype="float32")
    except Exception as exc:
        print(f"[KG] SIMILAR_TO embedding failed (non-fatal): {exc}")
        return []

    # Normalise for cosine similarity
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms

    edges = []
    n = len(entries)
    pair_count = 0
    for i in range(n):
        for j in range(i + 1, n):
            species_i, raw_i, _ = entries[i]
            species_j, raw_j, _ = entries[j]
            if species_i == species_j:
                continue  # only cross-species edges
            score = float(np.dot(vecs[i], vecs[j]))
            if score >= threshold:
                edges.append({
                    "type": "SIMILAR_TO",
                    "from_label": "Trait",
                    "from_id": raw_i,
                    "to_label": "Trait",
                    "to_id": raw_j,
                    "similarity_score": round(score, 4),
                    "run_id": run_id,
                })
                # Also add reverse direction so the graph is undirected
                edges.append({
                    "type": "SIMILAR_TO",
                    "from_label": "Trait",
                    "from_id": raw_j,
                    "to_label": "Trait",
                    "to_id": raw_i,
                    "similarity_score": round(score, 4),
                    "run_id": run_id,
                })
                pair_count += 1

    print(f"[KG] SIMILAR_TO edges added: {pair_count} cross-species pairs above threshold {threshold}")
    return edges


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------

def build_graph(
    species_trait_map: dict[str, list[dict]],
    run_id: str,
    input_file: str = "",
) -> tuple[list[dict], list[dict]]:
    """
    Build node and edge lists from mapped traits.

    species_trait_map: { species_name: [mapped_trait_result, ...] }
    Each mapped_trait_result has the schema from trait_mapper.map_trait().

    Returns (nodes, edges) where each dict has a '_label' key.
    """

    nodes: list[dict] = []
    edges: list[dict] = []
    seen_terms: set[str] = set()

    # Load owlready2 world for ancestor traversal
    adapter = None
    try:
        import owlready2
        from kg.ontology_index import CACHE_DIR, OWL_PATH
        quadstore = CACHE_DIR / "owlready2_quadstore.db"
        if quadstore.exists():
            world = owlready2.World()
            world.set_backend(filename=str(quadstore), exclusive=False)
            adapter = world  # pass world as adapter
            print("[KG] owlready2 world loaded for ancestor traversal.")
        else:
            print("[KG] owlready2 quadstore not found — ancestor traversal disabled. Run --build first.")
    except Exception as exc:
        print(f"[KG] Could not load owlready2 world (ancestor traversal disabled): {exc}")

    for species_name, trait_results in species_trait_map.items():
        # Species node
        nodes.append({
            "_label": "Species",
            "name": species_name,
            "run_id": run_id,
            "input_file": input_file,
        })

        for tr in trait_results:
            raw_trait = tr.get("raw_trait", "")
            if not raw_trait:
                continue

            # Trait node
            cluster_id = tr.get("cluster_id", "")
            nodes.append({
                "_label": "Trait",
                "raw_trait": raw_trait,
                "normalized_trait": tr.get("normalized_trait", ""),
                "run_id": run_id,
                "cluster_id": cluster_id,
            })

            # Species → Trait edge
            edges.append({
                "type": "HAS_TRAIT",
                "from_label": "Species",
                "from_id": species_name,
                "to_label": "Trait",
                "to_id": raw_trait,
                "confidence_score": tr.get("cosine_score") or 0.0,
                "cluster_id": cluster_id,
                "run_id": run_id,
            })

            if not tr.get("mapped") or not tr.get("term_id"):
                continue

            term_id = tr["term_id"]

            # OntologyTerm node (if not already added)
            if term_id not in seen_terms:
                seen_terms.add(term_id)
                nodes.append({
                    "_label": "OntologyTerm",
                    "term_id": term_id,
                    "term_name": tr.get("term_name", ""),
                    "ontology_source": term_id.split(":")[0] if ":" in term_id else "UNKNOWN",
                    "definition": "",
                    "run_id": run_id,
                })

            # Trait → OntologyTerm edge
            edges.append({
                "type": "MAPPED_TO",
                "from_label": "Trait",
                "from_id": raw_trait,
                "to_label": "OntologyTerm",
                "to_id": term_id,
                "cosine_score": tr.get("cosine_score") or 0.0,
                "confidence": tr.get("confidence", ""),
                "broadened": tr.get("broadened", False),
                "run_id": run_id,
            })

            # Ancestor traversal
            if adapter:
                ancestors = _load_ancestors(term_id, adapter, depth=MAX_ANCESTOR_DEPTH)
                for anc_id, rel_type in ancestors:
                    if anc_id not in seen_terms:
                        seen_terms.add(anc_id)
                        nodes.append({
                            "_label": "OntologyTerm",
                            "term_id": anc_id,
                            "term_name": "",
                            "ontology_source": anc_id.split(":")[0] if ":" in anc_id else "UNKNOWN",
                            "definition": "",
                            "run_id": run_id,
                        })
                    edges.append({
                        "type": rel_type,
                        "from_label": "OntologyTerm",
                        "from_id": term_id,
                        "to_label": "OntologyTerm",
                        "to_id": anc_id,
                        "depth_from_matched": 1,
                        "run_id": run_id,
                    })

    # Semantic similarity edges across species
    similar_edges = _add_similar_to_edges(species_trait_map, run_id)
    edges.extend(similar_edges)

    print(f"[KG] Graph built: {len(nodes)} nodes, {len(edges)} edges.")
    return nodes, edges


def push_and_serialize(
    nodes: list[dict],
    edges: list[dict],
    run_id: str,
    graphml_out: Path,
) -> None:
    """Push to Neo4j (non-fatal) and write GraphML."""
    from kg import neo4j_client
    neo4j_client.push_graph(nodes, edges, run_id)
    _write_graphml(nodes, edges, graphml_out)


def _write_graphml(nodes: list[dict], edges: list[dict], out_path: Path) -> None:
    try:
        import networkx as nx
        G = nx.DiGraph()
        for n in nodes:
            label = n.get("_label", "Node")
            node_id = n.get("term_id") or n.get("name") or n.get("raw_trait", "?")
            attrs = {k: v for k, v in n.items() if k not in ("_label",)}
            attrs["label"] = label
            G.add_node(node_id, **attrs)
        for e in edges:
            src = e.get("from_id", "")
            dst = e.get("to_id", "")
            etype = e.get("type", "RELATED")
            attrs = {k: v for k, v in e.items() if k not in ("from_id", "to_id", "from_label", "to_label", "type")}
            if src and dst:
                G.add_edge(src, dst, relation=etype, **attrs)
        import os
        tmp = Path(str(out_path) + ".tmp")
        nx.write_graphml(G, str(tmp))
        os.replace(tmp, out_path)
        print(f"[KG] GraphML written: {out_path} ({G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")
    except ImportError:
        print("[KG] networkx not installed — skipping GraphML output. pip install networkx")
    except Exception as exc:
        print(f"[KG] GraphML write failed (non-fatal): {exc}")
