#!/usr/bin/env python3
"""
Graph Indexer — Step 1 of the GraphRAG pipeline.

For a given species, pulls all text chunks from the Chroma vectorstore,
runs LLM entity/relation extraction on each chunk, and pushes the resulting
document graph (DocChunk + DocEntity + MENTIONS + RELATED_TO) to Neo4j.

This builds the document graph at ingest time so that retrieval can traverse
it instead of doing flat vector similarity search.

Usage (standalone, after ingestion):
    python kg/graph_indexer.py --species "talpa europaea" --chroma-dir ./chroma_store_ollama
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import re
import sys
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

ENTITY_EXTRACTION_PROMPT_FILE = "Prompts/prompt_entity_extraction.txt"
MAX_ENTITY_RETRIES = 2
CHUNK_ID_LEN = 16  # hex chars from sha256

ALLOWED_ENTITY_TYPES = {"Species", "Anatomy", "Phenotype", "Habitat", "Process", "Gene"}
ALLOWED_RELATION_TYPES = {
    "HAS_ANATOMY", "HAS_PHENOTYPE", "LIVES_IN", "PART_OF",
    "CONTAINS", "INVOLVED_IN", "CAUSES", "ASSOCIATED_WITH",
}

_DOC_GRAPH_SCHEMA = [
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:DocChunk)  REQUIRE c.chunk_id  IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:DocEntity) REQUIRE e.entity_id IS UNIQUE",
    "CREATE INDEX IF NOT EXISTS FOR (c:DocChunk)  ON (c.species_norm)",
    "CREATE INDEX IF NOT EXISTS FOR (e:DocEntity) ON (e.species_norm)",
    "CREATE INDEX IF NOT EXISTS FOR (e:DocEntity) ON (e.entity_type)",
]


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def _make_chunk_id(species_norm: str, source_path: str, chunk_index: int) -> str:
    raw = f"{species_norm}|{source_path}|{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:CHUNK_ID_LEN]


def _make_entity_id(entity_type: str, surface_form: str) -> str:
    raw = f"{entity_type.upper()}:{surface_form.strip().lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:CHUNK_ID_LEN]


# ---------------------------------------------------------------------------
# Neo4j helpers
# ---------------------------------------------------------------------------

def _get_driver():
    from kg.neo4j_client import _get_driver as _base_get_driver
    return _base_get_driver()


def _init_doc_graph_schema(driver) -> None:
    with driver.session() as session:
        for q in _DOC_GRAPH_SCHEMA:
            try:
                session.run(q)
            except Exception as exc:
                print(f"[GraphIndexer] Schema init warning (non-fatal): {exc}")


def _already_indexed(driver, chunk_id: str) -> bool:
    try:
        with driver.session() as session:
            result = session.run(
                "MATCH (c:DocChunk {chunk_id: $cid}) RETURN count(c) AS n",
                cid=chunk_id,
            )
            record = result.single()
            return (record["n"] > 0) if record else False
    except Exception as exc:
        print(f"[GraphIndexer] _already_indexed check failed (assuming not indexed): {exc}")
        return False


def _push_chunk_graph(
    driver,
    chunk_node: dict,
    entities: list[dict],
    triples: list[dict],
) -> tuple[int, int]:
    """
    Write one chunk's graph data to Neo4j in a single session.
    Returns (entities_merged, triples_merged).
    """
    chunk_id = chunk_node["chunk_id"]
    entities_merged = 0
    triples_merged = 0

    try:
        with driver.session() as session:
            # 1. Merge DocChunk node
            session.run(
                "MERGE (c:DocChunk {chunk_id: $cid}) SET c += $props",
                cid=chunk_id,
                props=chunk_node,
            )

            if entities:
                # 2. Merge all DocEntity nodes
                session.run(
                    "UNWIND $ents AS e "
                    "MERGE (n:DocEntity {entity_id: e.entity_id}) "
                    "SET n += e",
                    ents=entities,
                )
                entities_merged = len(entities)

                # 3. Merge MENTIONS edges (chunk → entity)
                session.run(
                    "MATCH (c:DocChunk {chunk_id: $cid}) "
                    "UNWIND $eids AS eid "
                    "MATCH (e:DocEntity {entity_id: eid}) "
                    "MERGE (c)-[:MENTIONS {chunk_id: $cid, species_norm: $sn}]->(e)",
                    cid=chunk_id,
                    sn=chunk_node["species_norm"],
                    eids=[e["entity_id"] for e in entities],
                )

            if triples:
                # 4. Merge RELATED_TO edges (entity → entity)
                for t in triples:
                    try:
                        session.run(
                            "MATCH (a:DocEntity {entity_id: $sid}), (b:DocEntity {entity_id: $oid}) "
                            "MERGE (a)-[r:RELATED_TO {relation_type: $rel, chunk_id: $cid}]->(b) "
                            "SET r.species_norm = $sn",
                            sid=t["subj_entity_id"],
                            oid=t["obj_entity_id"],
                            rel=t["relation"],
                            cid=chunk_id,
                            sn=chunk_node["species_norm"],
                        )
                        triples_merged += 1
                    except Exception as exc:
                        print(f"[GraphIndexer] RELATED_TO merge failed for triple {t}: {exc}")
    except Exception as exc:
        print(f"[GraphIndexer] _push_chunk_graph failed for chunk {chunk_id} (non-fatal): {exc}")

    return entities_merged, triples_merged


# ---------------------------------------------------------------------------
# LLM entity extraction
# ---------------------------------------------------------------------------

def _extract_entities_from_chunk(
    chunk_text: str,
    species_norm: str,
    llm,
    prompt_template: str,
) -> dict:
    """
    Call the LLM to extract entities and triples from a single chunk.
    Returns {"entities": [...], "triples": [...]} with resolved entity_ids,
    or {"entities": [], "triples": []} on failure.
    """
    prompt = prompt_template.replace("{species_name}", species_norm).replace("{chunk_text}", chunk_text)

    for attempt in range(1, MAX_ENTITY_RETRIES + 2):
        try:
            from langchain_core.messages import HumanMessage
            resp = llm.invoke([HumanMessage(content=prompt)])
            raw = resp.content if hasattr(resp, "content") else str(resp)
            raw = raw.strip()

            # Strip markdown fences if present
            fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
            if fence:
                raw = fence.group(1).strip()

            parsed = json.loads(raw)
        except Exception as exc:
            if attempt <= MAX_ENTITY_RETRIES:
                print(f"[GraphIndexer] Parse attempt {attempt} failed: {exc}. Retrying...")
                continue
            print(f"[GraphIndexer] Entity extraction failed after {MAX_ENTITY_RETRIES + 1} attempts: {exc}")
            return {"entities": [], "triples": []}

        # Validate and build entity list with IDs
        raw_entities = parsed.get("entities", [])
        if not isinstance(raw_entities, list):
            return {"entities": [], "triples": []}

        entities: list[dict] = []
        for e in raw_entities:
            if not isinstance(e, dict):
                continue
            text = str(e.get("text", "")).strip()
            etype = str(e.get("type", "")).strip()
            if not text or etype not in ALLOWED_ENTITY_TYPES:
                continue
            entities.append({
                "entity_id": _make_entity_id(etype, text),
                "surface_form": text,
                "entity_type": etype,
                "species_norm": species_norm,
                "canonical": text.lower().strip(),
            })

        # Deduplicate entities by entity_id
        seen_eids: set[str] = set()
        unique_entities: list[dict] = []
        for e in entities:
            if e["entity_id"] not in seen_eids:
                seen_eids.add(e["entity_id"])
                unique_entities.append(e)
        entities = unique_entities

        # Validate triples (reference entities by index)
        raw_triples = parsed.get("triples", [])
        if not isinstance(raw_triples, list):
            return {"entities": entities, "triples": []}

        triples: list[dict] = []
        for t in raw_triples:
            if not isinstance(t, dict):
                continue
            try:
                si = int(t.get("subj_idx", -1))
                oi = int(t.get("obj_idx", -1))
                rel = str(t.get("relation", "")).strip()
            except (ValueError, TypeError):
                continue
            if si < 0 or oi < 0 or si >= len(entities) or oi >= len(entities):
                continue
            if rel not in ALLOWED_RELATION_TYPES:
                continue
            triples.append({
                "subj_entity_id": entities[si]["entity_id"],
                "obj_entity_id": entities[oi]["entity_id"],
                "relation": rel,
            })

        return {"entities": entities, "triples": triples}

    return {"entities": [], "triples": []}


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def index_species(
    species_norm: str,
    chroma_vectorstore,
    llm_model: str | None = None,
    temperature: float = 0.0,
    force_reindex: bool = False,
    log_dir: pathlib.Path | None = None,
) -> dict:
    """
    Pull all Chroma chunks for species_norm, run LLM entity extraction per chunk,
    and push DocChunk + DocEntity + MENTIONS + RELATED_TO to Neo4j.

    Returns summary dict:
    {
      "species_norm": str,
      "chunks_seen": int,
      "chunks_skipped": int,   # already indexed
      "chunks_indexed": int,
      "entities_created": int,
      "triples_created": int,
    }
    """
    summary: dict[str, Any] = {
        "species_norm": species_norm,
        "chunks_seen": 0,
        "chunks_skipped": 0,
        "chunks_indexed": 0,
        "entities_created": 0,
        "triples_created": 0,
    }

    driver = _get_driver()
    if driver is None:
        print(f"[GraphIndexer] Neo4j not reachable — skipping graph indexing for {species_norm}.")
        return summary

    try:
        _init_doc_graph_schema(driver)
    except Exception as exc:
        print(f"[GraphIndexer] Schema init failed (non-fatal): {exc}")

    # Pull all chunks for this species from Chroma
    print(f"[GraphIndexer] Pulling chunks for '{species_norm}' from Chroma...")
    try:
        result = chroma_vectorstore.get(
            where={"specie": species_norm},
            include=["documents", "metadatas"],
        )
    except Exception as exc:
        print(f"[GraphIndexer] Chroma get failed: {exc}")
        driver.close()
        return summary

    ids = result.get("ids", [])
    documents = result.get("documents", [])
    metadatas = result.get("metadatas", [])

    if not ids:
        print(f"[GraphIndexer] No chunks found in Chroma for '{species_norm}'.")
        driver.close()
        return summary

    print(f"[GraphIndexer] Found {len(ids)} chunks. Loading prompt and LLM...")
    summary["chunks_seen"] = len(ids)

    # Load prompt template
    try:
        prompt_template = pathlib.Path(ENTITY_EXTRACTION_PROMPT_FILE).read_text(encoding="utf-8")
    except Exception as exc:
        print(f"[GraphIndexer] Cannot load prompt file {ENTITY_EXTRACTION_PROMPT_FILE}: {exc}")
        driver.close()
        return summary

    # Build LLM
    from core.llm_backend import make_chat_llm
    llm = make_chat_llm(model=llm_model, temperature=temperature)

    indexed_at = datetime.now(timezone.utc).isoformat()

    for i, (doc_id_chroma, text, meta) in enumerate(zip(ids, documents, metadatas)):
        meta = meta or {}
        source_path = meta.get("source_path", doc_id_chroma)
        chunk_index = int(meta.get("chunk_index", i))
        chunk_id = _make_chunk_id(species_norm, source_path, chunk_index)

        if not force_reindex and _already_indexed(driver, chunk_id):
            summary["chunks_skipped"] += 1
            continue

        # Build chunk node dict
        chunk_node = {
            "chunk_id": chunk_id,
            "species_norm": species_norm,
            "source_path": source_path,
            "doc_id": str(meta.get("doc_id", "")),
            "chunk_index": chunk_index,
            "title": str(meta.get("title", "")),
            "publication_year": meta.get("publication_year") or 0,
            "text_snippet": (text or "")[:300],
            "indexed_at": indexed_at,
        }

        print(f"[GraphIndexer] [{i + 1}/{len(ids)}] Indexing chunk {chunk_id} "
              f"(source: {source_path}, idx: {chunk_index})")

        extracted = _extract_entities_from_chunk(
            chunk_text=text or "",
            species_norm=species_norm,
            llm=llm,
            prompt_template=prompt_template,
        )

        ents_merged, triples_merged = _push_chunk_graph(
            driver,
            chunk_node,
            extracted["entities"],
            extracted["triples"],
        )

        summary["chunks_indexed"] += 1
        summary["entities_created"] += ents_merged
        summary["triples_created"] += triples_merged

        print(f"[GraphIndexer]   → {len(extracted['entities'])} entities, "
              f"{len(extracted['triples'])} triples")

    driver.close()

    print(f"[GraphIndexer] Done for '{species_norm}': "
          f"{summary['chunks_indexed']} indexed, "
          f"{summary['chunks_skipped']} skipped, "
          f"{summary['entities_created']} entities, "
          f"{summary['triples_created']} triples.")

    if log_dir:
        out = log_dir / "graph_indexer_summary.json"
        try:
            out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[GraphIndexer] Summary written to {out}")
        except Exception as exc:
            print(f"[GraphIndexer] Could not write summary: {exc}")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index species chunks into Neo4j document graph.")
    parser.add_argument("--species", required=True, help="Species name (normalized, e.g. 'talpa europaea').")
    parser.add_argument("--chroma-dir", default="./chroma_store_ollama", help="Chroma persistence directory.")
    parser.add_argument("--model", default=None, help="Override LLM model for entity extraction.")
    parser.add_argument("--force-reindex", action="store_true", help="Re-index even if chunk already in Neo4j.")
    parser.add_argument("--log-dir", default=None, help="Directory to write graph_indexer_summary.json.")
    args = parser.parse_args()

    from core.rag_cli import RAG
    rag = RAG(persist_dir=args.chroma_dir)
    log_dir = pathlib.Path(args.log_dir) if args.log_dir else None

    result = index_species(
        species_norm=args.species.lower().strip(),
        chroma_vectorstore=rag.vectorstore,
        llm_model=args.model,
        force_reindex=args.force_reindex,
        log_dir=log_dir,
    )
    print(json.dumps(result, indent=2))
