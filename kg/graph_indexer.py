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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).parents[1]))

ENTITY_EXTRACTION_PROMPT_FILE = "Prompts/prompt_entity_extraction.txt"
MAX_ENTITY_RETRIES = 1
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


def _batch_already_indexed(driver, chunk_ids: list[str]) -> set[str]:
    """Single query returning the set of chunk_ids already present in Neo4j."""
    try:
        with driver.session() as session:
            result = session.run(
                "MATCH (c:DocChunk) WHERE c.chunk_id IN $ids RETURN c.chunk_id AS cid",
                ids=chunk_ids,
            )
            return {record["cid"] for record in result}
    except Exception as exc:
        print(f"[GraphIndexer] Batch indexed check failed (assuming none indexed): {exc}")
        return set()


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
                # 4. Merge RELATED_TO edges (entity → entity) — single UNWIND call
                try:
                    session.run(
                        "UNWIND $triples AS t "
                        "MATCH (a:DocEntity {entity_id: t.subj_entity_id}), "
                              "(b:DocEntity {entity_id: t.obj_entity_id}) "
                        "MERGE (a)-[r:RELATED_TO {relation_type: t.relation, chunk_id: $cid}]->(b) "
                        "SET r.species_norm = $sn",
                        triples=triples,
                        cid=chunk_id,
                        sn=chunk_node["species_norm"],
                    )
                    triples_merged = len(triples)
                except Exception as exc:
                    print(f"[GraphIndexer] RELATED_TO UNWIND failed for chunk {chunk_id}: {exc}")
    except Exception as exc:
        print(f"[GraphIndexer] _push_chunk_graph failed for chunk {chunk_id} (non-fatal): {exc}")

    return entities_merged, triples_merged


# ---------------------------------------------------------------------------
# Junk chunk filter
# ---------------------------------------------------------------------------

def _is_junk_chunk(text: str) -> tuple[bool, str]:
    """
    Return (True, reason) if this chunk is unlikely to contain extractable
    biological entities, (False, "") otherwise.

    Uses a multi-signal approach: a chunk is junk only if it fires >= 2 signals,
    so real content with incidentally high numbers (methods sections etc.) passes.
    """
    text = text.strip()
    words = text.split()
    n_words = len(words)
    n_chars = len(text)

    signals: list[str] = []

    # Signal 1 — very low word count
    if n_words < 20:
        signals.append(f"S1:low_words({n_words})")

    # Signal 2 — figure/table caption prefix (only suspicious if also short)
    if n_words < 50 and re.match(
        r"^(fig(ure)?s?\.?|table|supplementary\s+(fig(ure)?|table)|supp\.)\s*[\d\w]",
        text,
        re.IGNORECASE,
    ):
        signals.append("S2:fig_table_prefix")

    # Signal 3 — high digit density (tables, data blocks, statistics sections)
    if n_chars > 0:
        digit_ratio = sum(1 for c in text if c.isdigit()) / n_chars
        if digit_ratio > 0.30:
            signals.append(f"S3:digit_density({digit_ratio:.0%})")

    # Signal 4 — reference list (>= 40% of non-empty lines start with digit or bracket)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 3:
        ref_lines = sum(1 for ln in lines if re.match(r"^[\d\[\{]", ln))
        if ref_lines / len(lines) >= 0.40:
            signals.append(f"S4:ref_list({ref_lines}/{len(lines)} lines)")

    # Signal 5 — almost no sentence-ending punctuation (headings, labels)
    sentence_ends = len(re.findall(r"[.!?][\s\n]", text)) + (1 if text.endswith((".", "!", "?")) else 0)
    if sentence_ends < 2 and n_words < 40:
        signals.append(f"S5:low_sentences({sentence_ends})")

    if len(signals) >= 2:
        return True, ", ".join(signals)
    return False, ""


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

            # Strip thinking-mode tags (qwen3/qwen3.5 family wraps output in <think>...</think>)
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

            # Strip markdown fences if present
            fence = re.search(r"```(?:json)?\s*(.*?)```", raw, re.DOTALL)
            if fence:
                raw = fence.group(1).strip()

            parsed = json.loads(raw)
        except Exception as exc:
            exc_str = str(exc)
            # Server crash (segfault, OOM, 500) — retrying won't help; abort the run
            if "500" in exc_str or "segmentation fault" in exc_str.lower() or "terminated" in exc_str.lower():
                raise RuntimeError(f"[GraphIndexer] LLM server crashed — aborting run: {exc}") from exc
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
# Per-chunk worker (called from ThreadPoolExecutor)
# ---------------------------------------------------------------------------

def _index_one_chunk(
    i: int,
    doc_id_chroma: str,
    text: str,
    meta: dict,
    species_norm: str,
    prompt_template: str,
    llm,
    driver,
    already_indexed_ids: set,
    indexed_at: str,
    total: int,
) -> dict:
    """
    Process a single chunk: junk filter → LLM extraction → Neo4j push.
    Returns a result dict with keys: status, chunk_id, entities, triples.
    Thread-safe: llm is a stateless HTTP client; driver supports concurrent sessions.
    already_indexed_ids is computed once upfront via _batch_already_indexed.
    """
    meta = meta or {}
    source_path = meta.get("source_path", doc_id_chroma)
    chunk_index = int(meta.get("chunk_index", i))
    chunk_id = _make_chunk_id(species_norm, source_path, chunk_index)

    if chunk_id in already_indexed_ids:
        return {"status": "skipped", "chunk_id": chunk_id, "entities": 0, "triples": 0}

    junk, junk_reason = _is_junk_chunk(text or "")
    if junk:
        print(f"[GraphIndexer] [{i + 1}/{total}] Skipping junk chunk {chunk_id} "
              f"(source: {source_path}, idx: {chunk_index}) — {junk_reason}")
        return {"status": "junk", "chunk_id": chunk_id, "entities": 0, "triples": 0}

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

    print(f"[GraphIndexer] [{i + 1}/{total}] Indexing chunk {chunk_id} "
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

    print(f"[GraphIndexer]   → {len(extracted['entities'])} entities, "
          f"{len(extracted['triples'])} triples")

    return {"status": "indexed", "chunk_id": chunk_id, "entities": ents_merged, "triples": triples_merged}


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------

def index_species(
    species_norm: str,
    chroma_vectorstore,
    llm_model: str | None = None,
    index_llm_model: str | None = None,
    temperature: float = 0.0,
    force_reindex: bool = False,
    log_dir: pathlib.Path | None = None,
    max_workers: int = 1,
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
        "chunks_junk": 0,
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

    # Build LLM — use dedicated index model (smaller/faster) if provided, else default to qwen2.5:7b
    from core.llm_backend import make_chat_llm
    _index_model = index_llm_model or "granite4.1:8b"
    print(f"[GraphIndexer] Using model '{_index_model}' for entity extraction "
          f"({max_workers} worker(s)).")
    llm = make_chat_llm(model=_index_model, temperature=temperature)

    indexed_at = datetime.now(timezone.utc).isoformat()
    total = len(ids)

    # Compute all chunk IDs upfront for the batch check
    all_chunk_ids = [
        _make_chunk_id(species_norm, (m or {}).get("source_path", did), int((m or {}).get("chunk_index", i)))
        for i, (did, m) in enumerate(zip(ids, metadatas))
    ]

    if force_reindex:
        already_indexed_ids: set[str] = set()
    else:
        print(f"[GraphIndexer] Checking which chunks are already indexed (single batch query)...")
        already_indexed_ids = _batch_already_indexed(driver, all_chunk_ids)
        if already_indexed_ids:
            print(f"[GraphIndexer] {len(already_indexed_ids)}/{total} chunks already indexed, skipping.")

    chunk_args = [
        (i, did, txt, meta, species_norm, prompt_template, llm, driver, already_indexed_ids, indexed_at, total)
        for i, (did, txt, meta) in enumerate(zip(ids, documents, metadatas))
    ]

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_index_one_chunk, *a): a[0] for a in chunk_args}
        for future in as_completed(futures):
            try:
                r = future.result()
                if r["status"] == "indexed":
                    summary["chunks_indexed"] += 1
                    summary["entities_created"] += r["entities"]
                    summary["triples_created"] += r["triples"]
                elif r["status"] == "skipped":
                    summary["chunks_skipped"] += 1
                elif r["status"] == "junk":
                    summary["chunks_junk"] += 1
            except RuntimeError as exc:
                # Server crash propagated from _extract_entities_from_chunk — abort immediately
                pool.shutdown(wait=False, cancel_futures=True)
                raise
            except Exception as exc:
                chunk_i = futures[future]
                print(f"[GraphIndexer] Chunk {chunk_i + 1}/{total} failed unexpectedly: {exc}")

    driver.close()

    print(f"[GraphIndexer] Done for '{species_norm}': "
          f"{summary['chunks_indexed']} indexed, "
          f"{summary['chunks_skipped']} skipped (already done), "
          f"{summary['chunks_junk']} skipped (junk filter), "
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
    parser.add_argument("--model", default=None, help="LLM model for entity extraction (default: qwen2.5:7b).")
    parser.add_argument("--index-workers", type=int, default=1,
                        help="Parallel threads for chunk indexing (default: 1).")
    parser.add_argument("--force-reindex", action="store_true", help="Re-index even if chunk already in Neo4j.")
    parser.add_argument("--log-dir", default=None, help="Directory to write graph_indexer_summary.json.")
    args = parser.parse_args()

    from core.rag_cli import RAG
    rag = RAG(persist_dir=args.chroma_dir)
    log_dir = pathlib.Path(args.log_dir) if args.log_dir else None

    result = index_species(
        species_norm=args.species.lower().strip(),
        chroma_vectorstore=rag.vectorstore,
        index_llm_model=args.model,
        force_reindex=args.force_reindex,
        log_dir=log_dir,
        max_workers=args.index_workers,
    )
    print(json.dumps(result, indent=2))
