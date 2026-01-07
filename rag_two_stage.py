#!/usr/bin/env python3
"""
Two-stage RAG pipeline:
1) Per-species extraction: build a cited trait inventory per species.
2) Cross-species synthesis: find common traits across inventories.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from collections import defaultdict
from typing import Optional
import numpy as np

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from rag_cli import (
    RAG,
    maximal_marginal_relevance,
    DEFAULT_QUERY,
)


def per_species_retrieve(rag: RAG, question: str, specie: str) -> list:
    """Reuse the vectorstore to get a diversified set of chunks for one species."""
    candidates = rag.vectorstore.similarity_search_with_score(
        question, k=rag.per_species_fetch_k, filter={"specie": specie}
    )
    if not candidates:
        return []
    try:
        query_embedding = np.array(
            rag.vectorstore._embedding_function.embed_query(question), dtype=float  # type: ignore[attr-defined]
        )
        doc_embeddings = np.array(
            rag.vectorstore._embedding_function.embed_documents(  # type: ignore[attr-defined]
                [doc.page_content for doc, _ in candidates]
            ),
            dtype=float,
        )
        mmr_indices = maximal_marginal_relevance(
            query_embedding,
            doc_embeddings,
            k=min(rag.per_species_final_k, len(candidates)),
            lambda_mult=rag.mmr_lambda,
        )
    except Exception:
        mmr_indices = list(range(min(rag.per_species_final_k, len(candidates))))
    selected = [candidates[idx] for idx in mmr_indices]
    sorted_by_score = sorted(selected, key=lambda t: t[1])
    keep_n = max(1, math.ceil(rag.per_species_keep_percentile * len(sorted_by_score)))
    kept = sorted_by_score[:keep_n]
    docs = []
    for doc, score in kept:
        if rag.threshold is None or score <= rag.threshold:
            docs.append(doc)
    return docs


def slugify(name: str) -> str:
    """Create a simple filename-friendly slug."""
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def build_allowed_tags(docs: list) -> list[str]:
    return sorted(
        {
            "[source: {id}|chunk:{idx}]".format(
                id=(doc.metadata or {}).get("doc_id", "unknown"),
                idx=(doc.metadata or {}).get("chunk_index", "na"),
            )
            for doc in docs
        }
    )


def format_docs(docs: list) -> str:
    formatted = []
    for doc in docs:
        meta = doc.metadata or {}
        tag = "[source: {id}|chunk:{idx}]".format(
            id=meta.get("doc_id") or meta.get("openalex_id", "unknown"),
            idx=meta.get("chunk_index", "na"),
        )
        formatted.append(f"{tag}\n{doc.page_content}")
    return "\n\n".join(formatted)


INVENTORY_PROMPT_FILE = "prompt_inventory.txt"
SYNTHESIS_PROMPT_FILE = "prompt_synthesis.txt"


def run_two_stage(question: Optional[str], species_groups: Optional[list], log_runs: bool = False) -> None:
    # derive final question: prefer provided; else join canonical species; else fallback default
    final_question = question
    if not final_question and species_groups:
        final_question = ", ".join([entry.get("canonical", "").strip() for entry in species_groups if entry.get("canonical")])
    if not final_question:
        final_question = DEFAULT_QUERY

    rag = RAG(log_runs=log_runs)

    # Ingestion only (no one-stage generation)
    groups = []
    if species_groups:
        for entry in species_groups:
            canonical = entry.get("canonical", "").strip().lower()
            aliases = [a.strip() for a in entry.get("aliases", []) if a.strip()]
            if canonical:
                search_terms = [canonical] + [a.lower() for a in aliases]
                groups.append((canonical, search_terms))
    else:
        terms = [t.strip() for t in final_question.split(",") if t.strip()]
        groups = [(t.lower(), [t.lower()]) for t in terms]

    for canonical, search_terms in groups:
        wiki_ingested = False
        if canonical:
            wiki_ingested = rag.ingest_wikipedia(title=canonical, specie_norm=canonical)
        if not wiki_ingested:
            for term in search_terms:
                if term == canonical:
                    continue
                if rag.ingest_wikipedia(title=term, specie_norm=canonical):
                    wiki_ingested = True
                    break

        rag.ingest_pmc_texts(query=canonical, specie_norm=canonical)

        for term in search_terms:
            papers = rag.fetch_and_prepare(query=term, specie_norm=canonical)
            for paper_entry in papers:
                pdf_path = paper_entry["pdf_path"]
                paper_meta = paper_entry.get("paper") or {}
                if "doc_id" not in paper_meta:
                    paper_meta["doc_id"] = paper_entry.get("doc_id")
                rag.load_ocr(pdf_path=str(pdf_path), puppy=canonical, paper_meta=paper_meta)
                rag.ingested_per_species[canonical].add(str(pdf_path))

    species_list = list(rag.ingested_per_species.keys())
    inventories: dict[str, list[dict]] = {}

    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)

    traits_dir = pathlib.Path("traits")
    traits_dir.mkdir(parents=True, exist_ok=True)

    # Stage 1: per-species extraction
    for specie in species_list:
        # use species name as the retrieval query to focus on that species' evidence
        docs = per_species_retrieve(rag, specie, specie)
        ingested = len(rag.ingested_per_species.get(specie, set()))
        used = len(docs)
        if not docs:
            inventories[specie] = []
            print(f"[Inventory] {specie}: {ingested} ingested / {used} used (no docs)")
            continue
        allowed_tags = build_allowed_tags(docs)
        print(f"[Inventory] {specie}: {ingested} ingested / {used} used")
        prompt = PromptTemplate(
            input_variables=["context", "allowed_tags"],
            template=pathlib.Path(INVENTORY_PROMPT_FILE).read_text(encoding="utf-8"),
            template_format="jinja2",
        )
        chain = (
            {
                "context": RunnableLambda(lambda _: format_docs(docs)),
                "allowed_tags": RunnableLambda(lambda _: "\n".join(allowed_tags)),
            }
            | prompt
            | llm
        )
        raw = chain.invoke({})
        payload = raw.content if hasattr(raw, "content") else raw
        try:
            text_payload = payload.strip()
            if text_payload.startswith("```"):
                import re as _re
                m = _re.search(r"```(?:json)?\s*(.*?)```", text_payload, _re.DOTALL)
                if m:
                    text_payload = m.group(1).strip()
            data = json.loads(text_payload)
            inventories[specie] = data if isinstance(data, list) else []
        except Exception:
            print(f"Debug: no valid JSON in inventory for {specie}")
            inventories[specie] = []
        # persist inventory per species
        with open(traits_dir / f"{slugify(specie)}.json", "w", encoding="utf-8") as f:
            json.dump(inventories[specie], f, ensure_ascii=False, indent=2)

    # Stage 2: synthesis
    synth_prompt = PromptTemplate(
        input_variables=["inventories"],
        template=pathlib.Path(SYNTHESIS_PROMPT_FILE).read_text(encoding="utf-8"),
        template_format="jinja2",
    )
    synth_chain = {"inventories": RunnablePassthrough()} | synth_prompt | llm
    raw_answer = synth_chain.invoke(json.dumps(inventories, ensure_ascii=False, indent=2))
    payload = raw_answer.content if hasattr(raw_answer, "content") else raw_answer
    try:
        text_payload = payload.strip()
        if text_payload.startswith("```"):
            import re as _re
            m = _re.search(r"```(?:json)?\s*(.*?)```", text_payload, _re.DOTALL)
            if m:
                text_payload = m.group(1).strip()
        if text_payload.strip().lower() == "no trait found":
            print("No trait found")
            return
        data = json.loads(text_payload)
        bullets = data if isinstance(data, list) else data.get("bullets", [])
    except Exception:
        print("Debug: no valid JSON in synthesis output")
        bullets = []

    allowed_tags_all = set()
    for inv in inventories.values():
        for item in inv:
            for tag in item.get("sources", []):
                allowed_tags_all.add(tag)

    out_lines = []
    for item in bullets:
        if isinstance(item, str):
            out_lines.append(item)
            continue
        trait = (item.get("trait") or "").strip()
        citations = item.get("citations") or item.get("combined_sources") or []
        citations = [c for c in citations if c in allowed_tags_all]
        if trait and citations:
            out_lines.append(f"- {trait} " + " ".join(citations))
    if not out_lines:
        print("No trait found")
    else:
        print("\n".join(out_lines))


def load_species_file(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Species file must contain a list of mappings")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run two-stage RAG pipeline.")
    parser.add_argument("--species-file", help="Path to JSON file with canonical/aliases mappings.")
    parser.add_argument("--question", help="Optional question / query string. If omitted, derived from species-file canonical list or fallback to default.")
    parser.add_argument("--log-run", action="store_true", help="Enable logging (uses RAG logging).")
    args = parser.parse_args()

    species_groups = load_species_file(args.species_file) if args.species_file else None
    run_two_stage(args.question, species_groups=species_groups, log_runs=args.log_run)
