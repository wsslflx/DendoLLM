#!/usr/bin/env python3
"""
Run a single per-species inventory prompt and print JSON traits to stdout.
Uses prompt_inventory_json_cited.txt and the existing RAG ingestion/retrieval utilities.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from datetime import datetime

import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI

from rag_cli import RAG, maximal_marginal_relevance

PROMPT_FILE = "Prompts/prompt_inventory_json_cited.txt"
SYNTHESIS_PROMPT_FILE = "Prompts/prompt_synthesis_common.txt"


def slugify(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def init_log_dir(log_runs: bool, log_root: str = "./logs") -> pathlib.Path | None:
    if not log_runs:
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = pathlib.Path(log_root) / timestamp
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def per_species_retrieve(rag: RAG, query: str, specie: str):
    """Retrieve diversified chunks for a single species."""
    candidates = rag.vectorstore.similarity_search_with_score(
        query, k=rag.per_species_fetch_k, filter={"specie": specie}
    )
    if not candidates:
        return []
    try:
        query_embedding = np.array(
            rag.vectorstore._embedding_function.embed_query(query), dtype=float  # type: ignore[attr-defined]
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


def ingest_species(rag: RAG, canonical: str, aliases: list[str]) -> None:
    """Ingest Wikipedia, PMC, and OpenAlex for one species and its aliases."""
    search_terms = [canonical] + [a for a in aliases if a]
    canonical_norm = canonical.lower().strip()

    # Wikipedia
    wiki_ingested = rag.ingest_wikipedia(title=canonical, specie_norm=canonical_norm)
    if not wiki_ingested:
        for term in search_terms:
            if term == canonical:
                continue
            if rag.ingest_wikipedia(title=term, specie_norm=canonical_norm):
                break

    # PMC
    rag.ingest_pmc_texts(query=canonical, specie_norm=canonical_norm)

    # OpenAlex PDFs/OA
    for term in search_terms:
        papers = rag.fetch_and_prepare(query=term, specie_norm=canonical_norm)
        for paper_entry in papers:
            pdf_path = paper_entry["pdf_path"]
            paper_meta = paper_entry.get("paper") or {}
            if "doc_id" not in paper_meta:
                paper_meta["doc_id"] = paper_entry.get("doc_id")
            rag.load_ocr(pdf_path=str(pdf_path), puppy=canonical_norm, paper_meta=paper_meta)
            rag.ingested_per_species[canonical_norm].add(str(pdf_path))


def run_inventory(specie: str, aliases: list[str], log_runs: bool = False, reuse_traits: bool = False) -> list[dict]:
    traits_dir = pathlib.Path("traits")
    traits_dir.mkdir(parents=True, exist_ok=True)
    out_path = traits_dir / f"{slugify(specie)}.json"
    if reuse_traits and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            traits = json.load(f)
        print(json.dumps(traits, ensure_ascii=False, indent=2))
        return traits if isinstance(traits, list) else []

    rag = RAG(log_runs=log_runs)
    ingest_species(rag, canonical=specie, aliases=aliases)

    docs = per_species_retrieve(rag, specie, specie.lower().strip())
    ingested = len(rag.ingested_per_species.get(specie.lower().strip(), set()))
    print(f"[Inventory] {specie}: {ingested} ingested / {len(docs)} used")
    if not docs:
        print("No relevant documents found.")
        return []

    log_dir = init_log_dir(log_runs)
    context_text = format_docs(docs)
    prompt = PromptTemplate(
        input_variables=["context", "species_name"],
        template=pathlib.Path(PROMPT_FILE).read_text(encoding="utf-8"),
    )
    formatted_prompt = prompt.format(context=context_text, species_name=specie)
    if log_dir:
        with open(log_dir / "prompt.txt", "w", encoding="utf-8") as f:
            f.write(formatted_prompt)
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
    chain = (
        {
            "context": RunnableLambda(lambda _: context_text),
            "species_name": RunnableLambda(lambda _: specie),
        }
        | prompt
        | llm
    )
    raw = chain.invoke({})
    payload = raw.content if hasattr(raw, "content") else raw
    if log_dir:
        with open(log_dir / "answer.txt", "w", encoding="utf-8") as f:
            f.write(str(payload))
    try:
        text_payload = str(payload).strip()
        if text_payload.startswith("```"):
            import re as _re
            m = _re.search(r"```(?:json)?\s*(.*?)```", text_payload, _re.DOTALL)
            if m:
                text_payload = m.group(1).strip()
        data = json.loads(text_payload)
        traits = data if isinstance(data, list) else []
    except Exception:
        print("Debug: no valid JSON in model output")
        traits = []
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(traits, f, ensure_ascii=False, indent=2)
    print(json.dumps(traits, ensure_ascii=False, indent=2))
    return traits


def load_species_file(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Species file must contain a list of mappings")
    return data


def run_synthesis(inventories: dict[str, list[dict]], log_runs: bool = False) -> None:
    prompt = PromptTemplate(
        input_variables=["extracted_species_lists"],
        template=pathlib.Path(SYNTHESIS_PROMPT_FILE).read_text(encoding="utf-8"),
    )
    llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.0)
    chain = {"extracted_species_lists": RunnablePassthrough()} | prompt | llm
    payload = chain.invoke(json.dumps(inventories, ensure_ascii=False, indent=2))
    content = payload.content if hasattr(payload, "content") else payload
    if log_runs:
        log_dir = init_log_dir(log_runs)
        if log_dir:
            with open(log_dir / "synthesis_prompt.txt", "w", encoding="utf-8") as f:
                f.write(prompt.format(extracted_species_lists=json.dumps(inventories, ensure_ascii=False, indent=2)))
            with open(log_dir / "synthesis_answer.txt", "w", encoding="utf-8") as f:
                f.write(str(content))
    print(content)


def main():
    parser = argparse.ArgumentParser(description="Run per-species inventory prompt and print JSON traits.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--species", help="Canonical species name to process.")
    group.add_argument("--species-file", help="Path to JSON file with canonical/aliases mappings.")
    parser.add_argument("--aliases", help="Comma-separated aliases to use for ingestion/search.")
    parser.add_argument("--log-run", action="store_true", help="Log prompt and answer to logs/<timestamp>/.")
    parser.add_argument(
        "--reuse-traits",
        action="store_true",
        help="Reuse existing traits/<species>.json if present (skip ingestion and prompting).",
    )
    args = parser.parse_args()

    if args.species_file:
        species_groups = load_species_file(args.species_file)
        inventories: dict[str, list[dict]] = {}
        for entry in species_groups:
            canonical = (entry.get("canonical") or "").strip()
            aliases = [a.strip() for a in entry.get("aliases", []) if a.strip()]
            if canonical:
                inventories[canonical] = run_inventory(
                    canonical, aliases, log_runs=args.log_run, reuse_traits=args.reuse_traits
                )
        if inventories:
            run_synthesis(inventories, log_runs=args.log_run)
        return

    aliases = [a.strip() for a in (args.aliases or "").split(",") if a.strip()]
    run_inventory(args.species.strip(), aliases, log_runs=args.log_run, reuse_traits=args.reuse_traits)


if __name__ == "__main__":
    main()
