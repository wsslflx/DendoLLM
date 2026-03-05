#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
from datetime import datetime

import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

from rag_cli import RAG, maximal_marginal_relevance

PROMPT_FILE = "Prompts/prompt_inventory_v3.txt"


def slugify(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def init_log_dir(
    log_runs: bool,
    log_root: pathlib.Path | None = None,
    subdir: str | None = None,
) -> pathlib.Path | None:
    if not log_runs:
        return None
    base = pathlib.Path(log_root) if log_root else pathlib.Path("logs")
    path = base / subdir if subdir else base
    path.mkdir(parents=True, exist_ok=True)
    return path


def init_run_log_dir(log_runs: bool, run_label: str, explicit_log_dir: pathlib.Path | None = None) -> pathlib.Path | None:
    if not log_runs:
        return None
    if explicit_log_dir is not None:
        explicit_log_dir.mkdir(parents=True, exist_ok=True)
        return explicit_log_dir
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = pathlib.Path("logs") / f"{timestamp}-{slugify(run_label)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def per_species_retrieve(rag: RAG, query1: str, query2: str, specie: str) -> list:
    candidates_q1 = rag.vectorstore.similarity_search_with_score(
        query1, k=rag.per_species_fetch_k, filter={"specie": specie}
    )
    candidates_q2 = rag.vectorstore.similarity_search_with_score(
        query2, k=rag.per_species_fetch_k, filter={"specie": specie}
    )
    if not candidates_q1 and not candidates_q2:
        return []

    def dedupe_key(doc) -> tuple:
        meta = doc.metadata or {}
        doc_id = meta.get("doc_id")
        chunk_index = meta.get("chunk_index")
        source_path = meta.get("source_path")
        if doc_id is not None and chunk_index is not None:
            return ("doc_id", doc_id, chunk_index)
        if source_path is not None and chunk_index is not None:
            return ("source_path", source_path, chunk_index)
        content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()
        return ("content_hash", content_hash)

    seen: dict[tuple, tuple] = {}
    for doc, score in candidates_q1 + candidates_q2:
        key = dedupe_key(doc)
        current = seen.get(key)
        if current is None or score < current[1]:
            seen[key] = (doc, score)

    candidates = list(seen.values())
    if not candidates:
        return []

    try:
        concat_query = f"{query1} {query2}".strip()
        query_embedding = np.array(
            rag.vectorstore._embedding_function.embed_query(concat_query), dtype=float  # type: ignore[attr-defined]
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
    per_doc_counts: dict[str, int] = {}
    for doc, score in kept:
        if rag.threshold is not None and score > rag.threshold:
            continue
        meta = doc.metadata or {}
        doc_key = str(meta.get("doc_id") or meta.get("source_path") or "unknown")
        if per_doc_counts.get(doc_key, 0) >= 10:
            continue
        per_doc_counts[doc_key] = per_doc_counts.get(doc_key, 0) + 1
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
    search_terms = [canonical] + [a for a in aliases if a]
    canonical_norm = canonical.lower().strip()

    wiki_ingested = rag.ingest_wikipedia(title=canonical, specie_norm=canonical_norm)
    if not wiki_ingested:
        for term in search_terms:
            if term == canonical:
                continue
            if rag.ingest_wikipedia(title=term, specie_norm=canonical_norm):
                break

    rag.ingest_pmc_texts(query=canonical, specie_norm=canonical_norm)

    for term in search_terms:
        papers = rag.fetch_and_prepare(query=term, specie_norm=canonical_norm)
        for paper_entry in papers:
            pdf_path = paper_entry["pdf_path"]
            paper_meta = paper_entry.get("paper") or {}
            if "doc_id" not in paper_meta:
                paper_meta["doc_id"] = paper_entry.get("doc_id")
            rag.load_ocr(pdf_path=str(pdf_path), puppy=canonical_norm, paper_meta=paper_meta)
            rag.ingested_per_species[canonical_norm].add(str(pdf_path))


def load_species_file(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Species file must contain a list of mappings")
    return data


def run_inventory(
    specie: str,
    aliases: list[str],
    log_runs: bool = False,
    reuse_traits: bool = False,
    log_root: pathlib.Path | None = None,
    skip_ingest: bool = False,
) -> list[dict]:
    traits_dir = pathlib.Path("traits")
    traits_dir.mkdir(parents=True, exist_ok=True)
    out_path = traits_dir / f"{slugify(specie)}.json"
    if reuse_traits and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            traits = json.load(f)
        log_dir = init_log_dir(log_runs, log_root=log_root, subdir=slugify(specie))
        if log_dir:
            with open(log_dir / "answer.txt", "w", encoding="utf-8") as f:
                f.write(json.dumps(traits, ensure_ascii=False, indent=2))
        print(json.dumps(traits, ensure_ascii=False, indent=2))
        return traits if isinstance(traits, list) else []

    rag = RAG(log_runs=log_runs)
    if not skip_ingest:
        ingest_species(rag, canonical=specie, aliases=aliases)

    specie_norm = specie.lower().strip()
    query1 = f"{specie} morphology behavior ecology sensory phenotype"
    query2 = (
        f"{specie} natural history trait field observations habitat use diet "
        "foraging locomotion coloration morphometric"
    )
    docs = per_species_retrieve(rag, query1, query2, specie_norm)
    ingested_docs = sorted(rag.ingested_per_species.get(specie_norm, set()))
    print(f"[Inventory] {specie}: {len(ingested_docs)} ingested / {len(docs)} used")
    if not docs:
        print("No relevant documents found.")
        return []

    log_dir = init_log_dir(log_runs, log_root=log_root, subdir=slugify(specie))
    context_text = format_docs(docs)

    if log_dir:
        with open(log_dir / "ingested_docs.json", "w", encoding="utf-8") as f:
            json.dump(ingested_docs, f, ensure_ascii=False, indent=2)
        used_chunks = []
        for doc in docs:
            meta = doc.metadata or {}
            used_chunks.append(
                {
                    "doc_id": meta.get("doc_id"),
                    "chunk_index": meta.get("chunk_index"),
                    "source_path": meta.get("source_path"),
                }
            )
        with open(log_dir / "used_chunks.json", "w", encoding="utf-8") as f:
            json.dump(used_chunks, f, ensure_ascii=False, indent=2)

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

            match = _re.search(r"```(?:json)?\s*(.*?)```", text_payload, _re.DOTALL)
            if match:
                text_payload = match.group(1).strip()
        data = json.loads(text_payload)
        traits = data if isinstance(data, list) else []
    except Exception:
        print("Debug: no valid JSON in model output")
        traits = []

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(traits, f, ensure_ascii=False, indent=2)
    print(json.dumps(traits, ensure_ascii=False, indent=2))
    return traits


def main() -> None:
    parser = argparse.ArgumentParser(description="Run per-species inventory prompt and print JSON traits.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--species", help="Canonical species name to process.")
    group.add_argument("--species-file", help="Path to JSON file with canonical/aliases mappings.")
    parser.add_argument("--aliases", help="Comma-separated aliases to use for ingestion/search.")
    parser.add_argument("--log-run", action="store_true", help="Write logs for each species.")
    parser.add_argument("--log-dir", help="Explicit directory for the run log root.")
    parser.add_argument("--reuse-traits", action="store_true", help="Reuse existing traits/<species>.json if present.")
    parser.add_argument("--skip-ingest", action="store_true", help="Skip ingestion/download and reuse vectorstore.")
    args = parser.parse_args()

    explicit_log_dir = pathlib.Path(args.log_dir) if args.log_dir else None
    run_log_dir = None
    if args.log_run:
        run_label = pathlib.Path(args.species_file).stem if args.species_file else args.species.strip()
        run_log_dir = init_run_log_dir(True, run_label, explicit_log_dir=explicit_log_dir)

    if args.species_file:
        species_groups = load_species_file(args.species_file)
        for entry in species_groups:
            canonical = (entry.get("canonical") or "").strip()
            aliases = [a.strip() for a in entry.get("aliases", []) if a.strip()]
            if not canonical:
                continue
            run_inventory(
                canonical,
                aliases,
                log_runs=args.log_run,
                reuse_traits=args.reuse_traits,
                log_root=run_log_dir,
                skip_ingest=args.skip_ingest,
            )
        return

    aliases = [a.strip() for a in (args.aliases or "").split(",") if a.strip()]
    run_inventory(
        args.species.strip(),
        aliases,
        log_runs=args.log_run,
        reuse_traits=args.reuse_traits,
        log_root=run_log_dir,
        skip_ingest=args.skip_ingest,
    )


if __name__ == "__main__":
    main()
