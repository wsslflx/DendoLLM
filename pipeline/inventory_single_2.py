#!/usr/bin/env python3
"""
Run a single per-species inventory prompt and print JSON traits to stdout.
Uses prompt_inventory_json_cited.txt and the existing RAG ingestion/retrieval utilities.
"""

from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).parents[1]))

import argparse
import hashlib
import json
import math
import pathlib
import re
from contextlib import contextmanager, nullcontext
from datetime import datetime

import numpy as np
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from core.llm_backend import make_chat_llm
from core.rag_cli import RAG, maximal_marginal_relevance

try:
    import fcntl
except ImportError:  # pragma: no cover - only relevant on non-POSIX platforms
    fcntl = None  # type: ignore[assignment]

PROMPT_FILE = "Prompts/prompt_inventory_5.txt"
SYNTHESIS_PROMPT_FILE = "Prompts/prompt_synthesis_5.txt"
MAX_SYNTHESIS_RETRIES = 3


def _extract_json_text(payload: object) -> str:
    text_payload = str(payload).strip()
    if text_payload.startswith("```"):
        match = re.search(r"```(?:json)?\s*(.*?)```", text_payload, re.DOTALL | re.IGNORECASE)
        if match:
            text_payload = match.group(1).strip()
    return text_payload


def _validate_synthesis_json(data: object) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Synthesis output must be a JSON object")
    required = ("strict_common_traits", "subgroup_common_traits", "mechanism_hypotheses")
    out: dict[str, list[dict]] = {}
    for key in required:
        value = data.get(key)
        if not isinstance(value, list):
            raise ValueError(f"Synthesis output field '{key}' must be a list")
        cleaned: list[dict] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            trait = item.get("trait")
            sources = item.get("sources")
            confidence = item.get("confidence")
            if not isinstance(trait, str) or not trait.strip():
                continue
            if not isinstance(sources, list):
                sources = []
            clean_sources = [str(s) for s in sources if isinstance(s, str) and s.strip()]
            try:
                conf = float(confidence)
            except Exception:
                conf = 0.0
            cleaned.append(
                {
                    "trait": trait.strip(),
                    "sources": clean_sources,
                    "confidence": max(0.0, min(1.0, conf)),
                }
            )
        out[key] = cleaned
    return out


def slugify(name: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in name.lower()).strip("_")


def init_log_dir(
    log_runs: bool, log_root: pathlib.Path | None = None, subdir: str | None = None
) -> pathlib.Path | None:
    if not log_runs:
        return None
    base = pathlib.Path(log_root) if log_root else pathlib.Path("logs")
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
    run_dir = pathlib.Path("logs") / f"{timestamp}-{slugify(run_label)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


@contextmanager
def ingest_lock(lock_path: pathlib.Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as handle:
        if fcntl is None:
            print(f"[Ingest lock] fcntl unavailable, continuing without lock: {lock_path}")
            yield
            return
        print(f"[Ingest lock] waiting: {lock_path}")
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        print(f"[Ingest lock] acquired: {lock_path}")
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            print(f"[Ingest lock] released: {lock_path}")


def per_species_retrieve(rag: RAG, query1: str, query2: str, specie: str):
    """Retrieve diversified chunks for a single species."""
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
        if key in seen:
            if score < seen[key][1]:
                seen[key] = (doc, score)
            continue
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
    best_score = sorted_by_score[0][1]
    cutoff = best_score + getattr(rag, "score_margin", 0.3)
    kept = [t for t in sorted_by_score if t[1] <= cutoff]
    if not kept:
        kept = sorted_by_score[:1]
    docs = []
    per_doc_counts: dict[str, int] = {}
    for doc, score in kept:
        if rag.threshold is None or score <= rag.threshold:
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


def ingest_species(
    rag: RAG,
    canonical: str,
    aliases: list[str],
    pdf_dir: pathlib.Path | None = None,
) -> None:
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
    pdf_location = str(pdf_dir) if pdf_dir else "./pdfs"
    for term in search_terms:
        papers = rag.fetch_and_prepare(query=term, specie_norm=canonical_norm, location=pdf_location)
        for paper_entry in papers:
            pdf_path = paper_entry["pdf_path"]
            paper_meta = paper_entry.get("paper") or {}
            if "doc_id" not in paper_meta:
                paper_meta["doc_id"] = paper_entry.get("doc_id")
            rag.load_ocr(pdf_path=str(pdf_path), puppy=canonical_norm, paper_meta=paper_meta)
            rag.ingested_per_species[canonical_norm].add(str(pdf_path))


def run_inventory(
    specie: str,
    aliases: list[str],
    log_runs: bool = False,
    reuse_traits: bool = False,
    log_root: pathlib.Path | None = None,
    skip_ingest: bool = False,
    traits_dir: pathlib.Path | None = None,
    pdf_dir: pathlib.Path | None = None,
    chroma_dir: pathlib.Path | None = None,
    ingest_lock_file: pathlib.Path | None = None,
) -> list[dict]:
    traits_root = traits_dir if traits_dir else pathlib.Path("traits")
    traits_root.mkdir(parents=True, exist_ok=True)
    out_path = traits_root / f"{slugify(specie)}.json"
    if reuse_traits and out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            traits = json.load(f)
        print(json.dumps(traits, ensure_ascii=False, indent=2))
        return traits if isinstance(traits, list) else []

    rag = RAG(
        log_runs=log_runs,
        persist_dir=str(chroma_dir) if chroma_dir else "./chroma_store_ollama",
    )
    if not skip_ingest:
        lock_ctx = ingest_lock(ingest_lock_file) if ingest_lock_file else nullcontext()
        with lock_ctx:
            ingest_species(rag, canonical=specie, aliases=aliases, pdf_dir=pdf_dir)

    specie_norm = specie.lower().strip()
    query1 = f"{specie} morphology behavior ecology sensory phenotype"
    query2 = (
        f"{specie} natural history trait field observations habitat use diet "
        "foraging locomotion coloration morphometric"
    )
    docs = per_species_retrieve(rag, query1, query2, specie_norm)
    ingested_docs = sorted(rag.ingested_per_species.get(specie_norm, set()))
    ingested = len(ingested_docs)
    print(f"[Inventory] {specie}: {ingested} ingested / {len(docs)} used")
    if not docs:
        print("No relevant documents found.")
        return []

    log_dir = init_log_dir(log_runs, log_root=log_root, subdir=slugify(specie))
    context_text = format_docs(docs)
    wiki_ingested = [s for s in ingested_docs if str(s).startswith("wikipedia:")]
    papers_ingested = [s for s in ingested_docs if not str(s).startswith("wikipedia:")]
    wiki_used = [doc for doc in docs if str((doc.metadata or {}).get("source_path", "")).startswith("wikipedia:")]
    print(
        f"[Wiki] {specie}: ingested={'yes' if wiki_ingested else 'no'} "
        f"used={'yes' if wiki_used else 'no'} "
        f"(papers ingested={len(papers_ingested)}; chunks used={len(docs)}; wiki chunks used={len(wiki_used)})"
    )
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
        with open(log_dir / "source_usage.txt", "w", encoding="utf-8") as f:
            f.write(f"Species: {specie}\n")
            f.write(f"Wikipedia ingested: {'yes' if wiki_ingested else 'no'}\n")
            f.write(f"Wikipedia used: {'yes' if wiki_used else 'no'}\n")
            f.write(f"Papers ingested: {len(papers_ingested)}\n")
            f.write(f"Chunks used: {len(docs)}\n")
            f.write(f"Wikipedia chunks used: {len(wiki_used)}\n")
    prompt = PromptTemplate(
        input_variables=["context", "species_name"],
        template=pathlib.Path(PROMPT_FILE).read_text(encoding="utf-8"),
    )
    formatted_prompt = prompt.format(context=context_text, species_name=specie)
    if log_dir:
        with open(log_dir / "prompt.txt", "w", encoding="utf-8") as f:
            f.write(formatted_prompt)
    llm = make_chat_llm(model=None, temperature=0.0)
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


def run_synthesis(
    inventories: dict[str, list[dict]], log_runs: bool = False, log_root: pathlib.Path | None = None
) -> None:
    prompt = PromptTemplate(
        input_variables=["extracted_species_lists"],
        template=pathlib.Path(SYNTHESIS_PROMPT_FILE).read_text(encoding="utf-8"),
    )
    llm = make_chat_llm(model=None, temperature=0.0, format="json")
    chain = {"extracted_species_lists": RunnablePassthrough()} | prompt | llm
    inventories_json = json.dumps(inventories, ensure_ascii=False, indent=2)
    parsed_output: dict | None = None
    raw_attempts: list[str] = []
    for attempt in range(1, MAX_SYNTHESIS_RETRIES + 1):
        payload = chain.invoke(inventories_json)
        content = payload.content if hasattr(payload, "content") else payload
        raw_text = str(content)
        raw_attempts.append(raw_text)
        try:
            parsed = json.loads(_extract_json_text(content))
            parsed_output = _validate_synthesis_json(parsed)
            break
        except Exception as exc:
            print(f"Debug: invalid synthesis JSON on attempt {attempt}/{MAX_SYNTHESIS_RETRIES}: {exc}")
            if attempt == MAX_SYNTHESIS_RETRIES:
                raise RuntimeError("Synthesis failed: model did not return valid JSON after retries") from exc
    assert parsed_output is not None

    content = json.dumps(parsed_output, ensure_ascii=False, indent=2)
    log_dir = init_log_dir(log_runs, log_root=log_root, subdir="synthesis")
    if log_dir:
        with open(log_dir / "synthesis_prompt.txt", "w", encoding="utf-8") as f:
            f.write(prompt.format(extracted_species_lists=inventories_json))
        with open(log_dir / "synthesis_answer.txt", "w", encoding="utf-8") as f:
            f.write(content)
        with open(log_dir / "synthesis_raw_attempts.txt", "w", encoding="utf-8") as f:
            f.write("\n\n--- attempt ---\n\n".join(raw_attempts))
    print(content)


def main():
    parser = argparse.ArgumentParser(description="Run per-species inventory prompt and print JSON traits.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--species", help="Canonical species name to process.")
    group.add_argument("--species-file", help="Path to JSON file with canonical/aliases mappings.")
    parser.add_argument("--aliases", help="Comma-separated aliases to use for ingestion/search.")
    parser.add_argument("--log-run", action="store_true", help="Log prompt and answer to logs/<timestamp>/.")
    parser.add_argument("--log-dir", help="Explicit directory for run logs.")
    parser.add_argument("--traits-dir", help="Directory to write per-species traits JSON files.")
    parser.add_argument("--pdf-dir", help="Directory to store downloaded PDFs.")
    parser.add_argument("--chroma-dir", help="Directory for persisted Chroma vectorstore.")
    parser.add_argument(
        "--ingest-lock-file",
        default=".ingest.lock",
        help="Global lock file path to serialize ingestion across parallel processes.",
    )
    parser.add_argument(
        "--reuse-traits",
        action="store_true",
        help="Reuse existing traits/<species>.json if present (skip ingestion and prompting).",
    )
    parser.add_argument(
        "--skip-ingest",
        action="store_true",
        help="Skip ingestion/download; use existing vectorstore content only.",
    )
    args = parser.parse_args()

    run_log_dir = None
    explicit_log_dir = pathlib.Path(args.log_dir) if args.log_dir else None
    traits_dir = pathlib.Path(args.traits_dir) if args.traits_dir else None
    pdf_dir = pathlib.Path(args.pdf_dir) if args.pdf_dir else None
    chroma_dir = pathlib.Path(args.chroma_dir) if args.chroma_dir else None
    ingest_lock_file = pathlib.Path(args.ingest_lock_file) if args.ingest_lock_file else None
    if args.log_run:
        run_label = pathlib.Path(args.species_file).stem if args.species_file else args.species.strip()
        run_log_dir = init_run_log_dir(True, run_label, explicit_log_dir=explicit_log_dir)

    if args.species_file:
        species_groups = load_species_file(args.species_file)
        inventories: dict[str, list[dict]] = {}
        for entry in species_groups:
            canonical = (entry.get("canonical") or "").strip()
            aliases = [a.strip() for a in entry.get("aliases", []) if a.strip()]
            if canonical:
                inventories[canonical] = run_inventory(
                    canonical,
                    aliases,
                    log_runs=args.log_run,
                    reuse_traits=args.reuse_traits,
                    log_root=run_log_dir,
                    skip_ingest=args.skip_ingest,
                    traits_dir=traits_dir,
                    pdf_dir=pdf_dir,
                    chroma_dir=chroma_dir,
                    ingest_lock_file=ingest_lock_file,
                )
        if inventories:
            run_synthesis(inventories, log_runs=args.log_run, log_root=run_log_dir)
        return

    aliases = [a.strip() for a in (args.aliases or "").split(",") if a.strip()]
    run_inventory(
        args.species.strip(),
        aliases,
        log_runs=args.log_run,
        reuse_traits=args.reuse_traits,
        log_root=run_log_dir,
        skip_ingest=args.skip_ingest,
        traits_dir=traits_dir,
        pdf_dir=pdf_dir,
        chroma_dir=chroma_dir,
        ingest_lock_file=ingest_lock_file,
    )


if __name__ == "__main__":
    main()
