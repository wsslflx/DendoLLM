#!/usr/bin/env python3
"""
Command-line entry point mirroring the RAG notebook pipeline.
Edit DEFAULT_QUERY below to target different species.
"""

from __future__ import annotations

import os
import pathlib
import xml.etree.ElementTree as ET
import re
import json
import hashlib
import argparse
import random
import time
from collections import defaultdict
from typing import Optional
import math
import numpy as np
from datetime import datetime

import requests
from langchain_core.documents import Document
from dotenv import load_dotenv
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ImportError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from PyPDF2 import PdfReader 
from PyPDF2.errors import PdfReadError
from pydantic import BaseModel, constr
try:
    from langchain_community.vectorstores.utils import maximal_marginal_relevance
except ImportError:  # fallback for older langchain versions
    from langchain.vectorstores.utils import maximal_marginal_relevance

from core.llm_backend import make_chat_llm, make_embeddings

load_dotenv()

EMAIL = os.getenv("EMAIL", "trifonova.kate.s@gmail.com")
BASE_URL = "https://api.openalex.org/works"
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DEFAULT_QUERY = "Cape golden mole, star-nosed mole, naked mole-rat, blind mole-rat"
PROMPT_FILE = "prompt3.txt"
MMR_QUERY_FILE = "Prompts/mmr_query.txt"

# when False, keep bullets even if citations don't match allowed_tags
STRICT_CITATION_FILTER = False


_RETRY_BUDGET_S = 30.0  # max total seconds spent on retries per request

_REF_HEADER_RE = re.compile(
    r'\n\s*(References|Bibliography|Literature\s+Cited|Works\s+Cited|Acknowledgements?)\s*\n',
    re.IGNORECASE,
)

def _strip_references(text: str) -> str:
    """Truncate text at the references/bibliography section header."""
    m = _REF_HEADER_RE.search(text)
    if m:
        truncated = text[:m.start()]
        removed_pct = round(100 * (len(text) - len(truncated)) / len(text))
        print(f"[RAG] Stripped references section ({removed_pct}% of document removed).")
        return truncated
    return text


def _apply_chunk_cap(splits: list, ratio: float, source_label: str) -> list:
    """Keep first ratio% and last ratio% of chunks, dropping the middle."""
    n = len(splits)
    keep = max(1, round(n * ratio))
    if 2 * keep >= n:
        return splits  # cap would keep everything — skip
    first = splits[:keep]
    last = splits[n - keep:]
    kept = first + last
    print(f"[RAG] Chunk cap ({ratio:.0%} first+last): {n} → {len(kept)} chunks ({source_label})")
    # re-index chunk_index so downstream sees contiguous indices
    for i, doc in enumerate(kept):
        doc.metadata["chunk_index"] = i
    return kept


def _http_get_with_retry(
    url: str,
    *,
    params: dict,
    timeout: int,
    headers: Optional[dict] = None,
    max_retries: int = 6,
    label: str = "HTTP request",
) -> requests.Response:
    budget = _RETRY_BUDGET_S
    attempt = 0
    last_exc: Optional[Exception] = None

    while True:
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as exc:
            last_exc = exc
            delay = min(1.5 * (2 ** attempt), budget)
            if delay <= 0:
                print(f"{label} failed ({exc}); retry budget exhausted, skipping")
                raise
            print(f"{label} failed ({exc}); retrying in {delay:.1f}s (budget left: {budget:.1f}s)")
            time.sleep(delay)
            budget -= delay
            attempt += 1
            continue

        status = resp.status_code
        if status < 400:
            return resp

        retryable = status == 429 or 500 <= status < 600
        if not retryable:
            resp.raise_for_status()

        # determine how long we'd need to sleep before retrying
        retry_after_hdr = resp.headers.get("Retry-After")
        if retry_after_hdr:
            try:
                delay = float(retry_after_hdr)
            except ValueError:
                delay = 1.5 * (2 ** attempt)
        else:
            delay = 1.5 * (2 ** attempt)
        delay = min(delay, budget)

        if delay <= 0:
            print(f"{label} returned HTTP {status} (Retry-After: {retry_after_hdr}); "
                  f"retry budget exhausted, skipping")
            resp.raise_for_status()

        print(f"{label} returned HTTP {status}; retrying in {delay:.1f}s "
              f"(budget left: {budget:.1f}s)")
        time.sleep(delay)
        budget -= delay
        attempt += 1


def load_mmr_query_template() -> str:
    path = pathlib.Path(MMR_QUERY_FILE)
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return "{species_name}"


MMR_QUERY_TEMPLATE = load_mmr_query_template()


def canonical_doc_id(paper: dict) -> Optional[str]:
    """Derive a stable doc identifier for deduplication."""
    doi = paper.get("doi")
    if doi:
        return f"doi:{doi.lower()}"
    paper_id = paper.get("id")
    if paper_id:
        return f"oa:{paper_id}"
    title = paper.get("title")
    if title:
        norm = re.sub(r"\s+", " ", title.strip().lower())
        title_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()
        return f"title:{title_hash}"
    return None


def works_with_oa(query: str, first_n: int = 1) -> list[dict]:
    """Query OpenAlex for OA works whose abstract mentions the taxon."""
    params = {
        "filter": f'abstract.search:"{query}",best_open_version:published',
        "per-page": first_n,
        # request publication year for downstream citations
        "select": "id,title,publication_year,doi,best_oa_location,primary_location,locations",
        "mailto": EMAIL,
    }
    try:
        resp = _http_get_with_retry(
            BASE_URL,
            params=params,
            timeout=60,
            max_retries=6,
            label=f"OpenAlex works ({query})",
        )
        data = resp.json()
    except Exception as exc:
        print(f"OpenAlex query failed for '{query}', skipping OpenAlex retrieval: {exc}")
        return []

    if not isinstance(data, dict):
        print(f"OpenAlex returned non-dict payload for '{query}', skipping")
        return []

    papers = data.get("results")
    if not isinstance(papers, list):
        error_msg = data.get("error") if isinstance(data.get("error"), str) else "missing 'results'"
        print(f"OpenAlex payload issue for '{query}', skipping: {error_msg}")
        return []
    return [p for p in papers if isinstance(p, dict)]


def pmc_search(query: str, first_n: int = 3) -> list[str]:
    """Find PMC IDs for the query."""
    params = {
        "db": "pmc",
        "term": query,
        "retmax": first_n,
        "retmode": "json",
        "email": EMAIL,
    }
    resp = _http_get_with_retry(
        f"{NCBI_BASE}/esearch.fcgi",
        params=params,
        timeout=60,
        max_retries=6,
        label=f"PMC esearch ({query})",
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("esearchresult", {}).get("idlist", [])


def pmc_fetch_fulltext(pmcid: str) -> Optional[str]:
    """Fetch full-text XML for a PMCID and return concatenated text from the body."""
    params = {
        "db": "pmc",
        "id": pmcid,
        "retmode": "xml",
    }
    resp = _http_get_with_retry(
        f"{NCBI_BASE}/efetch.fcgi",
        params=params,
        timeout=60,
        max_retries=6,
        label=f"PMC efetch ({pmcid})",
    )
    resp.raise_for_status()
    # basic content-type guard
    if "xml" not in resp.headers.get("Content-Type", "").lower():
        return None
    try:
        root = ET.fromstring(resp.content)
    except ET.ParseError:
        return None
    # pull all text within the <body> element
    body_elems = root.findall(".//body")
    texts = []
    for body in body_elems:
        texts.append(" ".join(body.itertext()))
    full_text = " ".join(texts).strip()
    return full_text or None

def wikipedia_fetch_plain(species_name: str, lang: str = "en") -> Optional[str]:
    """Search Wikipedia for a best title match and fetch full article plaintext."""
    api_url = f"https://{lang}.wikipedia.org/w/api.php"
    headers = {
        "User-Agent": f"SMTB2025/0.1 (mailto:{EMAIL})",
        "From": EMAIL,
    }

    def extract_by_pageid(pageid: int) -> Optional[str]:
        extract_params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "pageids": pageid,
        }
        resp = requests.get(api_url, params=extract_params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract")
            if extract:
                return extract.strip()
        return None

    def extract_by_title(title: str) -> Optional[str]:
        params = {
            "action": "query",
            "format": "json",
            "prop": "extracts",
            "explaintext": 1,
            "redirects": 1,
            "titles": title,
        }
        resp = requests.get(api_url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        pages = data.get("query", {}).get("pages", {})
        for page in pages.values():
            extract = page.get("extract")
            if extract:
                return extract.strip()
        return None

    def pick_best_hit(hits: list[dict]) -> Optional[dict]:
        if not hits:
            return None
        norm = species_name.lower()
        tokens = [t for t in norm.split() if t]
        best = hits[0]
        best_score = -1
        for hit in hits:
            title = (hit.get("title") or "").lower()
            snippet = (hit.get("snippet") or "").lower()
            score = 0
            if title == norm:
                score += 5
            if norm in title:
                score += 3
            if tokens and all(t in title for t in tokens):
                score += 2
            if norm in snippet:
                score += 1
            if "genus" in title or "genus" in snippet:
                score -= 2
            if score > best_score:
                best = hit
                best_score = score
        return best

    try:
        direct = extract_by_title(species_name)
        if direct:
            return direct

        search_params = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": species_name,
            "srlimit": 5,
        }
        search_resp = requests.get(api_url, params=search_params, headers=headers, timeout=20)
        search_resp.raise_for_status()
        search_data = search_resp.json()
        hits = search_data.get("query", {}).get("search", [])
        if not hits:
            # fallback: try a title-scoped search if general search fails
            search_params["srsearch"] = f"intitle:{species_name}"
            search_resp = requests.get(api_url, params=search_params, headers=headers, timeout=20)
            search_resp.raise_for_status()
            search_data = search_resp.json()
            hits = search_data.get("query", {}).get("search", [])
            if not hits:
                return None
        best_hit = pick_best_hit(hits)
        pageid = (best_hit or {}).get("pageid")
        if not pageid:
            return None
        data_text = extract_by_pageid(pageid)
    except Exception as exc:
        print(f"Wikipedia fetch failed for '{species_name}': {exc}")
        return None

    return data_text


def get_pdf(paper: dict, location: str) -> pathlib.Path | None:
    """Download the PDF associated with an OpenAlex record if available."""
    best_location = paper.get("best_oa_location") or {}
    primary_location = paper.get("primary_location") or {}
    locations = paper.get("locations") or []
    doi = paper.get("doi")

    # start with best_oa_location, then fall back to any OA location, then primary_location
    url = best_location.get("pdf_url")
    if not url:
        for loc in locations:
            if loc.get("is_oa") and loc.get("pdf_url"):
                url = loc["pdf_url"]
                break
    if not url and primary_location.get("pdf_url"):
        url = primary_location["pdf_url"]
    if not url:
        # as a final fallback, accept any pdf_url in locations even if is_oa flag is missing
        for loc in locations:
            if loc.get("pdf_url"):
                url = loc["pdf_url"]
                break
    def fetch_via_unpaywall(doi_str: str) -> Optional[str]:
        try:
            resp = requests.get(
                f"https://api.unpaywall.org/v2/{doi_str}",
                params={"email": EMAIL},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            up_loc = data.get("best_oa_location") or {}
            if not up_loc:
                oa_locs = data.get("oa_locations") or []
                if oa_locs:
                    up_loc = oa_locs[0]
            candidate = up_loc.get("url_for_pdf") or up_loc.get("url")
            return candidate
        except Exception as exc:
            print(f"Unpaywall lookup failed for {doi}: {exc}")
            return None

    tried_unpaywall = False
    if not url and doi:
        doi_str = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        url = fetch_via_unpaywall(doi_str)
        tried_unpaywall = True

    if not url:
        print(f"No PDF URL for paper {paper.get('id')}")
        return None

    paper_id = paper["id"].split("/")[-1]
    pdf_dir = pathlib.Path(location)
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"{paper_id}.pdf"

    if not url:
        print(f"No PDF URL for paper {paper_id}")
        return None

    if pdf_path.exists():
        return pdf_path

    with requests.Session() as session:
        session.headers.update({"From": EMAIL})
        def attempt_download(target_url: str) -> bool:
            nonlocal tried_unpaywall
            try:
                resp = session.get(target_url, stream=True, timeout=120)
                resp.raise_for_status()
            except requests.HTTPError as exc:
                # on 403, try Unpaywall (if not yet tried) to get an alternate link
                if exc.response is not None and exc.response.status_code == 403 and doi:
                    doi_str = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
                    alt_url = fetch_via_unpaywall(doi_str) if not tried_unpaywall else None
                    tried_unpaywall = True
                    if alt_url and alt_url != target_url:
                        return attempt_download(alt_url)
                print(f"HTTPError downloading PDF for paper {paper_id}: {exc}")
                return False
            except Exception as exc:
                print(f"Error downloading PDF for paper {paper_id}: {exc}")
                return False

            content_type = resp.headers.get("Content-Type", "")
            # skip HTML or other non-PDF payloads to avoid corrupt files
            if "pdf" not in content_type.lower():
                print(f"Non-PDF content for paper {paper_id}: {content_type}")
                return False

            with open(pdf_path, "wb") as handle:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        handle.write(chunk)
            return True

        success = attempt_download(url)

    return pdf_path if pdf_path.exists() else None


class RAG:
    """Lightweight retrieval-augmented generation helper."""

    def __init__(
        self,
        log_runs: bool = False,
        log_root: str = "./logs",
        persist_dir: str = "./chroma_store_ollama",
        embed_backend: str | None = None,
        paper_chunk_cap: float | None = None,
    ) -> None:
        self.vectorstore = Chroma(
            collection_name="bunch_of_docs",
            embedding_function=make_embeddings(embed_backend=embed_backend),
            persist_directory=persist_dir,  # keep embeddings/metadata across runs
        )
        self.threshold: float | None = None  # optional hard cutoff on distance
        self.per_species_keep_percentile = 0.6  # keep best 40% (by distance; lower is better) of MMR-selected
        self.per_species_final_k = 50
        self.per_species_fetch_k = 100
        self.mmr_lambda = 0.5
        self.ingested_per_species: dict[str, set[str]] = defaultdict(set)
        self.paper_chunk_cap: float | None = paper_chunk_cap
        self.log_runs = log_runs
        self.log_root = pathlib.Path(log_root)
        self.log_dir: pathlib.Path | None = None
        self._restore_ingested_from_store()

    def _add_documents_with_retry(self, splits: list) -> None:
        """Add documents to Chroma with a single 30s retry on 503 server-busy errors."""
        try:
            self.vectorstore.add_documents(splits)
        except Exception as exc:
            if "503" in str(exc) or "server busy" in str(exc).lower() or "maximum pending" in str(exc).lower():
                print(f"[RAG] Embedding server busy (503) — waiting 30s then retrying once...")
                time.sleep(30)
                self.vectorstore.add_documents(splits)  # raises on second failure → shuts down run
            else:
                raise

    def _restore_ingested_from_store(self) -> None:
        """
        Pre-populate ingested_per_species from an existing Chroma store.
        This allows a shared store to be reused across pipeline runs without
        re-downloading or re-embedding already-ingested documents.
        """
        try:
            result = self.vectorstore.get(include=["metadatas"])
            metadatas = result.get("metadatas") or []
            for meta in metadatas:
                if not meta:
                    continue
                specie = meta.get("specie") or ""
                source = meta.get("source_path") or meta.get("doc_id") or ""
                if specie and source:
                    self.ingested_per_species[specie].add(source)
            if self.ingested_per_species:
                total = sum(len(v) for v in self.ingested_per_species.values())
                print(
                    f"[RAG] Restored {total} source entries for "
                    f"{len(self.ingested_per_species)} species from existing Chroma store."
                )
        except Exception as exc:
            print(f"[RAG] Could not restore ingested state from store (non-fatal): {exc}")

    def _init_log_dir(self) -> None:
        if not self.log_runs:
            return
        if self.log_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.log_dir = self.log_root / timestamp
            self.log_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def has_text(pdf_path: str, max_pages: int = 3, min_words: int = 10) -> bool:
        """Detect whether a PDF has extractable text before triggering OCR."""
        try:
            reader = PdfReader(pdf_path)
        except (PdfReadError, FileNotFoundError, ValueError, OSError) as exc:
            print(f"Unable to read PDF {pdf_path}: {exc}")
            return False
        word_count = 0
        for i, page in enumerate(reader.pages):
            if i >= max_pages:
                break
            text = page.extract_text()
            if text:
                words = text.strip().split()
                word_count += len(words)
                if word_count >= min_words:
                    return True
        return False

    @staticmethod
    def is_valid_pdf(pdf_path: str) -> bool:
        """Quick validation to avoid crashing on malformed cached downloads."""
        try:
            reader = PdfReader(pdf_path)
            _ = len(reader.pages)
            return True
        except (PdfReadError, FileNotFoundError, ValueError, OSError) as exc:
            print(f"Skipping invalid PDF {pdf_path}: {exc}")
            return False

    def _already_ingested(self, doc_id: Optional[str]) -> bool:
        """Check vector store for an existing doc_id."""
        if not doc_id:
            return False
        existing = self.vectorstore.get(where={"doc_id": doc_id})
        return bool(existing.get("ids"))

    def fetch_and_prepare(
        self, query: str, location: str = "./pdfs", first_new: int = 10, specie_norm: Optional[str] = None
    ) -> list[dict]:
        """Download PDFs for a species query and keep accompanying metadata."""
        pdf_dir = pathlib.Path(location)
        pdf_dir.mkdir(parents=True, exist_ok=True)

        candidates = works_with_oa(query, first_new * 3)
        downloaded_entries: list[dict] = []

        for paper in candidates:
            # keep both PDF path and OpenAlex metadata together
            paper_id = paper["id"].split("/")[-1]
            pdf_path = pdf_dir / f"{paper_id}.pdf"
            doc_id = canonical_doc_id(paper)

            if self._already_ingested(doc_id):
                if specie_norm and doc_id:
                    self.ingested_per_species[specie_norm].add(doc_id)
                continue

            if pdf_path.exists():
                if not self.is_valid_pdf(str(pdf_path)):
                    try:
                        pdf_path.unlink()
                    except OSError:
                        pass
                else:
                    downloaded_entries.append({"pdf_path": pdf_path, "paper": paper, "doc_id": doc_id})
                    continue

            try:
                downloaded = get_pdf(paper, location)
                if downloaded and self.is_valid_pdf(str(downloaded)):
                    downloaded_entries.append({"pdf_path": downloaded, "paper": paper, "doc_id": doc_id})
                elif downloaded:
                    try:
                        pathlib.Path(downloaded).unlink()
                    except OSError:
                        pass
            except Exception as exc:
                print(f"Failed to download PDF for paper {paper['id']}: {exc}")
                continue

            if len(downloaded_entries) >= first_new:
                break

        return downloaded_entries

    def load_ocr(
        self, pdf_path: str, puppy: str | None = None, paper_meta: Optional[dict] = None
    ) -> list:
        """Load a PDF (with OCR fallback), chunk it, and push metadata to Chroma."""
        if not self.is_valid_pdf(pdf_path):
            return []

        # short-circuit if this PDF is already in the vector store
        existing = self.vectorstore.get(where={"source_path": str(pdf_path)})
        if existing.get("ids"):
            return []

        if self.has_text(pdf_path):
            print("Text detected in PDF. Using fast strategy.")
            loader = PyPDFLoader(pdf_path)
        else:
            print("No text detected. Using OCR.")
            loader = UnstructuredPDFLoader(pdf_path, strategy="ocr_only")

        try:
            docs = loader.load()
        except Exception as exc:
            print(f"Skipping unreadable PDF during load: {pdf_path} ({exc})")
            return []
        for doc in docs:
            doc.page_content = _strip_references(doc.page_content)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)
        splits = [doc for doc in splits if len(doc.page_content.strip()) >= 80]
        if self.paper_chunk_cap:
            splits = _apply_chunk_cap(splits, self.paper_chunk_cap, source_label=str(pdf_path))

        specie_norm = puppy.strip().lower() if puppy else None

        for i, doc in enumerate(splits):
            # merge loader metadata with paper details before storing
            metadata = dict(doc.metadata) if doc.metadata else {}
            # annotate each chunk with local bookkeeping info
            metadata.update(
                {
                    "chunk_index": i,
                    "source_path": str(pdf_path),
                    "specie": specie_norm,
                    # carry a canonical doc_id so we can dedup across sources (doi -> oa/id -> title hash)
                    "doc_id": paper_meta.get("doc_id") if paper_meta else None,
                }
            )
            if paper_meta:
                best_oa_location = paper_meta.get("best_oa_location") or {}
                # attach bibliographic metadata for later citation
                metadata.update(
                    {
                        "openalex_id": paper_meta.get("id"),
                        "title": paper_meta.get("title"),
                        "publication_year": paper_meta.get("publication_year"),
                        "pdf_url": best_oa_location.get("pdf_url"),
                    }
                )
            doc.metadata = metadata

        self._add_documents_with_retry(splits)
        # persist to disk so future runs can reuse without re-embedding
        return splits

    def ingest_pmc_texts(
        self, query: str, specie_norm: str, first_n: int = 3
    ) -> None:
        """Fetch PMC full text for the query and ingest as chunks."""
        try:
            pmc_ids = pmc_search(query, first_n=first_n)
        except Exception as exc:
            print(f"PMC search failed for '{query}', skipping PMC ingestion: {exc}")
            return
        for pmcid in pmc_ids:
            source_path = f"pmc:{pmcid}"
            # skip if already present in the vector store
            existing = self.vectorstore.get(where={"source_path": source_path})
            if existing.get("ids"):
                self.ingested_per_species[specie_norm].add(source_path)
                continue

            try:
                full_text = pmc_fetch_fulltext(pmcid)
            except Exception as exc:
                print(f"PMC fetch failed for {pmcid}, skipping: {exc}")
                continue
            if not full_text:
                print(f"No PMC full text for {pmcid}")
                continue

            full_text = _strip_references(full_text)
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=200
            )
            splits = splitter.split_documents(
                [Document(page_content=full_text, metadata={})]
            )
            splits = [doc for doc in splits if len(doc.page_content.strip()) >= 80]
            if self.paper_chunk_cap:
                splits = _apply_chunk_cap(splits, self.paper_chunk_cap, source_label=f"pmc:{pmcid}")
            for i, doc in enumerate(splits):
                doc.metadata.update(
                    {
                        "chunk_index": i,
                        "source_path": source_path,
                        "specie": specie_norm,
                        "pmcid": pmcid,
                        "source": "pmc",
                        "doc_id": f"pmc:{pmcid}",
                    }
                )

            self._add_documents_with_retry(splits)
            self.ingested_per_species[specie_norm].add(source_path)

    def ingest_wikipedia(
        self, title: str, specie_norm: str, lang: str = "en"
    ) -> bool:
        """Fetch and ingest a Wikipedia article as plaintext."""
        source_path = f"wikipedia:{lang}:{title}"
        # skip if already present in the vector store
        existing = self.vectorstore.get(where={"source_path": source_path})
        if existing.get("ids"):
            self.ingested_per_species[specie_norm].add(source_path)
            return True

        content = wikipedia_fetch_plain(title, lang=lang)
        if not content:
            print(f"No Wikipedia content for '{title}'")
            return False

        content = _strip_references(content)
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(
            [Document(page_content=content, metadata={})]
        )
        splits = [doc for doc in splits if len(doc.page_content.strip()) >= 80]
        for i, doc in enumerate(splits):
            doc.metadata.update(
                {
                    "chunk_index": i,
                    "source_path": source_path,
                    "specie": specie_norm,
                    "doc_id": f"wiki:{lang}:{title}".lower(),
                    "title": title,
                    "source": "wikipedia",
                }
            )
        self._add_documents_with_retry(splits)
        self.ingested_per_species[specie_norm].add(source_path)
        return True

    def chain(self, question: str) -> str:
        """Run the retrieval + generation chain for a question."""
        if not self.vectorstore:
            return "Vector store not initialized."

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

        # run per-species retrieval to avoid one species crowding out others
        species_list = list(self.ingested_per_species.keys())
        results_with_scores = []
        for specie in species_list:
            # grab a wider candidate pool then pick diverse chunks via MMR
            mmr_query = MMR_QUERY_TEMPLATE.format(species_name=specie)
            candidates = self.vectorstore.similarity_search_with_score(
                mmr_query, k=self.per_species_fetch_k, filter={"specie": specie}
            )
            if not candidates:
                continue
            try:
                query_embedding = np.array(
                    self.vectorstore._embedding_function.embed_query(mmr_query),  # type: ignore[attr-defined]
                    dtype=float,
                )
                doc_embeddings = np.array(
                    self.vectorstore._embedding_function.embed_documents(  # type: ignore[attr-defined]
                        [doc.page_content for doc, _ in candidates]
                    ),
                    dtype=float,
                )
                mmr_indices = maximal_marginal_relevance(
                    query_embedding,
                    doc_embeddings,
                    k=min(self.per_species_final_k, len(candidates)),
                    lambda_mult=self.mmr_lambda,
                )
            except Exception as exc:
                print(f"MMR fallback for specie '{specie}': {exc}")
                mmr_indices = list(range(min(self.per_species_final_k, len(candidates))))
            selected = [candidates[idx] for idx in mmr_indices]
            # percentile filter within species based on distance (lower is better)
            if selected:
                sorted_by_score = sorted(selected, key=lambda t: t[1])
                keep_n = max(1, math.ceil(self.per_species_keep_percentile * len(sorted_by_score)))
                results_with_scores.extend(sorted_by_score[:keep_n])

        # dedupe by document/chunk to avoid repeats across species pulls
        seen_keys = set()
        filtered_docs = []
        for doc, score in results_with_scores:
            key = (doc.metadata.get("source_path"), doc.metadata.get("chunk_index"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if self.threshold is None or score <= self.threshold:
                filtered_docs.append(doc)

        if not filtered_docs:
            return "booooooo, no docs found within the given threshold"

        # initialize logging folder if enabled
        self._init_log_dir()

        # summarize how many distinct PDFs were actually used per species
        used_per_species: dict[str, set[str]] = defaultdict(set)
        for doc in filtered_docs:
            specie = (doc.metadata or {}).get("specie", "").strip()
            source = (doc.metadata or {}).get("source_path")
            if specie and source:
                used_per_species[specie].add(source)

        # build allowed citation tags from retrieved chunks
        allowed_tags = sorted(
            {
                "[source: {id}|chunk:{idx}]".format(
                    id=(doc.metadata or {}).get("doc_id", "unknown"),
                    idx=(doc.metadata or {}).get("chunk_index", "na"),
                )
                for doc in filtered_docs
            }
        )
        # log input chunk metadata
        if self.log_dir:
            chunk_meta = []
            for doc in filtered_docs:
                meta = doc.metadata or {}
                chunk_meta.append(
                    {
                        "doc_id": meta.get("doc_id"),
                        "chunk_index": meta.get("chunk_index"),
                        "specie": meta.get("specie"),
                        "source_path": meta.get("source_path"),
                        "openalex_id": meta.get("openalex_id"),
                        "title": meta.get("title"),
                        "publication_year": meta.get("publication_year"),
                        "pdf_url": meta.get("pdf_url"),
                        "snippet": (doc.page_content or "")[:500],
                    }
                )
            with open(self.log_dir / "input_chunks.json", "w", encoding="utf-8") as f:
                json.dump(chunk_meta, f, ensure_ascii=False, indent=2)

        prompt_file=PROMPT_FILE    
        prompt = PromptTemplate(
            input_variables=["context", "question", "format_instructions", "allowed_tags"],
            template=pathlib.Path(prompt_file).read_text(encoding="utf8"),
            template_format="jinja2",
        )

        llm = make_chat_llm(model=None, temperature=0.2)

        def inspect_prompt(inputs: dict) -> dict:
            formatted_prompt = prompt.format(
                context=format_docs(filtered_docs),
                question=question,
                format_instructions="Return a JSON array of bullets; each bullet has 'trait' (string), 'subclaims' (list of {statement, sources}), and 'combined_sources' (list of citation tags).",
                allowed_tags="\n".join(allowed_tags),
            )
            if self.log_dir:
                with open(self.log_dir / prompt_file, "w", encoding="utf-8") as f:
                    f.write(formatted_prompt)
            return inputs

        rag_chain = (
            {
                "context": RunnableLambda(lambda _: format_docs(filtered_docs)),
                "question": RunnablePassthrough(),
                "format_instructions": RunnableLambda(lambda _: "Return JSON with a top-level 'bullets' array; each item has 'statement' (string) and 'citations' (string of citation tags)."),
                "allowed_tags": RunnableLambda(lambda _: "\n".join(allowed_tags)),
            }
            | RunnableLambda(inspect_prompt)
            | prompt
            | llm
        )

        raw_answer = rag_chain.invoke(question)
        if self.log_dir:
            with open(self.log_dir / "raw_answer.txt", "w", encoding="utf-8") as f:
                f.write(str(raw_answer))
        try:
            payload = raw_answer.content if hasattr(raw_answer, "content") else raw_answer
            text_payload = payload.strip()
            if text_payload.startswith("```"):
                # Strip Markdown fences, optionally with a leading 'json' tag
                import re as _re
                m = _re.search(r"```(?:json)?\s*(.*?)```", text_payload, _re.DOTALL)
                if m:
                    text_payload = m.group(1).strip()
            if text_payload.strip().lower() == "no trait found":
                bullets = []
                no_trait_found = True
            else:
                no_trait_found = False
                data = json.loads(text_payload)
                if isinstance(data, dict):
                    bullets = data.get("bullets", [])
                elif isinstance(data, list):
                    bullets = data
                else:
                    bullets = []
        except Exception:
            print("Debug: no valid JSON in model output")
            bullets = []
            no_trait_found = False
        if not bullets and not no_trait_found:
            print("Debug: no bullets returned after parsing")
        bullets_out = []
        allowed_set = set(allowed_tags)
        for item in bullets:
            trait = (item.get("trait") or "").strip()
            subclaims = item.get("subclaims") or []
            combined_sources = item.get("combined_sources") or []
            citations = list(combined_sources)
            if not citations and subclaims:
                for sub in subclaims:
                    citations.extend(sub.get("sources") or [])
            citations = [c for c in citations if c in allowed_set]
            if not trait:
                print("Debug: skipped bullet with empty trait")
                continue
            if STRICT_CITATION_FILTER and not citations:
                print(f"Debug: skipped bullet '{trait}' due to no allowed citations")
                continue
            use_cits = citations if citations else []
            if not use_cits:
                print(f"Debug: bullet '{trait}' had no citations")
            if trait and use_cits:
                bullets_out.append(f"- {trait} " + " ".join(use_cits))
        answer = "No trait found" if no_trait_found else "\n".join(bullets_out)

        # Append counts summary for transparency
        summaries = []
        all_species = set(self.ingested_per_species.keys()) | set(used_per_species.keys())
        for specie in sorted(all_species):
            ingested = self.ingested_per_species.get(specie, set())
            used = used_per_species.get(specie, set())
            # separate counts by source type
            ingested_pdfs = {s for s in ingested if not str(s).startswith("wikipedia:")}
            ingested_wiki = {s for s in ingested if str(s).startswith("wikipedia:")}
            used_pdfs = {s for s in used if not str(s).startswith("wikipedia:")}
            used_wiki = {s for s in used if str(s).startswith("wikipedia:")}
            summaries.append(
                f"{specie}: PDFs {len(ingested_pdfs)} ingested / {len(used_pdfs)} used; "
                f"Wikipedia {len(ingested_wiki)} ingested / {len(used_wiki)} used"
            )
        counts_line = "Counts: " + "; ".join(summaries) if summaries else "Counts: none"

        final_answer = f"{answer}\n\n{counts_line}"

        if self.log_dir:
            with open(self.log_dir / "answer.txt", "w", encoding="utf-8") as f:
                f.write(final_answer)

        return final_answer

    def query(self, question: str, species_groups: Optional[list] = None) -> None:
        """Download PDFs for each species (with aliases), then run the chain."""
        self.ingested_per_species.clear()

        # Build list of (canonical, search_terms) tuples
        groups = []
        if species_groups:
            for entry in species_groups:
                canonical = entry.get("canonical", "").strip().lower()
                aliases = [a.strip() for a in entry.get("aliases", []) if a.strip()]
                if canonical:
                    search_terms = [canonical] + [a.lower() for a in aliases]
                    groups.append((canonical, search_terms))
        else:
            # fallback to comma-separated question terms
            terms = [t.strip() for t in question.split(",") if t.strip()]
            groups = [(t.lower(), [t.lower()]) for t in terms]

        for canonical, search_terms in groups:
            # Wikipedia first: try canonical, then aliases until one succeeds
            wiki_ingested = False
            for term in search_terms:
                if term == canonical:
                    continue
                if self.ingest_wikipedia(title=term, specie_norm=canonical):
                    wiki_ingested = True
                    break

            # First pull PMC texts for this species
            self.ingest_pmc_texts(query=canonical, specie_norm=canonical)

            for term in search_terms:
                # then fall back to OpenAlex PDFs for each alias/term
                papers = self.fetch_and_prepare(query=term, specie_norm=canonical)

                for paper_entry in papers:
                    pdf_path = paper_entry["pdf_path"]
                    paper_meta = paper_entry.get("paper") or {}
                    # attach doc_id from fetch phase so chunks carry it
                    if "doc_id" not in paper_meta:
                        paper_meta["doc_id"] = paper_entry.get("doc_id")
                    print(f"Processing PDF: {pdf_path}")
                    # pass metadata so chunks retain citation fields
                    self.load_ocr(
                        pdf_path=str(pdf_path), puppy=canonical, paper_meta=paper_meta
                    )
                    self.ingested_per_species[canonical].add(str(pdf_path))

        # If species_groups was provided, set question to canonical list for prompt
        final_question = question
        if species_groups:
            final_question = ", ".join([c for c, _ in groups])

        print(self.chain(final_question))


def load_species_file(path: str) -> list:
    """Load species/alias mapping from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Species file must contain a list of mappings")
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG pipeline.")
    parser.add_argument("--species-file", help="Path to JSON file with canonical/aliases mappings.")
    parser.add_argument("--question", default=DEFAULT_QUERY, help="Fallback comma-separated species string.")
    parser.add_argument("--log-run", action="store_true", help="Log inputs, prompt, and answer to logs/<timestamp>/")
    args = parser.parse_args()

    species_groups = load_species_file(args.species_file) if args.species_file else None

    rag = RAG(log_runs=args.log_run)
    rag.query(args.question, species_groups=species_groups)
