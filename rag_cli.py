#!/usr/bin/env python3
"""
Command-line entry point mirroring the RAG notebook pipeline.
Edit DEFAULT_QUERY below to target different species.
"""

from __future__ import annotations

import os
import pathlib
from collections import defaultdict
from typing import Optional

import requests
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredPDFLoader,
)
from PyPDF2 import PdfReader 
from PyPDF2.errors import PdfReadError

load_dotenv()

EMAIL = os.getenv("EMAIL", "trifonova.kate.s@gmail.com")
BASE_URL = "https://api.openalex.org/works"
DEFAULT_QUERY = "Cape golden mole, star-nosed mole, naked mole-rat, blind mole-rat"


def works_with_oa(query: str, first_n: int = 1) -> list[dict]:
    """Query OpenAlex for OA works whose abstract mentions the taxon."""
    params = {
        "filter": f'abstract.search:"{query}",best_open_version:published',
        "per-page": first_n,
        # request publication year for downstream citations
        "select": "id,title,publication_year,best_oa_location",
        "mailto": EMAIL,
    }
    papers = requests.get(BASE_URL, params=params, timeout=60).json()["results"]
    return papers

def get_pdf(paper: dict, location: str) -> pathlib.Path | None:
    """Download the PDF associated with an OpenAlex record if available."""
    best_location = paper.get("best_oa_location") or {}
    url = best_location.get("pdf_url")
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
        try:
            resp = session.get(url, stream=True, timeout=120)
            resp.raise_for_status()
        except requests.HTTPError as exc:
            print(f"HTTPError downloading PDF for paper {paper_id}: {exc}")
            return None
        except Exception as exc:
            print(f"Error downloading PDF for paper {paper_id}: {exc}")
            return None

        content_type = resp.headers.get("Content-Type", "")
        # skip HTML or other non-PDF payloads to avoid corrupt files
        if "pdf" not in content_type.lower():
            print(f"Non-PDF content for paper {paper_id}: {content_type}")
            return None

        with open(pdf_path, "wb") as handle:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    handle.write(chunk)

    return pdf_path


class RAG:
    """Lightweight retrieval-augmented generation helper."""

    def __init__(self) -> None:
        persist_dir = "./chroma_store"
        self.vectorstore = Chroma(
            collection_name="bunch_of_docs",
            embedding_function=OpenAIEmbeddings(),
            persist_directory=persist_dir,  # keep embeddings/metadata across runs
        )
        self.threshold = 0.5
        self.ingested_per_species: dict[str, set[str]] = defaultdict(set)

    @staticmethod
    def has_text(pdf_path: str, max_pages: int = 3, min_words: int = 10) -> bool:
        """Detect whether a PDF has extractable text before triggering OCR."""
        try:
            reader = PdfReader(pdf_path)
        except (PdfReadError, FileNotFoundError) as exc:
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
    def fetch_and_prepare(
        query: str, location: str = "./pdfs", first_new: int = 10
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

            if pdf_path.exists():
                downloaded_entries.append({"pdf_path": pdf_path, "paper": paper})
                continue

            try:
                downloaded = get_pdf(paper, location)
                if downloaded:
                    downloaded_entries.append({"pdf_path": downloaded, "paper": paper})
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

        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = splitter.split_documents(docs)

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
                }
            )
            if paper_meta:
                # attach bibliographic metadata for later citation
                metadata.update(
                    {
                        "openalex_id": paper_meta.get("id"),
                        "title": paper_meta.get("title"),
                        "publication_year": paper_meta.get("publication_year"),
                        "pdf_url": paper_meta.get("best_oa_location", {}).get("pdf_url"),
                    }
                )
            doc.metadata = metadata

        self.vectorstore.add_documents(splits)
        # persist to disk so future runs can reuse without re-embedding
        self.vectorstore.persist()
        return splits

    def chain(self, question: str) -> str:
        """Run the retrieval + generation chain for a question."""
        if not self.vectorstore:
            return "Vector store not initialized."

        def format_docs(docs: list) -> str:
            return "\n".join(doc.page_content for doc in docs)

        # run per-species retrieval to avoid one species crowding out others
        species_list = list(self.ingested_per_species.keys())
        per_species_k = 5
        results_with_scores = []
        for specie in species_list:
            results_with_scores.extend(
                self.vectorstore.similarity_search_with_score(
                    question, k=per_species_k, filter={"specie": specie}
                )
            )

        # dedupe by document/chunk to avoid repeats across species pulls
        seen_keys = set()
        filtered_docs = []
        for doc, score in results_with_scores:
            key = (doc.metadata.get("source_path"), doc.metadata.get("chunk_index"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if score <= self.threshold:
                filtered_docs.append(doc)

        if not filtered_docs:
            return "booooooo, no docs found within the given threshold"

        # summarize how many distinct PDFs were actually used per species
        used_per_species: dict[str, set[str]] = defaultdict(set)
        for doc in filtered_docs:
            specie = (doc.metadata or {}).get("specie", "").strip()
            source = (doc.metadata or {}).get("source_path")
            if specie and source:
                used_per_species[specie].add(source)

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=pathlib.Path("prompt.txt").read_text(encoding="utf8"),
        )

        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2)

        def inspect_prompt(inputs: dict) -> dict:
            formatted_prompt = prompt.format(
                context=format_docs(filtered_docs), question=question
            )
            print(formatted_prompt)
            return inputs

        rag_chain = (
            {
                "context": RunnableLambda(lambda _: format_docs(filtered_docs)),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(inspect_prompt)
            | prompt
            | llm
            | StrOutputParser()
        )

        answer = rag_chain.invoke(question)

        # Append counts summary for transparency
        summaries = []
        all_species = set(self.ingested_per_species.keys()) | set(used_per_species.keys())
        for specie in sorted(all_species):
            ingested = len(self.ingested_per_species.get(specie, set()))
            used = len(used_per_species.get(specie, set()))
            summaries.append(f"{specie}: {ingested} PDFs ingested / {used} used")
        counts_line = "Counts: " + "; ".join(summaries) if summaries else "Counts: none"

        return f"{answer}\n\n{counts_line}"

    def query(self, question: str) -> None:
        """Download PDFs for each species in the query, then run the chain."""
        self.ingested_per_species.clear()
        for puppy in question.split(","):
            specie_norm = puppy.strip().lower()
            papers = self.fetch_and_prepare(query=puppy)

            for paper_entry in papers:
                pdf_path = paper_entry["pdf_path"]
                paper_meta = paper_entry.get("paper")
                print(f"Processing PDF: {pdf_path}")
                # pass metadata so chunks retain citation fields
                self.load_ocr(
                    pdf_path=str(pdf_path), puppy=puppy.strip(), paper_meta=paper_meta
                )
                self.ingested_per_species[specie_norm].add(str(pdf_path))

        print(self.chain(question))


if __name__ == "__main__":
    rag = RAG()
    rag.query(DEFAULT_QUERY)
