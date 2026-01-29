import re
import sys

from rag_cli import RAG

TAG_RE = re.compile(r"\[source:\s*(?P<doc_id>[^|]+)\|chunk:(?P<chunk>\d+)\]")


def parse_tag(tag: str) -> tuple[str, int]:
    match = TAG_RE.fullmatch(tag.strip())
    if not match:
        raise ValueError("Expected tag format: [source: <doc_id>|chunk:<int>]")
    return match.group("doc_id"), int(match.group("chunk"))


def main() -> None: 
    if len(sys.argv) != 2:
        print("Usage: python checkChunk.py \"[source: <doc_id>|chunk:<int>]\"")
        raise SystemExit(2)

    doc_id, chunk_idx = parse_tag(sys.argv[1])
    store = RAG().vectorstore  # loads the persisted Chroma collection

    res = store.get(
        where={
            "$and": [
                {"doc_id": {"$eq": doc_id}},
                {"chunk_index": {"$eq": chunk_idx}},
            ]
        },
        include=["metadatas", "documents"],
    )
    documents = res.get("documents", []) or []
    metadatas = res.get("metadatas", []) or []
    found = len(documents)

    print("Found:", found)
    if found == 0:
        print(f"No chunk found for doc_id='{doc_id}' with chunk_index={chunk_idx}")
        # fallback: list available chunk indices for this doc_id
        fallback = store.get(
            where={"doc_id": {"$eq": doc_id}},
            include=["metadatas"],
        )
        metas = fallback.get("metadatas", []) or []
        if metas:
            available = sorted({m.get("chunk_index") for m in metas if "chunk_index" in m})
            print(f"Available chunk_index values for doc_id='{doc_id}': {available}")
        else:
            print(f"No entries found at all for doc_id='{doc_id}'")
    else:
        for doc, meta in zip(documents, metadatas):
            print("ID:", meta.get("doc_id"), "chunk_index:", meta.get("chunk_index"))
            print("Source path:", meta.get("source_path"))
            print("Content:\n", doc)
            print("-" * 40)


if __name__ == "__main__":
    main()
