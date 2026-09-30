"""
ISDO - Knowledge Base Loader (ChromaDB)
=======================================
1. Reads every .md file in data/kb/
2. Splits each file into chunks at '## ' headings
3. Stores the chunks in a ChromaDB collection called 'isdo_kb'
4. Runs 4 sample queries and prints the best-matching article + confidence

Setup:   pip install chromadb
Run:     python data/kb_setup.py   (from the project root)

Note: on first run ChromaDB downloads its small default embedding model
(all-MiniLM-L6-v2, ~80 MB). After that it works offline.
"""

from pathlib import Path

import chromadb

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR.parent.parent / "data" / "kb"
DB_DIR = BASE_DIR.parent.parent / "data" / "chroma_db"
COLLECTION_NAME = "isdo_kb"

SAMPLE_QUERIES = [
    "I forgot my password and my account is locked",
    "VPN keeps disconnecting when I work from home",
    "Outlook is not receiving new emails",
    "My laptop is very slow and says disk is almost full",
]


def split_into_chunks(md_text: str, article_name: str) -> list[dict]:
    """Split one markdown article at '## ' headings.

    Text before the first '## ' (title + metadata) becomes an 'Overview' chunk.
    Each chunk is prefixed with the article title so it keeps its context
    when retrieved on its own.
    """
    lines = md_text.splitlines()
    title = article_name
    for line in lines:
        if line.startswith("# "):
            title = line[2:].strip()
            break

    chunks = []
    section = "Overview"
    buffer = []

    def flush():
        body = "\n".join(buffer).strip()
        if body:
            chunks.append({
                "section": section,
                "text": f"{title}\n{section}\n{body}",
            })

    for line in lines:
        if line.startswith("## "):
            flush()
            section = line[3:].strip()
            buffer = []
        else:
            buffer.append(line)
    flush()

    for chunk in chunks:
        chunk["title"] = title
    return chunks


def load_kb(collection) -> int:
    md_files = sorted(KB_DIR.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files found in {KB_DIR}")

    ids, documents, metadatas = [], [], []
    for md_file in md_files:
        text = md_file.read_text(encoding="utf-8")
        chunks = split_into_chunks(text, md_file.stem)
        print(f"  {md_file.name}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            ids.append(f"{md_file.stem}::{i}")
            documents.append(chunk["text"])
            metadatas.append({
                "article": md_file.name,
                "title": chunk["title"],
                "section": chunk["section"],
            })

    collection.add(ids=ids, documents=documents, metadatas=metadatas)
    return len(md_files)


def run_queries(collection) -> None:
    print("\n--- Sample queries ---")
    for query in SAMPLE_QUERIES:
        result = collection.query(query_texts=[query], n_results=1)
        meta = result["metadatas"][0][0]
        distance = result["distances"][0][0]
        # Cosine distance: 0 = identical, 1 = unrelated. Convert to a 0-100% score.
        confidence = max(0.0, 1.0 - distance) * 100

        print(f'\nQuery      : "{query}"')
        print(f"Best match : {meta['article']}  (section: {meta['section']})")
        print(f"Confidence : {confidence:.1f}%")


def main() -> None:
    print("=== ISDO KB Loader (ChromaDB) ===")
    print(f"KB folder  : {KB_DIR}")
    print(f"DB folder  : {DB_DIR}\n")

    client = chromadb.PersistentClient(path=str(DB_DIR))

    # Start fresh each run so re-running never creates duplicate chunks
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # collection didn't exist yet

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # makes 1 - distance a sensible score
    )

    article_count = load_kb(collection)
    print(f"\nStored {collection.count()} chunks from {article_count} articles "
          f"in collection '{COLLECTION_NAME}'")

    run_queries(collection)
    print("\nDone.")


if __name__ == "__main__":
    main()
