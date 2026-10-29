from pathlib import Path

import chromadb
import fitz
from sentence_transformers import SentenceTransformer

DOCUMENTS_DIR = Path("documents")
DATABASE_DIR = Path("chroma_db")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

def load_pdf_pages(pdf_path: Path) -> list[dict]:
    """Extract text from each PDF page."""
    pages = []

    with fitz.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text=page.get_text("text").strip()

            if text:
                pages.append({
                    "text": text,
                    "source": pdf_path.name,
                    "page": page_number
                })
    return pages

def split_text(
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP
) -> list[str]:
    """Split text into overlapping chunks."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end == len(text):
            break

        start = end - overlap

    return chunks

def main():
    DOCUMENTS_DIR.mkdir(exist_ok=True)

    pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

    if not pdf_files:
        print("No PDF files found in the documents folder.")
        return

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(DATABASE_DIR))

    collection = client.get_or_create_collection(
        name="policy_documents",
        metadata={"hnsw:space": "cosine"},
    )
    for pdf_path in pdf_files:
            print(f"Processing: {pdf_path.name}")

            pages = load_pdf_pages(pdf_path)

            if not pages:
                print("  No extractable text found. Skipping.")
                continue

            ids = []
            documents = []
            metadatas = []

            for page in pages:
                chunks = split_text(page["text"])

                for chunk_index, chunk in enumerate(chunks):
                    chunk_id = (
                        f"{pdf_path.name}:"
                        f"{page['page']}:"
                        f"{chunk_index}"
                    )

                    ids.append(chunk_id)
                    documents.append(chunk)
                    metadatas.append({
                        "source": page["source"],
                        "page": page["page"],
                        "chunk_index": chunk_index,
                    })

            if not documents:
                continue

            embeddings = model.encode(
                documents,
                normalize_embeddings=True,
                show_progress_bar=True,
            ).tolist()

            # Upsert makes rerunning this script safe for unchanged IDs.
            collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            print(f"  Indexed {len(documents)} chunks.")

    print(f"\nTotal chunks in database: {collection.count()}")
    print(f"Database saved locally at: {DATABASE_DIR}")


if __name__ == "__main__":
    main()