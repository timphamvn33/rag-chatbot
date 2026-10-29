# Local PDF RAG Chatbot

A command-line tool that indexes PDF documents and answers questions about them using locally stored vectors and a local Ollama language model. Answers are generated from retrieved passages and include source filenames and page numbers.

## Technology stack

- **Language:** Python
- **PDF text extraction:** PyMuPDF
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` via Sentence Transformers
- **Vector database:** ChromaDB, persisted locally in `chroma_db/`
- **Language model:** `llama3.2:3b` served locally by Ollama

## Architecture

This project uses a retrieval-augmented generation (RAG) architecture with two stages:

1. **Indexing (`ingest.py`)** — PyMuPDF extracts text page by page from PDFs in `documents/`. Each page is split into chunks of up to 1,000 characters with 150 characters of overlap. Sentence Transformers creates normalized embeddings for the chunks, which are stored in the `policy_documents` ChromaDB collection along with source filename and page metadata. ChromaDB uses cosine distance and saves its data locally.
2. **Question answering (`query.py`)** — The question is embedded using the same embedding model. ChromaDB retrieves up to 10 relevant chunks by cosine similarity. The retrieved text and citation metadata are passed to Ollama with the question; the local Llama model generates an answer grounded in that context.

```text
PDF files
   | PyMuPDF: extract text and split into overlapping chunks
   v
Sentence Transformers: create embeddings
   |
   v
ChromaDB (local persistent vector database)
   ^
   | retrieve up to 10 relevant chunks
Question -> Sentence Transformers -> Ollama / llama3.2:3b -> Answer with citations
```

## Requirements

- Python 3.9 or newer
- [Ollama](https://ollama.com/) installed and running
- The `llama3.2:3b` model downloaded in Ollama

## Setup

From the project directory, create and activate a virtual environment in PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

Download the language model used by the chatbot:

```powershell
ollama pull llama3.2:3b
```

The embedding model (`sentence-transformers/all-MiniLM-L6-v2`) is downloaded automatically the first time you run either script.

## Add documents and build the index

Put PDF files in the `documents` directory, then run:

```powershell
python ingest.py
```

The script extracts text from each PDF, splits it into overlapping chunks, creates embeddings, and saves them in the local `chroma_db` directory. Running ingestion again updates existing chunks with matching IDs.

Only PDFs with extractable text are indexed; scanned PDFs that require OCR are not supported by the current implementation.

## Desktop app (GUI)

Launch the desktop chat window:

```powershell
python gui.py
```

Click **Open PDF...** to choose a file. It is copied into `documents/`, indexed, and then you can ask questions in the chat box. Answers are restricted to the currently selected PDF. The first launch takes longer while the embedding model loads. Ollama must be running.

## Ask questions (command line)

Start the interactive question-and-answer prompt:

```powershell
python query.py
```

Enter a question at the `You:` prompt. Type `exit` or `quit` to end the session. The chatbot retrieves up to 10 relevant chunks and asks Ollama to answer using those passages. If the retrieved context does not contain the answer, it is instructed to say so.

Run `ingest.py` again after adding or changing PDFs so the index reflects those documents.

## Project files

- `ingest.py` — extracts, chunks, embeds, and indexes PDFs.
- `gui.py` — Tkinter desktop app: pick a PDF and chat about it.
- `query.py` — retrieves relevant chunks and generates answers with Ollama.
- `documents/` — input PDFs.
- `chroma_db/` — locally persisted Chroma vector database.
- `requirements.txt` — Python package dependencies.

The embedding model and vector database run locally. Ollama must be running on the machine when asking questions.
