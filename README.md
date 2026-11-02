# PDF Q&A System

A small, practical **retrieval-augmented generation (RAG)** app for asking questions about PDF files. It uses Streamlit for the UI, SentenceTransformers for embeddings, PostgreSQL + pgvector for retrieval, and the OpenAI API to generate answers grounded in retrieved PDF passages.

> This is a learning project, not a production-ready multi-user service. See [Limitations](#limitations) before deploying it with real documents.

## What it does

1. Upload a PDF.
2. Extract selectable text and use OCR for pages that appear to be scanned.
3. Split the text into overlapping chunks.
4. Generate normalized 384-dimensional embeddings with `all-MiniLM-L6-v2`.
5. Store the document, content hash, chunks, and vectors in PostgreSQL.
6. Retrieve the most similar chunks **from the selected PDF only**.
7. Send those passages and the question to an LLM, which generates an answer and cites source chunks.

## Features

- **PDF text extraction:** PyMuPDF extracts text page by page.
- **OCR fallback:** Tesseract OCR is used on pages with very little selectable text.
- **Overlap-aware chunking:** Adjacent chunks share context around their boundaries.
- **Semantic retrieval:** SentenceTransformers embeddings + pgvector cosine distance.
- **Document-scoped search:** A query cannot retrieve chunks from a different uploaded PDF.
- **Grounded generation:** OpenAI chat completions answer from retrieved passages and are instructed to say when the evidence is insufficient.
- **Duplicate detection:** SHA-256 hashes prevent the same PDF bytes from being indexed repeatedly.
- **Small in-process caches:** The embedding model is loaded once, and repeated query embeddings are cached.
- **Database indexes:** HNSW index for vector similarity and a B-tree index for document filtering.
- **Safer initialization:** Database startup creates/migrates schema objects without dropping existing embeddings.

## Architecture

### Ingestion

`PDF upload → SHA-256 duplicate check → text extraction / OCR → chunking with overlap → embeddings → PostgreSQL`

### Question answering

`Question + selected PDF → query embedding → pgvector top-K search scoped to document_id → retrieved evidence → OpenAI generation → answer + source passages`

### Main files

| File | Responsibility |
| --- | --- |
| `app/main.py` | Streamlit interface and end-to-end orchestration |
| `app/pdf_processor.py` | PDF extraction, OCR fallback, text cleaning, chunking, and embedding generation |
| `app/qa_engine.py` | Query embedding cache, retrieval coordination, and LLM answer generation |
| `app/database.py` | PostgreSQL schema, migrations, duplicate lookup, persistence, and vector search |
| `app/config.py` | Environment-variable configuration |
| `tests/test_pdf_processor.py` | Unit tests for chunking and text cleaning |

## Requirements

- Python 3.10 or newer recommended.
- PostgreSQL.
- The PostgreSQL `pgvector` extension installed; HNSW indexing requires a pgvector version that supports HNSW (0.5+).
- An OpenAI API key for answer generation.
- Tesseract OCR installed on your operating system if you need scanned-PDF support.

### Install PostgreSQL and pgvector

Create a PostgreSQL database, for example `pdf_qna`, and ensure the `vector` extension is available to that database. The app runs `CREATE EXTENSION IF NOT EXISTS vector`, but the pgvector extension itself must already be installed on the PostgreSQL server.

## Setup

### 1. Create and activate a virtual environment

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy `.env.example` to `.env`, then fill in your local database settings and API key.

**Windows (PowerShell)**

```powershell
Copy-Item .env.example .env
```

**macOS / Linux**

```bash
cp .env.example .env
```

Set at least these values in `.env`:

```dotenv
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pdf_qna
DB_USER=your_postgres_user
DB_PASSWORD=your_local_database_password

OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini

OCR_ENABLED=true
OCR_LANGUAGES=eng
```

Add your key to `OPENAI_API_KEY` locally. **Do not commit `.env` or put API keys in source code.** The repository ignores `.env`.

### 4. Install Tesseract for OCR

OCR is optional for PDFs with selectable text. It is required for scanned/image-only pages.

- **Windows:** Install the Tesseract OCR application and make sure `tesseract.exe` is on your `PATH`. If needed, configure `pytesseract.pytesseract.tesseract_cmd` in `app/pdf_processor.py`.
- **Ubuntu/Debian:** `sudo apt-get install tesseract-ocr`
- **macOS with Homebrew:** `brew install tesseract`

For languages other than English, install the corresponding Tesseract language data and set `OCR_LANGUAGES` accordingly. To disable OCR, set `OCR_ENABLED=false`.

### 5. Run the app

From the repository root:

```bash
streamlit run app/main.py
```

Open the local URL shown by Streamlit, upload a PDF, click **Process PDF**, select the document, and submit a question.

### 6. Run the tests

```bash
pytest -q
```

The unit tests cover chunking and text cleaning. Database, OCR, and live OpenAI integration still need environment-backed integration testing.

## How retrieval works

The app uses `all-MiniLM-L6-v2` to embed both PDF chunks and questions into 384-dimensional vectors. pgvector's cosine-distance operator (`<=>`) ranks chunks by distance; **lower distance means greater similarity**.

Search is constrained with `WHERE document_id = ...`, so only chunks belonging to the selected PDF are eligible. The database uses an HNSW cosine index for vector search and a B-tree index on `document_id` for filtering. Query plans depend on corpus size and PostgreSQL/pgvector's planner choices; inspect `EXPLAIN ANALYZE` for real workloads.

## Duplicate handling and caching

- The SHA-256 hash is computed from the original PDF bytes.
- If the same bytes have already been indexed, the app reuses the existing document rather than embedding and inserting it again.
- The SentenceTransformer model is cached per Python process.
- Query embeddings are cached in memory (up to 256 distinct query strings per QA engine instance).

Changing the PDF bytes creates a new document record. This is content-based deduplication, not fuzzy duplicate detection: PDFs that contain the same text but have different bytes may still be indexed separately.

## Advantages

- Clear, compact example of a full PDF-to-RAG pipeline.
- Components are separated by responsibility.
- Persistent vector storage rather than an in-memory-only index.
- Search is scoped to the selected document.
- LLM generation is grounded in retrieved excerpts and can report insufficient evidence.
- OCR fallback supports many scanned PDFs.
- Duplicate uploads avoid unnecessary reprocessing.
- Startup no longer drops existing embedding data.

## Limitations

- **Not production-ready:** no authentication, multi-tenant authorization, rate limiting, job queue, or deployment hardening.
- **OCR quality varies:** scanned pages, skew, handwriting, tables, and complex layouts may extract poorly.
- **Character-based chunking:** chunk sizes are measured in characters, not model tokens. Tune chunk size and overlap for the corpus.
- **Basic dense retrieval only:** no BM25 hybrid search, reranker, query transformation, or retrieval evaluation dataset yet.
- **LLM limitations:** generation can still be wrong. Source citations help trace the answer but do not guarantee correctness.
- **OpenAI dependency:** answer generation requires network access and a valid API key; API usage may incur cost.
- **Database scaling:** HNSW consumes additional storage and memory. Benchmark with your real corpus before making production claims.
- **No document deletion UI:** documents remain in the database until removed manually or through a future management feature.
- **OCR system dependency:** installing the Python package `pytesseract` does not install the Tesseract executable.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Database connection fails | Check `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, and `DB_PASSWORD` in `.env`. |
| `vector` extension cannot be created | Install pgvector on the PostgreSQL server and ensure the database user has permission. |
| HNSW index creation fails | Confirm pgvector is 0.5+ and that the database user can create indexes. |
| `OPENAI_API_KEY is not set` | Add your key to local `.env`; restart Streamlit. |
| OCR says Tesseract is missing | Install Tesseract and make sure it is available on `PATH`. |
| No text found in PDF | Confirm it is a valid PDF; for scanned pages, enable OCR and install the right language data. |
| Answer misses a detail | Inspect the retrieved evidence, try a more specific question, and consider tuning chunking/top-K or adding reranking. |

## Security notes

- Keep `.env` local; `.env.example` is safe to commit because it contains placeholders only.
- Do not upload confidential documents to a service unless you are authorized to process them.
- Retrieved PDF text is treated as untrusted input in the generation prompt, but prompt-injection defenses are not perfect.
- Review your organization's data-handling requirements before sending document excerpts to an external LLM provider.
