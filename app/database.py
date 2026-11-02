import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

from app.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def init_db():
    """Open a connection and create missing schema objects without deleting data."""
    conn = None
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
        )
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        register_vector(conn)

        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    content TEXT NOT NULL,
                    content_hash TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Safe, additive migration for databases created by the older version.
            cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash TEXT")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    embedding vector(384) NOT NULL,
                    text_chunk TEXT NOT NULL,
                    chunk_index INTEGER
                )
            """)
            cur.execute("ALTER TABLE embeddings ADD COLUMN IF NOT EXISTS chunk_index INTEGER")
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS documents_content_hash_uidx
                ON documents (content_hash) WHERE content_hash IS NOT NULL
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS embeddings_document_id_idx
                ON embeddings (document_id)
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS embeddings_embedding_hnsw_idx
                ON embeddings USING hnsw (embedding vector_cosine_ops)
            """)
        conn.commit()
        return conn
    except Exception:
        if conn is not None:
            conn.rollback()
            conn.close()
        raise


def get_document_by_hash(conn, content_hash):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, filename, created_at FROM documents WHERE content_hash = %s",
            (content_hash,),
        )
        return cur.fetchone()


def store_document(conn, filename, content, content_hash):
    """Insert a document once; return (document_id, was_created)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (filename, content, content_hash)
            VALUES (%s, %s, %s)
            ON CONFLICT (content_hash) WHERE content_hash IS NOT NULL DO NOTHING
            RETURNING id
            """,
            (filename, content, content_hash),
        )
        row = cur.fetchone()
        if row:
            conn.commit()
            return row[0], True

        cur.execute("SELECT id FROM documents WHERE content_hash = %s", (content_hash,))
        row = cur.fetchone()
        conn.commit()
        if row is None:
            raise RuntimeError("Document insert did not return an ID.")
        return row[0], False


def store_embeddings(conn, doc_id, embeddings, text_chunks):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if len(text_chunks) == 0:
        return
    if embeddings.ndim != 2 or embeddings.shape != (len(text_chunks), 384):
        raise ValueError(
            f"Expected {len(text_chunks)} embeddings with 384 dimensions; got {embeddings.shape}."
        )

    rows = [
        (doc_id, embedding, chunk, index)
        for index, (embedding, chunk) in enumerate(zip(embeddings, text_chunks))
    ]
    try:
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO embeddings (document_id, embedding, text_chunk, chunk_index)
                VALUES (%s, %s, %s, %s)
                """,
                rows,
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def list_documents(conn):
    with conn.cursor() as cur:
        cur.execute("SELECT id, filename, created_at FROM documents ORDER BY created_at DESC, id DESC")
        return cur.fetchall()


def search_similar_chunks(conn, query_embedding, document_id, limit=4):
    """Search only within the selected PDF, never across unrelated uploads."""
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    if query_embedding.shape != (384,):
        raise ValueError(f"Expected a 384-dimensional query embedding; got {query_embedding.shape}.")

    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT text_chunk, (embedding <=> %s) AS distance, chunk_index
            FROM embeddings
            WHERE document_id = %s
            ORDER BY embedding <=> %s
            LIMIT %s
            """,
            (query_embedding, document_id, query_embedding, limit),
        )
        return cur.fetchall()


def cleanup_db(conn):
    if conn is not None:
        conn.close()
