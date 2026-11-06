import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

from app.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def init_db():
    """Connect and create additive schema objects. Apply migrations/001_user_isolation.sql once."""
    conn = psycopg2.connect(
        host=DB_HOST, port=DB_PORT, database=DB_NAME,
        user=DB_USER, password=DB_PASSWORD,
    )
    try:
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
                    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS embeddings (
                    id SERIAL PRIMARY KEY,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    embedding vector(384) NOT NULL,
                    text_chunk TEXT NOT NULL,
                    chunk_index INTEGER
                )
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS embeddings_document_id_idx ON embeddings(document_id)")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS embeddings_embedding_hnsw_idx
                ON embeddings USING hnsw (embedding vector_cosine_ops)
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS documents_user_id_idx ON documents(user_id)")
            cur.execute("DROP INDEX IF EXISTS documents_content_hash_uidx")
            cur.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS documents_user_content_hash_uidx
                ON documents(user_id, content_hash)
                WHERE user_id IS NOT NULL AND content_hash IS NOT NULL
            """)
        conn.commit()
        return conn
    except Exception:
        conn.rollback()
        conn.close()
        raise


def get_document_by_hash(conn, content_hash, user_id):
    with conn.cursor() as cur:
        cur.execute(
            """SELECT id, filename, created_at FROM documents
               WHERE content_hash = %s AND user_id = %s""",
            (content_hash, user_id),
        )
        return cur.fetchone()


def store_document(conn, filename, content, content_hash, user_id):
    if not user_id:
        raise PermissionError("An authenticated user is required to store a document.")
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO documents (filename, content, content_hash, user_id)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (user_id, content_hash)
               WHERE user_id IS NOT NULL AND content_hash IS NOT NULL
               DO NOTHING RETURNING id""",
            (filename, content, content_hash, user_id),
        )
        row = cur.fetchone()
        if row:
            return row[0], True
        cur.execute(
            "SELECT id FROM documents WHERE content_hash = %s AND user_id = %s",
            (content_hash, user_id),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("Document insert did not return an ID.")
        return row[0], False


def store_embeddings(conn, doc_id, embeddings, text_chunks, user_id):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if len(text_chunks) == 0:
        raise ValueError("Cannot store a document without text chunks.")
    if embeddings.ndim != 2 or embeddings.shape != (len(text_chunks), 384):
        raise ValueError(f"Expected {len(text_chunks)} embeddings with 384 dimensions; got {embeddings.shape}.")
    rows = [(doc_id, vector, chunk, i) for i, (vector, chunk) in enumerate(zip(embeddings, text_chunks))]
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM documents WHERE id = %s AND user_id = %s FOR UPDATE", (doc_id, user_id))
            if cur.fetchone() is None:
                raise PermissionError("Document ownership check failed.")
            cur.executemany(
                """INSERT INTO embeddings (document_id, embedding, text_chunk, chunk_index)
                   VALUES (%s, %s, %s, %s)""", rows
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def list_documents(conn, user_id):
    with conn.cursor() as cur:
        cur.execute(
            """SELECT id, filename, created_at FROM documents
               WHERE user_id = %s ORDER BY created_at DESC, id DESC""",
            (user_id,),
        )
        return cur.fetchall()


def search_similar_chunks(conn, query_embedding, document_id, user_id, limit=4):
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    if query_embedding.shape != (384,):
        raise ValueError(f"Expected a 384-dimensional query embedding; got {query_embedding.shape}.")
    with conn.cursor() as cur:
        cur.execute(
            """SELECT e.text_chunk, (e.embedding <=> %s) AS distance, e.chunk_index
               FROM embeddings e
               INNER JOIN documents d ON d.id = e.document_id
               WHERE e.document_id = %s AND d.user_id = %s
               ORDER BY e.embedding <=> %s
               LIMIT %s""",
            (query_embedding, document_id, user_id, query_embedding, limit),
        )
        return cur.fetchall()


def cleanup_db(conn):
    if conn is not None:
        conn.close()
