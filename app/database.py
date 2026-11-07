import numpy as np
import psycopg2
from pgvector.psycopg2 import register_vector

from app.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD


def init_db():
    """Open the existing Supabase Postgres schema; migrations are applied separately."""
    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        sslmode="require",
        connect_timeout=10,
    )
    try:
        register_vector(conn)
        return conn
    except Exception:
        conn.close()
        raise


def get_document_by_hash(conn, content_hash, user_id):
    with conn.cursor() as cur:
        cur.execute(
            """SELECT id, filename, created_at
               FROM documents
               WHERE content_hash = %s AND user_id = %s""",
            (content_hash, user_id),
        )
        return cur.fetchone()


def store_document(conn, filename, content, content_hash, user_id):
    if not user_id:
        raise PermissionError("An authenticated user is required.")
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO documents (filename, content, content_hash, user_id)
               VALUES (%s, %s, %s, %s)
               ON CONFLICT (user_id, content_hash)
               WHERE user_id IS NOT NULL AND content_hash IS NOT NULL
               DO NOTHING
               RETURNING id""",
            (filename, content, content_hash, user_id),
        )
        row = cur.fetchone()
        if row:
            return row[0], True
        cur.execute(
            """SELECT id FROM documents
               WHERE content_hash = %s AND user_id = %s""",
            (content_hash, user_id),
        )
        row = cur.fetchone()
        if row is None:
            raise RuntimeError("Document insert did not return an ID.")
        return row[0], False


def store_embeddings(conn, doc_id, embeddings, text_chunks, user_id):
    embeddings = np.asarray(embeddings, dtype=np.float32)
    if not text_chunks:
        raise ValueError("Cannot store a document without text chunks.")
    if embeddings.ndim != 2 or embeddings.shape != (len(text_chunks), 384):
        raise ValueError(
            f"Expected {len(text_chunks)} embeddings with 384 dimensions; "
            f"got {embeddings.shape}."
        )

    rows = [
        (doc_id, vector, chunk, index)
        for index, (vector, chunk) in enumerate(zip(embeddings, text_chunks))
    ]
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM documents WHERE id = %s AND user_id = %s FOR UPDATE",
                (doc_id, user_id),
            )
            if cur.fetchone() is None:
                raise PermissionError("Document ownership check failed.")
            cur.executemany(
                """INSERT INTO embeddings
                   (document_id, embedding, text_chunk, chunk_index)
                   VALUES (%s, %s, %s, %s)""",
                rows,
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def list_documents(conn, user_id):
    with conn.cursor() as cur:
        cur.execute(
            """SELECT id, filename, created_at
               FROM documents
               WHERE user_id = %s
               ORDER BY created_at DESC, id DESC""",
            (user_id,),
        )
        return cur.fetchall()


def search_similar_chunks(conn, query_embedding, document_id, user_id, limit=4):
    query_embedding = np.asarray(query_embedding, dtype=np.float32)
    if query_embedding.shape != (384,):
        raise ValueError(
            f"Expected a 384-dimensional query embedding; got {query_embedding.shape}."
        )
    with conn.cursor() as cur:
        cur.execute(
            """SELECT e.text_chunk,
                      (e.embedding <=> %s) AS distance,
                      e.chunk_index
               FROM embeddings e
               INNER JOIN documents d ON d.id = e.document_id
               WHERE e.document_id = %s AND d.user_id = %s
               ORDER BY e.embedding <=> %s
               LIMIT %s""",
            (query_embedding, document_id, user_id, query_embedding, limit),
        )
        return cur.fetchall()


def fetch_document_overview_chunks(conn, document_id, user_id, limit=12):
    """Fetch evenly distributed chunks for a document overview, scoped to its owner."""
    if not user_id:
        raise PermissionError("Authentication required.")
    limit = max(1, min(int(limit), 24))
    with conn.cursor() as cur:
        cur.execute(
            """WITH ordered_chunks AS (
                   SELECT e.text_chunk,
                          e.chunk_index,
                          ROW_NUMBER() OVER (ORDER BY e.chunk_index) AS row_num,
                          COUNT(*) OVER () AS total_chunks
                   FROM embeddings e
                   INNER JOIN documents d ON d.id = e.document_id
                   WHERE e.document_id = %s AND d.user_id = %s
               ),
               sampled_chunks AS (
                   SELECT text_chunk, chunk_index
                   FROM ordered_chunks
                   WHERE row_num = 1
                      OR MOD(row_num, GREATEST(1, CEIL(total_chunks::numeric / %s)::int)) = 0
               )
               SELECT text_chunk, 0.0::double precision AS distance, chunk_index
               FROM sampled_chunks
               ORDER BY chunk_index
               LIMIT %s""",
            (document_id, user_id, limit, limit),
        )
        rows = cur.fetchall()
    return [
        {"text": text, "distance": float(distance), "chunk_index": chunk_index}
        for text, distance, chunk_index in rows
    ]


def cleanup_db(conn):
    if conn is not None:
        conn.close()
