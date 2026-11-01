import psycopg2
from psycopg2.extensions import register_adapter
import numpy as np
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD

def init_db():
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        cur = conn.cursor()
        
        # Create vector extension if it doesn't exist
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        
        # Create tables if they don't exist
        cur.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                filename TEXT,
                content TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Drop existing embeddings table to recreate with correct dimensions
        cur.execute("DROP TABLE IF EXISTS embeddings")
        
        # Create embeddings table with 384 dimensions
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                id SERIAL PRIMARY KEY,
                document_id INTEGER REFERENCES documents(id),
                embedding vector(384),
                text_chunk TEXT
            )
        """)
        
        conn.commit()
        return conn
    except Exception as e:
        print(f"Database initialization error: {str(e)}")
        if conn:
            conn.close()
        raise

def store_document(conn, filename, content):
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO documents (filename, content) VALUES (%s, %s) RETURNING id",
            (filename, content)
        )
        doc_id = cur.fetchone()[0]
        conn.commit()
        return doc_id
    except Exception as e:
        conn.rollback()
        print(f"Error storing document: {str(e)}")
        raise

def store_embeddings(conn, doc_id, embeddings, text_chunks):
    try:
        cur = conn.cursor()
        
        # Verify embeddings dimensions
        if embeddings.shape[1] != 384:  # Expecting 384 dimensions
            raise ValueError(f"Invalid embedding dimensions. Expected 384, got {embeddings.shape[1]}")
        
        # Store embeddings and chunks
        for embedding, chunk in zip(embeddings, text_chunks):
            cur.execute(
                "INSERT INTO embeddings (document_id, embedding, text_chunk) VALUES (%s, %s, %s)",
                (doc_id, embedding.tolist(), chunk)
            )
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error storing embeddings: {str(e)}")
        raise

def search_similar_chunks(conn, query_embedding, limit=5):
    try:
        cur = conn.cursor()
        
        # Verify query embedding dimension
        if len(query_embedding) != 384:
            raise ValueError(f"Invalid query embedding dimension. Expected 384, got {len(query_embedding)}")
        
        # Search for similar chunks using cosine similarity
        cur.execute("""
            SELECT text_chunk, embedding <=> %s as distance
            FROM embeddings
            ORDER BY distance ASC
            LIMIT %s
        """, (query_embedding.tolist(), limit))
        
        results = cur.fetchall()
        return results
    except Exception as e:
        print(f"Error searching similar chunks: {str(e)}")
        raise

def cleanup_db(conn):
    if conn:
        try:
            conn.close()
        except Exception as e:
            print(f"Error closing database connection: {str(e)}")

# Example usage:
if __name__ == "__main__":
    conn = None
    try:
        # Initialize database
        conn = init_db()
        
        # Example document storage
        doc_id = store_document(conn, "example.pdf", "Sample content")
        
        # Example embeddings storage (assuming embeddings is a numpy array with shape (n, 384))
        embeddings = np.random.rand(5, 384)  # Example embeddings
        text_chunks = ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]
        store_embeddings(conn, doc_id, embeddings, text_chunks)
        
        # Example similarity search
        query_embedding = np.random.rand(384)  # Example query embedding
        results = search_similar_chunks(conn, query_embedding, limit=3)
        for chunk, distance in results:
            print(f"Chunk: {chunk}, Distance: {distance}")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
    finally:
        cleanup_db(conn)