import numpy as np
from sentence_transformers import SentenceTransformer
from psycopg2.extensions import register_adapter, AsIs
import json

class QAEngine:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
    def adapt_numpy_array(self, numpy_array):
        """Convert numpy array to a format PostgreSQL can understand"""
        return AsIs("'[" + ",".join(map(str, numpy_array)) + "]'::vector")
    
    def find_similar_chunks(self, conn, query, top_k=3):
        try:
            # Generate query embedding
            query_embedding = self.model.encode(query)
            
            # Register numpy array adapter
            register_adapter(np.ndarray, self.adapt_numpy_array)
            
            cur = conn.cursor()
            
            # Use the correct operator (<=> for cosine distance)
            cur.execute("""
                SELECT text_chunk, (embedding <=> %s) as distance
                FROM embeddings
                ORDER BY distance ASC
                LIMIT %s
            """, (query_embedding, top_k))
            
            results = cur.fetchall()
            
            # Check if we got any results
            if not results:
                return []
                
            return [result[0] for result in results]
            
        except Exception as e:
            print(f"Error in find_similar_chunks: {str(e)}")
            return []
        finally:
            if cur:
                cur.close()
    
    def generate_response(self, similar_chunks, query):
        try:
            if not similar_chunks:
                return "No relevant information found."
            
            # For basic implementation, return the most relevant chunk
            # You can enhance this with more sophisticated response generation
            return similar_chunks[0]
            
        except Exception as e:
            print(f"Error in generate_response: {str(e)}")
            return "Error generating response."

    def search_documents(self, conn, query, top_k=3):
        """Wrapper method that combines search and response generation"""
        try:
            similar_chunks = self.find_similar_chunks(conn, query, top_k)
            response = self.generate_response(similar_chunks, query)
            
            return {
                'response': response,
                'relevant_chunks': similar_chunks,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'response': f"Error processing query: {str(e)}",
                'relevant_chunks': [],
                'status': 'error'
            }

# Example usage:
if __name__ == "__main__":
    import psycopg2
    from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
    
    try:
        # Connect to database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )
        
        # Initialize QA engine
        qa_engine = QAEngine()
        
        # Example query
        query = "What is machine learning?"
        
        # Search documents
        result = qa_engine.search_documents(conn, query)
        
        # Print results
        print("Query:", query)
        print("Response:", result['response'])
        print("Status:", result['status'])
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if conn:
            conn.close()