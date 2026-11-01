import streamlit as st
from database import init_db, store_document, store_embeddings
from pdf_processor import PDFProcessor
from qa_engine import QAEngine
import tempfile

def main():
    st.title("PDF Question-Answering System")
    
    # Initialize components
    db_conn = init_db()
    pdf_processor = PDFProcessor()
    qa_engine = QAEngine()
    
    # File upload
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            
            # Process PDF
            text = pdf_processor.extract_text(tmp_file.name)
            chunks = pdf_processor.create_chunks(text)
            embeddings = pdf_processor.generate_embeddings(chunks)
            
            # Store in database
            doc_id = store_document(db_conn, uploaded_file.name, text)
            store_embeddings(db_conn, doc_id, embeddings, chunks)
            
            st.success("PDF processed and stored successfully!")
    
    # Question answering
    question = st.text_input("Ask a question about the uploaded PDF:")
    
    if question:
        similar_chunks = qa_engine.find_similar_chunks(db_conn, question)
        response = qa_engine.generate_response(similar_chunks, question)
        
        st.subheader("Answer:")
        st.write(response)
        
        st.subheader("Relevant Context:")
        for i, chunk in enumerate(similar_chunks):
            st.write(f"Chunk {i+1}:")
            st.write(chunk)
            st.write("---")

if __name__ == "__main__":
    main()