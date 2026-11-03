import hashlib

import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.database import (
    cleanup_db,
    get_document_by_hash,
    init_db,
    list_documents,
    store_document,
    store_embeddings,
)
from app.pdf_processor import PDFProcessor
from app.qa_engine import QAEngine


@st.cache_resource
def get_pdf_processor():
    return PDFProcessor()


@st.cache_resource
def get_qa_engine():
    return QAEngine()


def main():
    st.set_page_config(page_title="PDF Q&A", page_icon="📄", layout="wide")
    st.title("📄 PDF Question Answering")
    st.caption("Ask questions about one selected PDF. Answers are generated from retrieved passages.")

    try:
        conn = init_db()
    except Exception as exc:
        st.error(f"Could not connect to PostgreSQL / initialize pgvector: {exc}")
        st.info("Check your local .env settings and confirm PostgreSQL has the pgvector extension installed.")
        st.stop()

    try:
        processor = get_pdf_processor()
        qa_engine = get_qa_engine()

        st.subheader("1. Add a PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        if uploaded_file and st.button("Process PDF", type="primary"):
            pdf_bytes = uploaded_file.getvalue()
            content_hash = hashlib.sha256(pdf_bytes).hexdigest()
            existing = get_document_by_hash(conn, content_hash)

            if existing:
                document_id, filename, _created_at = existing
                st.session_state["selected_doc_id"] = document_id
                st.info(f"Already indexed as **{filename}**. Reused the existing document; no duplicate chunks were added.")
                st.rerun()

            try:
                with st.spinner("Extracting text, running OCR where needed, chunking, and embedding..."):
                    processed = processor.process_pdf(pdf_bytes)
                    document_id, created = store_document(
                        conn, uploaded_file.name, processed["text"], content_hash
                    )
                    if created:
                        store_embeddings(
                            conn, document_id, processed["embeddings"], processed["chunks"]
                        )
                st.session_state["selected_doc_id"] = document_id
                st.success(
                    f"Indexed **{uploaded_file.name}**: {len(processed['chunks'])} chunks created."
                )
                st.rerun()
            except Exception as exc:
                st.error(f"PDF processing failed: {exc}")

        documents = list_documents(conn)
        if not documents:
            st.info("Upload and process a PDF to get started.")
            return

        document_ids = [row[0] for row in documents]
        if st.session_state.get("selected_doc_id") not in document_ids:
            st.session_state["selected_doc_id"] = document_ids[0]

        st.subheader("2. Ask a question")
        selected_document_id = st.selectbox(
            "Search within this PDF only",
            options=document_ids,
            format_func=lambda doc_id: next(
                (f"{filename} (uploaded {created_at:%Y-%m-%d})" for item_id, filename, created_at in documents if item_id == doc_id),
                f"Document {doc_id}",
            ),
            key="selected_doc_id",
        )

        with st.form("question_form"):
            question = st.text_input("Your question", placeholder="What are the key findings?")
            submitted = st.form_submit_button("Get answer")

        if submitted:
            if not question.strip():
                st.warning("Enter a question first.")
            else:
                try:
                    with st.spinner("Retrieving evidence and generating an answer..."):
                        result = qa_engine.search_documents(
                            conn, question.strip(), document_id=selected_document_id, top_k=4
                        )
                    st.subheader("Answer")
                    st.markdown(result["response"])
                    with st.expander("Retrieved evidence", expanded=True):
                        for index, chunk in enumerate(result["relevant_chunks"], start=1):
                            st.markdown(
                                f"**Source {index} · chunk {chunk['chunk_index']} · cosine distance {chunk['distance']:.3f}**"
                            )
                            st.write(chunk["text"])
                except Exception as exc:
                    st.error(f"Question answering failed: {exc}")
    finally:
        cleanup_db(conn)


if __name__ == "__main__":
    main()
