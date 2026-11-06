import hashlib

import streamlit as st
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.auth import require_authentication, current_user
from app.database import (
    cleanup_db, get_document_by_hash, init_db, list_documents,
    store_document, store_embeddings,
)
from app.observability import event, timed
from app.pdf_processor import PDFProcessor
from app.qa_engine import QAEngine
from app.security import validate_query, validate_upload
from app.redis_store import allow_request


@st.cache_resource
def get_pdf_processor():
    return PDFProcessor()


@st.cache_resource
def get_qa_engine():
    return QAEngine()


def main():
    st.set_page_config(page_title="Private PDF Q&A", page_icon="📄", layout="wide")
    if not require_authentication():
        st.stop()

    user = current_user()
    user_id = user["id"]
    st.title("📄 Private PDF Question Answering")
    st.caption("Your documents are isolated by account. Only you can access your indexed PDFs.")

    try:
        conn = init_db()
    except Exception:
        event("database_initialization_failed", user_id=user_id)
        st.error("Could not connect to PostgreSQL or initialize the database. Check the deployment configuration.")
        st.stop()

    try:
        processor = get_pdf_processor()
        qa_engine = get_qa_engine()

        st.subheader("1. Add a PDF")
        uploaded_file = st.file_uploader("Choose a PDF file (20 MB maximum)", type=["pdf"])
        if uploaded_file and st.button("Process PDF", type="primary"):
            try:
                allowed, retry_ms = allow_request(user_id, "upload", limit=5, window_seconds=3600)
                if not allowed:
                    st.warning(f"Upload limit reached. Try again in about {max(1, retry_ms // 60000)} minute(s).")
                    st.stop()
                pdf_bytes = validate_upload(uploaded_file)
                content_hash = hashlib.sha256(pdf_bytes).hexdigest()
                existing = get_document_by_hash(conn, content_hash, user_id)
                if existing:
                    document_id, filename, _created_at = existing
                    st.session_state["selected_doc_id"] = document_id
                    st.info(f"Already indexed in your account as **{filename}**.")
                    st.rerun()

                with st.spinner("Extracting, OCR-ing, chunking, and embedding..."):
                    with timed("pdf_ingestion", user_id=user_id, bytes=len(pdf_bytes)):
                        processed = processor.process_pdf(pdf_bytes)
                        document_id, created = store_document(
                            conn, uploaded_file.name, processed["text"], content_hash, user_id
                        )
                        if created:
                            store_embeddings(
                                conn, document_id, processed["embeddings"],
                                processed["chunks"], user_id
                            )
                st.session_state["selected_doc_id"] = document_id
                st.success(f"Indexed **{uploaded_file.name}**: {len(processed['chunks'])} chunks.")
                st.rerun()
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:
                event("pdf_ingestion_failed", user_id=user_id, error_type=type(exc).__name__)
                st.error("PDF processing failed. Check the application logs for the error details.")

        documents = list_documents(conn, user_id)
        if not documents:
            st.info("Upload and process a PDF to get started.")
            return

        document_ids = [row[0] for row in documents]
        if st.session_state.get("selected_doc_id") not in document_ids:
            st.session_state["selected_doc_id"] = document_ids[0]

        st.subheader("2. Ask a question")
        selected_document_id = st.selectbox(
            "Search within your PDF",
            options=document_ids,
            format_func=lambda doc_id: next(
                (f"{filename} (uploaded {created_at:%Y-%m-%d})"
                 for item_id, filename, created_at in documents if item_id == doc_id),
                "Document",
            ),
            key="selected_doc_id",
        )

        with st.form("question_form"):
            question = st.text_input("Your question", placeholder="What are the key findings?")
            submitted = st.form_submit_button("Get answer")

        if submitted:
            try:
                question = validate_query(question)
                allowed, retry_ms = allow_request(user_id, "question", limit=30, window_seconds=60)
                if not allowed:
                    st.warning(f"Question limit reached. Retry in {max(1, retry_ms // 1000)} second(s).")
                    st.stop()
                with st.spinner("Retrieving evidence and generating an answer..."):
                    with timed("question_answering", user_id=user_id, document_id=selected_document_id):
                        result = qa_engine.search_documents(
                            conn, question, document_id=selected_document_id,
                            user_id=user_id, top_k=4
                        )
                st.subheader("Answer")
                st.markdown(result["response"])
                with st.expander("Retrieved evidence", expanded=True):
                    for index, chunk in enumerate(result["relevant_chunks"], start=1):
                        st.markdown(
                            f"**Source {index} · chunk {chunk['chunk_index']} · cosine distance {chunk['distance']:.3f}**"
                        )
                        st.write(chunk["text"])
            except ValueError as exc:
                st.warning(str(exc))
            except PermissionError:
                st.error("You do not have access to that document.")
            except Exception as exc:
                event("question_answering_failed", user_id=user_id, error_type=type(exc).__name__)
                st.error("Question answering failed. Check the application logs for details.")
    finally:
        cleanup_db(conn)


if __name__ == "__main__":
    main()
