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
)
from app.observability import event, timed
from app.qa_engine import QAEngine
from app.security import validate_query, validate_upload, sanitize_filename
from app.redis_store import allow_request
from app.ingestion_queue import enqueue_pdf, get_ingestion_job


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

                with st.spinner("Uploading securely and queuing background ingestion..."):
                    safe_name = sanitize_filename(uploaded_file.name)
                    job_id = enqueue_pdf(user_id, safe_name, content_hash, pdf_bytes)
                st.session_state["last_ingestion_job"] = job_id
                st.success(f"PDF queued for processing. Job ID: {job_id}")
            except ValueError as exc:
                st.warning(str(exc))
            except Exception as exc:
                event("pdf_enqueue_failed", user_id=user_id, error_type=type(exc).__name__)
                st.error("Could not queue PDF processing. Check Redis, storage bucket, and worker configuration.")



        job_id = st.session_state.get("last_ingestion_job")
        if job_id:
            try:
                job = get_ingestion_job(job_id)
                st.info(f"Latest ingestion job: {job.get_status()} | {job_id}")
                if job.is_finished:
                    st.success(f"Ingestion finished: {job.result}")
                    st.session_state.pop("last_ingestion_job", None)
                    st.rerun()
                elif job.is_failed:
                    st.error("Ingestion failed. Check the worker logs.")
            except Exception:
                st.caption("Ingestion status is temporarily unavailable.")

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
