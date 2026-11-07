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


@st.fragment(run_every="3s")
def render_ingestion_status():
    """Poll active RQ jobs without requiring a manual browser refresh."""
    active_jobs = st.session_state.get("active_ingestion_jobs", [])
    if not active_jobs:
        return

    remaining = []
    completed = []
    failed = []

    for item in active_jobs:
        job_id = item["job_id"]
        filename = item["filename"]
        try:
            job = get_ingestion_job(job_id)
            status = str(job.get_status(refresh=True)).upper()
            if job.is_finished:
                completed.append(f"**{filename}** — ingestion completed.")
            elif job.is_failed:
                failed.append(f"**{filename}** — ingestion failed. Check Worker logs.")
            else:
                st.info(f"**{filename}** — {status.lower()} · Job {job_id}")
                remaining.append(item)
        except Exception:
            st.caption(f"Status temporarily unavailable for **{filename}**.")

    st.session_state["active_ingestion_jobs"] = remaining
    if completed or failed:
        st.session_state.setdefault("ingestion_notices", []).extend(completed + failed)
        # A full rerun refreshes the document list once a job reaches a terminal state.
        st.rerun()


def main():
    st.set_page_config(page_title="DocuMind | PDF Q&A", page_icon="📄", layout="wide")
    if not require_authentication():
        st.stop()

    user = current_user()
    user_id = user["id"]
    st.title("🧠 DocuMind · PDF Q&A")
    st.caption("Your documents are isolated by account. Only you can access your indexed PDFs.")

    try:
        conn = init_db()
    except Exception:
        event("database_initialization_failed", user_id=user_id)
        st.error("Could not connect to PostgreSQL or initialize the database. Check the deployment configuration.")
        st.stop()

    try:
        qa_engine = get_qa_engine()

        st.subheader("1. Add PDFs")
        uploaded_files = st.file_uploader(
            "Choose one or more PDF files (20 MB maximum per file)",
            type=["pdf"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button("Process selected PDFs", type="primary"):
            batch_hashes = set()
            queued_count = 0
            for uploaded_file in uploaded_files:
                try:
                    pdf_bytes = validate_upload(uploaded_file)
                    content_hash = hashlib.sha256(pdf_bytes).hexdigest()

                    # Avoid re-enqueueing documents already indexed for this account.
                    existing = get_document_by_hash(conn, content_hash, user_id)
                    if existing:
                        document_id, filename, _created_at = existing
                        st.session_state["selected_doc_id"] = document_id
                        st.info(f"Already ingested: **{uploaded_file.name}** matches **{filename}** in your account. Skipping reprocessing.")
                        continue

                    # Also avoid duplicate files selected together in one batch.
                    if content_hash in batch_hashes:
                        st.info(f"Already included in this upload batch: **{uploaded_file.name}**. Skipping duplicate.")
                        continue
                    batch_hashes.add(content_hash)

                    allowed, retry_ms = allow_request(
                        user_id, "upload", limit=5, window_seconds=3600
                    )
                    if not allowed:
                        st.warning(
                            f"Upload limit reached. Try again in about "
                            f"{max(1, retry_ms // 60000)} minute(s)."
                        )
                        continue

                    safe_name = sanitize_filename(uploaded_file.name)
                    with st.spinner(f"Uploading {safe_name} and queuing ingestion..."):
                        job_id = enqueue_pdf(
                            user_id, safe_name, content_hash, pdf_bytes
                        )
                    st.session_state.setdefault("active_ingestion_jobs", []).append(
                        {"job_id": job_id, "filename": safe_name}
                    )
                    queued_count += 1
                    st.success(f"Queued **{safe_name}** · Job ID: {job_id}")
                except ValueError as exc:
                    st.warning(f"{uploaded_file.name}: {exc}")
                except Exception as exc:
                    event(
                        "pdf_enqueue_failed",
                        user_id=user_id,
                        error_type=type(exc).__name__,
                    )
                    st.error(
                        f"Could not queue **{uploaded_file.name}**. "
                        "Check Redis, storage bucket, and Worker configuration."
                    )
            if queued_count:
                st.info(
                    f"Queued {queued_count} new PDF(s). Processing status updates automatically below."
                )

        notices = st.session_state.pop("ingestion_notices", [])
        for notice in notices:
            if "failed" in notice.lower():
                st.error(notice)
            else:
                st.success(notice)

        render_ingestion_status()

        documents = list_documents(conn, user_id)
        if not documents:
            st.info("Upload and process PDFs to get started.")
            return

        document_ids = [row[0] for row in documents]
        if st.session_state.get("selected_doc_id") not in document_ids:
            st.session_state["selected_doc_id"] = document_ids[0]

        st.subheader("2. Ask a question")
        selected_document_id = st.selectbox(
            "Search within your PDFs",
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
