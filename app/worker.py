from supabase import create_client

from app.config import SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, UPLOAD_BUCKET
from app.database import init_db, get_document_by_hash, store_document, store_embeddings, cleanup_db
from app.observability import event, timed
from app.pdf_processor import PDFProcessor


def process_ingestion(user_id, object_path, filename, content_hash):
    """RQ worker entrypoint. Worker must run with the same environment as the app."""
    storage = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    conn = None
    try:
        with timed("background_pdf_ingestion", user_id=user_id):
            existing = storage.storage.from_(UPLOAD_BUCKET).download(object_path)
            conn = init_db()
            duplicate = get_document_by_hash(conn, content_hash, user_id)
            if duplicate:
                return {"status": "already_indexed", "document_id": duplicate[0]}

            processed = PDFProcessor().process_pdf(existing)
            document_id, created = store_document(
                conn, filename, processed["text"], content_hash, user_id
            )
            if created:
                store_embeddings(
                    conn, document_id, processed["embeddings"],
                    processed["chunks"], user_id
                )
            event("background_ingestion_completed", user_id=user_id, document_id=document_id,
                  chunk_count=len(processed["chunks"]))
            return {"status": "completed", "document_id": document_id}
    except Exception as exc:
        event("background_ingestion_failed", user_id=user_id, error_type=type(exc).__name__)
        raise
    finally:
        cleanup_db(conn)
        try:
            storage.storage.from_(UPLOAD_BUCKET).remove([object_path])
        except Exception:
            pass
