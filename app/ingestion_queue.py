import uuid

from redis import Redis
from rq import Queue, Retry
from supabase import create_client

from app.config import REDIS_URL, SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, UPLOAD_BUCKET


def enqueue_pdf(user_id, filename, content_hash, pdf_bytes):
    if not REDIS_URL or not SUPABASE_SERVICE_ROLE_KEY:
        raise RuntimeError("Async ingestion requires REDIS_URL and SUPABASE_SERVICE_ROLE_KEY.")
    storage = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)
    object_path = f"{user_id}/{uuid.uuid4().hex}.pdf"
    storage.storage.from_(UPLOAD_BUCKET).upload(
        object_path, pdf_bytes,
        file_options={"content-type": "application/pdf", "upsert": "false"},
    )
    try:
        queue = Queue("pdf-ingestion", connection=Redis.from_url(REDIS_URL))
        job = queue.enqueue(
            "app.worker.process_ingestion",
            user_id, object_path, filename, content_hash,
            job_timeout=1800,
            result_ttl=86400,
            failure_ttl=604800,
            retry=Retry(max=2, interval=[30, 120]),
        )
        return job.id
    except Exception:
        try:
            storage.storage.from_(UPLOAD_BUCKET).remove([object_path])
        except Exception:
            pass
        raise


def get_ingestion_job(job_id):
    from rq.job import Job
    connection = Redis.from_url(REDIS_URL)
    return Job.fetch(job_id, connection=connection)
