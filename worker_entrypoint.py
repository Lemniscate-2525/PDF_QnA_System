from redis import Redis
from rq import Queue, Worker

from app.config import REDIS_URL


if __name__ == "__main__":
    if not REDIS_URL:
        raise RuntimeError("REDIS_URL is required.")
    connection = Redis.from_url(REDIS_URL)
    Worker([Queue("pdf-ingestion", connection=connection)], connection=connection).work(with_scheduler=True)
