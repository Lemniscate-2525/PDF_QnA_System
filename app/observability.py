import json
import logging
import time
from contextlib import contextmanager

logger = logging.getLogger("pdf_qna")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def event(name, **fields):
    logger.info(json.dumps({"event": name, **fields}, default=str))


@contextmanager
def timed(name, **fields):
    start = time.perf_counter()
    try:
        yield
        event(name, outcome="success", duration_ms=round((time.perf_counter()-start)*1000, 2), **fields)
    except Exception as exc:
        event(name, outcome="error", error_type=type(exc).__name__, duration_ms=round((time.perf_counter()-start)*1000, 2), **fields)
        raise
