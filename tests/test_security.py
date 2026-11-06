from types import SimpleNamespace

import pytest

from app.security import validate_query, validate_upload
from app.redis_store import cache_key


def test_validate_query_trims_whitespace():
    assert validate_query("  explain the conclusion  ") == "explain the conclusion"


@pytest.mark.parametrize("value", ["", "   ", "   \n"])
def test_validate_query_rejects_empty(value):
    with pytest.raises(ValueError):
        validate_query(value)


def test_validate_query_rejects_oversized_input():
    with pytest.raises(ValueError):
        validate_query("x" * 2001)


def test_validate_upload_accepts_pdf():
    upload = SimpleNamespace(name="paper.pdf", getvalue=lambda: b"%PDF-1.7 test")
    assert validate_upload(upload) == b"%PDF-1.7 test"


@pytest.mark.parametrize("name,data", [
    ("paper.txt", b"%PDF-1.7"),
    ("paper.pdf", b"not a pdf"),
    ("paper.pdf", b""),
])
def test_validate_upload_rejects_invalid_files(name, data):
    upload = SimpleNamespace(name=name, getvalue=lambda: data)
    with pytest.raises(ValueError):
        validate_upload(upload)


def test_cache_keys_are_user_scoped_and_not_plaintext():
    key_a = cache_key("answer", "user-a", 12, "question")
    key_b = cache_key("answer", "user-b", 12, "question")
    assert key_a != key_b
    assert "question" not in key_a


def test_cache_key_is_scoped_to_document_and_user():
    from app.redis_store import cache_key

    assert cache_key("answer", "u1", 1, "same question") != cache_key(
        "answer", "u1", 2, "same question"
    )
    assert cache_key("answer", "u1", 1, "same question") != cache_key(
        "answer", "u2", 1, "same question"
    )
