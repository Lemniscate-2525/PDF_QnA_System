import pytest

from app.pdf_processor import PDFProcessor


@pytest.fixture
def processor():
    # Chunking and cleaning do not need to load the embedding model.
    return PDFProcessor.__new__(PDFProcessor)


def test_chunking_creates_multiple_chunks_with_overlap(processor):
    text = " ".join(f"token{i}" for i in range(100))
    chunks = processor.create_chunks(text, chunk_size=100, overlap=20)

    assert len(chunks) > 1
    assert all(chunk for chunk in chunks)
    assert all(len(chunk) <= 100 for chunk in chunks)
    assert all(
        __import__("re").fullmatch(r"token\\d+", word)
        for chunk in chunks
        for word in chunk.split()
    )
    assert any(
        set(chunks[index].split()) & set(chunks[index + 1].split())
        for index in range(len(chunks) - 1)
    )


def test_chunking_rejects_invalid_overlap(processor):
    with pytest.raises(ValueError):
        processor.create_chunks("some text", chunk_size=10, overlap=10)


def test_clean_text_preserves_unicode(processor):
    assert processor.clean_text("café   東京\nvalue") == "café 東京 value"


def test_empty_text_returns_no_chunks(processor):
    assert processor.create_chunks("   ") == []
