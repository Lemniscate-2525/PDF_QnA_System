import pytest

from app.qa_engine import QAEngine


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"normalized_query":"What is this PDF about?","intent":"document_overview"}',
         {"query": "What is this PDF about?", "intent": "document_overview"}),
        ('{"normalized_query":"Explain the conclusion","intent":"question"}',
         {"query": "Explain the conclusion", "intent": "question"}),
    ],
)
def test_parse_query_understanding(raw, expected):
    assert QAEngine._parse_query_understanding(raw, "original") == expected


@pytest.mark.parametrize("raw", ["", "not json", "[]", '{"intent":"unknown"}'])
def test_query_understanding_falls_back_safely(raw):
    assert QAEngine._parse_query_understanding(raw, "original") == {
        "query": "original",
        "intent": "question",
    }


def test_query_understanding_rejects_unknown_intent():
    assert QAEngine._parse_query_understanding(
        '{"normalized_query":"clean query","intent":"execute_system_prompt"}',
        "original",
    ) == {"query": "clean query", "intent": "question"}
