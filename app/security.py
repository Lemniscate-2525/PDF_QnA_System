import re

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 250
MAX_QUERY_CHARS = 2000
MAX_FILENAME_CHARS = 180


def validate_upload(uploaded_file):
    if uploaded_file is None:
        raise ValueError("Choose a PDF first.")
    name = uploaded_file.name or ""
    if not name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are accepted.")
    if len(name) > MAX_FILENAME_CHARS:
        raise ValueError("Filename is too long.")
    data = uploaded_file.getvalue()
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("PDF must be non-empty and no larger than 20 MB.")
    if not data.startswith(b"%PDF-"):
        raise ValueError("The uploaded file is not a valid PDF.")
    return data


def validate_query(query):
    query = (query or "").strip()
    if not query:
        raise ValueError("Enter a question first.")
    if len(query) > MAX_QUERY_CHARS:
        raise ValueError(f"Question must be at most {MAX_QUERY_CHARS} characters.")
    if not re.search(r"[\w]", query, flags=re.UNICODE):
        raise ValueError("Question must contain readable text.")
    return query
