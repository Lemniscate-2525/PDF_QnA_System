from functools import lru_cache
import re

import pymupdf as fitz
import numpy as np
import pytesseract
from PIL import Image
from sentence_transformers import SentenceTransformer
from pytesseract.pytesseract import TesseractNotFoundError

from app.config import OCR_ENABLED, OCR_LANGUAGES


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load one embedding model per Python process and reuse it across reruns."""
    return SentenceTransformer("all-MiniLM-L6-v2")


class PDFProcessor:
    def __init__(self, model=None):
        self.model = model or get_embedding_model()

    def clean_text(self, text):
        # Preserve Unicode; remove only control characters that commonly corrupt extracted text.
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text or "")
        return re.sub(r"\s+", " ", text).strip()

    def extract_text(self, pdf_source, ocr_enabled=OCR_ENABLED):
        """Extract text page by page, OCR-ing pages with little or no selectable text.

        pdf_source may be raw PDF bytes, a file-like object, or a filesystem path.
        OCR requires the Tesseract executable to be installed separately.
        """
        if isinstance(pdf_source, (bytes, bytearray)):
            document = fitz.open(stream=bytes(pdf_source), filetype="pdf")
        elif hasattr(pdf_source, "read"):
            document = fitz.open(stream=pdf_source.read(), filetype="pdf")
        else:
            document = fitz.open(pdf_source)

        page_texts = []
        try:
            if document.page_count > 250:
                raise ValueError("PDF exceeds the 250-page processing limit.")
            for page_number, page in enumerate(document, start=1):
                page_text = page.get_text("text") or ""
                if ocr_enabled and len(page_text.strip()) < 40:
                    try:
                        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                        image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                        ocr_text = pytesseract.image_to_string(image, lang=OCR_LANGUAGES)
                        if ocr_text.strip():
                            page_text = ocr_text
                    except TesseractNotFoundError as exc:
                        raise RuntimeError(
                            "OCR is enabled but Tesseract is not installed or not on PATH. "
                            "Install Tesseract, or set OCR_ENABLED=false in .env for text-based PDFs."
                        ) from exc
                    except pytesseract.TesseractError as exc:
                        raise RuntimeError(
                            f"OCR failed on PDF page {page_number}. Check the installed Tesseract language data."
                        ) from exc

                cleaned = self.clean_text(page_text)
                if cleaned:
                    page_texts.append(cleaned)
        finally:
            document.close()

        return " ".join(page_texts).strip()

    def create_chunks(self, text, chunk_size=1000, overlap=150):
        """Create character-sized windows with real overlap and whitespace-aware boundaries."""
        if not text or not text.strip():
            return []
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be >= 0 and smaller than chunk_size.")

        text = re.sub(r"\s+", " ", text).strip()
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(start + chunk_size, text_length)
            if end < text_length:
                # Prefer a natural boundary near the end without creating tiny chunks.
                boundary_floor = start + int(chunk_size * 0.60)
                sentence_boundary = max(
                    text.rfind(". ", boundary_floor, end),
                    text.rfind("? ", boundary_floor, end),
                    text.rfind("! ", boundary_floor, end),
                )
                whitespace_boundary = text.rfind(" ", boundary_floor, end)
                boundary = sentence_boundary if sentence_boundary > boundary_floor else whitespace_boundary
                if boundary > start:
                    end = boundary + 1

            chunk = text[start:end].strip()
            if chunk and (not chunks or chunk != chunks[-1]):
                chunks.append(chunk)
            if end >= text_length:
                break

            next_start = max(start + 1, end - overlap)
            # Never begin the next chunk in the middle of a word.
            next_space = text.find(" ", next_start, end)
            if next_space != -1:
                next_start = next_space + 1
            else:
                next_start = end
            start = next_start if next_start > start else end

        return chunks

    def generate_embeddings(self, text_chunks):
        if not text_chunks:
            return np.empty((0, 384), dtype=np.float32)
        embeddings = self.model.encode(
            text_chunks,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def process_pdf(self, pdf_source, chunk_size=1000, overlap=150):
        text = self.extract_text(pdf_source)
        if not text:
            raise ValueError(
                "No text could be extracted. If this is a scanned PDF, install Tesseract OCR and retry."
            )
        chunks = self.create_chunks(text, chunk_size=chunk_size, overlap=overlap)
        if not chunks:
            raise ValueError("No chunks could be created from the extracted PDF text.")
        embeddings = self.generate_embeddings(chunks)
        return {"text": text, "chunks": chunks, "embeddings": embeddings}
