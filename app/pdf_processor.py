from PyPDF2 import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np
import re

class PDFProcessor:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def clean_text(self, text):
        # Remove special characters and normalize whitespace
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
        text = re.sub(r'\s+', ' ', text)
        # Handle encoding by replacing problematic characters
        text = text.encode('ascii', errors='ignore').decode('ascii')
        return text.strip()
    
    def extract_text(self, pdf_file):
        try:
            reader = PdfReader(pdf_file)
            text = ""
            for page in reader.pages:
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += self.clean_text(page_text) + " "
                except Exception as e:
                    print(f"Error extracting text from page: {str(e)}")
                    continue
            return text.strip()
        except Exception as e:
            print(f"Error processing PDF: {str(e)}")
            return ""
    
    def create_chunks(self, text, chunk_size=1000, overlap=100):
        if not text:
            return []
        
        # Split text into sentences first
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            sentence_size = len(sentence)
            
            if sentence_size > chunk_size:
                # Handle very long sentences by splitting them
                words = sentence.split()
                for i in range(0, len(words), chunk_size):
                    chunk = " ".join(words[i:i + chunk_size])
                    if chunk:
                        chunks.append(chunk)
                continue
                
            if current_size + sentence_size > chunk_size:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    
    def generate_embeddings(self, text_chunks):
        if not text_chunks:
            return np.array([])
            
        embeddings = []
        for chunk in text_chunks:
            try:
                if chunk.strip():  # Only process non-empty chunks
                    embedding = self.model.encode(chunk)
                    embeddings.append(embedding)
            except Exception as e:
                print(f"Error generating embedding: {str(e)}")
                continue
                
        return np.array(embeddings) if embeddings else np.array([])

    def process_pdf(self, pdf_file, chunk_size=1000):
        """Convenience method to process a PDF file end-to-end"""
        try:
            text = self.extract_text(pdf_file)
            if not text:
                raise ValueError("No text could be extracted from the PDF")
                
            chunks = self.create_chunks(text, chunk_size)
            if not chunks:
                raise ValueError("No chunks could be created from the text")
                
            embeddings = self.generate_embeddings(chunks)
            return {
                'text': text,
                'chunks': chunks,
                'embeddings': embeddings
            }
        except Exception as e:
            print(f"Error in PDF processing pipeline: {str(e)}")
            return None