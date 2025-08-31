import os
import fitz  # PyMuPDF
from typing import List

# -------------------------
# Config
# -------------------------
DATA_DIR = "../data"    # relative path to the data folder containing the PDF files
CHUNK_SIZE = 500        # approximate number of tokens per chunk

# -------------------------
# Functions
# -------------------------
def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    text = ""
    with fitz.open(pdf_path) as pdf:
        for page in pdf:
            text += page.get_text()
    return text

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks with roughly chunk_size tokens."""
    # Using simple white space-based chunking
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])   # concatenate chunk_size words using white spaces as connectors
        chunks.append(chunk)
    return chunks

def ingest_pdfs(data_dir: str = DATA_DIR) -> List[dict]:
    """Process all PDFs in data folder and return chunks with metadata"""
    all_chunks =[]
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            print(f"Processing: {pdf_path}")
            text = extract_text_from_pdf(pdf_path)
            chunks = chunk_text(text, CHUNK_SIZE)

            for idx, chunk in enumerate(chunks):
                all_chunks.append({
                    "content": chunk,
                    "metadata": {
                        "title": file_name,
                        "chunk index": idx
                    }
                })
    return all_chunks


#-------------------------
# Test/Run
# ------------------------
if __name__ == "__main__":
    chunks = ingest_pdfs()
    print(f"Total chunks extracted: {len(chunks)}")
    print(f"Example chunk: {chunks[22]['content'][:500]}")

