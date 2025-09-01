import os
from pathlib import Path
import fitz  # PyMuPDF
from typing import List
import tiktoken
import argparse


# -------------------------
# Config
# -------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # project root path
DATA_DIR = PROJECT_ROOT/"data"    # path to data folder containing the PDF files
MODEL_ENCODING = "cl100k_base"    # Encoding name for tiktoken; should match the model used for chunking
CHUNK_SIZE = 500        # approximate number of tokens per chunk
CHUNK_OVERLAP = 50      # overlapping tokens for context continuity

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

def chunk_text_by_words(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """Split text into chunks with roughly chunk_size words."""
    # Using simple white space-based chunking
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])   # concatenate chunk_size words using white spaces as connectors
        chunks.append(chunk)
    return chunks

def chunk_text_by_tokens(text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into chunks with roughly chunk_size tokens with overlap."""
    encoding = tiktoken.get_encoding(MODEL_ENCODING)
    tokens = encoding.encode(text)

    chunks = []
    for i in range(0, len(tokens), chunk_size-chunk_overlap):
        chunk_tokens = tokens[i:i+chunk_size]
        chunk_text = encoding.decode(chunk_tokens)
        chunks.append(chunk_text)
    return chunks

def ingest_pdfs(data_dir: str = DATA_DIR, chunking_type: str = 'tokens') -> List[dict]:
    """Process all PDFs in data folder and return chunks with metadata"""
    all_chunks =[]
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            print(f"Processing: {pdf_path}")
            print(f"Chunking type: {chunking_type}")
            text = extract_text_from_pdf(pdf_path)

            match chunking_type:
                case "tokens":
                    chunks = chunk_text_by_tokens(text)
                case "words":
                    chunks = chunk_text_by_words(text)
                case _:
                    raise ValueError(f"Invalid chunking_type: {chunking_type}. Must be 'tokens' or 'words'.")

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
    parser = argparse.ArgumentParser(description = "Ingest PDFs into text chunks.")
    parser.add_argument(
        "--chunking_type",
        type = str,
        default = "tokens",
        choices = ["tokens", "words"],
        help = "Chunking type: 'tokens' (default) or 'words'."
            )
    parser.add_argument(
        "--data_dir",
        type = str,
        default = DATA_DIR,
        help = "Directory where PDFs are located."
    )
    args = parser.parse_args()

    chunks = ingest_pdfs(data_dir = args.data_dir, chunking_type= args.chunking_type)
    print(f"Total chunks extracted: {len(chunks)}")
    print(f"Example chunk: {chunks[45]['content'][:500]}")

