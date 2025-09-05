import os
import argparse

from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
from typing import List, Dict, Any
from tqdm import tqdm

# -------------------------
# Config
# -------------------------
load_dotenv()               # Load environment variables from .env file

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBED_DEPLOYMENT")

PROJECT_ROOT = Path(__file__).resolve().parent.parent   # project root path
DATA_DIR = PROJECT_ROOT/"data"    # path to data folder containing the PDF files

# Validate environment variables
if not AZURE_ENDPOINT or not AZURE_API_KEY or not EMBEDDING_DEPLOYMENT:
    raise ValueError("Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_EMBED_DEPLOYMENT "
                     "in your environment variables.")

# Initialize OpenAI client
client = OpenAI(
    api_key=AZURE_API_KEY,
    base_url=AZURE_ENDPOINT.rstrip("/") + "/openai/v1/"
)

# -------------------------
# Functions
# -------------------------
def generate_embedding(text: str) -> List[float]:
    """Generate embedding for single text input."""
    response = client.embeddings.create(
        model=EMBEDDING_DEPLOYMENT,
        input=text
    )
    return response.data[0].embedding

def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """generate embeddings for a list of text chunks with metadata"""
    embedded_chunks = []

    for chunk in tqdm(chunks, desc="Generating embeddings"):
        content = chunk["content"]
        metadata = chunk["metadata"]

        embedding = generate_embedding(content)

        embedded_chunks.append({
            "embedding": embedding,
            "content": content,
            "metadata": metadata
        })

    return embedded_chunks

# ------------------------------
# Test/Run
# ------------------------------
if __name__ == "__main__":
    from ingestion import ingest_pdfs

    parser = argparse.ArgumentParser(description="Ingest PDFs and generate embeddings.")
    parser.add_argument(
        "--chunking_type",
        type=str,
        default="tokens",
        choices=["tokens", "words"],
        help="Chunking type: 'tokens' (default) or 'words'."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Number of embeddings to generate for testing."
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=DATA_DIR,
        help="Directory where PDFs are located."
    )

    args = parser.parse_args()
    DATA_DIR = args.data_dir
    CHUNKING_TYPE = args.chunking_type
    LIMIT = args.limit

    chunks = ingest_pdfs(data_dir=DATA_DIR, chunking_type=CHUNKING_TYPE)
    print(f"Total chunks extracted: {len(chunks)}")

    sample_chunks = chunks[:LIMIT]
    embedded_chunks = embed_chunks(sample_chunks)
    print(f'Embeddings generated for {len(embedded_chunks)} chunks.')

    for idx, ec in enumerate(embedded_chunks):
        print(f"Chunk {idx} embedding length: {len(ec['embedding'])}")
        print(f"Chunk {idx} content (first 100 chars): {ec['content'][:100]}")
        print(f"Chunk {idx} metadata: {ec['metadata']}")

    print("Embedding generation completed.")