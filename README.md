# RAG Paper Query System

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for querying research papers.  
It allows ingestion of PDFs, splitting them into chunks (word-based or token-based), and preparing them for embeddings and vector database storage.  
A Streamlit UI will be added in later steps.

---

## 🚀 Features
- Extract text from PDFs using [PyMuPDF](https://pymupdf.readthedocs.io/).
- Flexible chunking:
  - **Word-based** (simple, quick).
  - **Token-based** using [tiktoken](https://github.com/openai/tiktoken) (preferred for embeddings).
- Command-line interface with `argparse`.
- Dynamic path handling (works both in CLI and PyCharm).
- Modular design for extension with embeddings, ChromaDB, and Streamlit UI.

---

## Installation
1. Clone the repo:
```bash
git clone https://github.com/DarielPereira/RAG-paper-query.git
cd rag-paper-query
```
2. Create a conda environment:
```bash
conda create --name rag-paper-env python=3.11
conda activate rag-paper-env
```
3. Install dependencies:
```bash
pip install -r requirements
```
---

## Usage

1. Place PDFs in the data/ folder
2. Run ingestion:
- For token-based chunking (default)
```bash
python src/ingestion.py --chunking_type tokens
```
- or for word-based chunking
```bash
python src/ingestion.py --chunking_type words
```
- or for custom data folder
```bash
python src/ingestion.py --chunking_type tokens --data_dir ./papers
```
3. Output:
- Extracted text chunks are printed with metadata (title, chunk index, method).
- Example
```plaintext
Processing: data/example.pdf
Total chunks extracted: 42
Example chunk:
"This paper discusses..."
```




