# RAG Paper Query System

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for querying research papers.  
It allows ingestion of PDFs, splitting them into chunks (word-based or token-based). Chunks are embedded and prepared for vector database storage.  
A Streamlit UI will be added in later steps.

---

## 🚀 Current Features
- Extract text from PDFs using [PyMuPDF](https://pymupdf.readthedocs.io/).
- Flexible chunking:
  - **Word-based** (simple, quick).
  - **Token-based** using [tiktoken](https://github.com/openai/tiktoken) (preferred for embeddings).
- Embedding generation.

---

## Features To Do
- Set up DB for embedding storage.
- Implement retrieval and generation pipeline.
- Build Streamlit UI for querying.

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

## Module Testing
### Ingestion and Chunking
To test the ingestion and chunking of PDFs:
1. Place PDFs in the data/ folder
2. Run ingestion:
- For token-based chunks (default)
```bash
python src/ingestion.py --chunking_type tokens 
```
- or for word-based chunking
```bash
python src/igestion.py --chunking_type words
```
- or for custom data folder
```bash
python src/ingestion.py --chunking_type tokens --data_dir ./papers
```
3. Output:
- Extracted text chunks are printed with metadata (title, chunk index, method).
- Example
```console
Chunking type: words
Ingesting PDFs: 100%|████████████████████████...██████████████████████████████████████| 3/3 [00:00<00:00,  7.20it/s] 
Total PDFs processed: 3
Total chunks extracted: 129
Example chunk:
 '2020, respectively. She was a Postdoctoral Fellow ...'
Metadata: {'title': 'Han et al. - 2023 - Toward Extra Large-Scale MIMO New Channel Propert.pdf', 'chunk index': 45}
```
### Embedding Generation
To test embedding generation:
1. Ensure you have set your OpenAI API key in the environment:
2. Run the embeddings script:
- For embedding of '--limit' first '--chuking_type'-based chunks at '--data_dir'
```bash
python src/embeddings.py --chunking_type tokens --limit 10 --data_dir ./data
```
3. Output:
- Embedding details are printed for each chunk.
- Example:
```console
Chunking type: tokens
Ingesting PDFs: 100%|██████████████████...███████████████████████████████████████████████| 3/3 [00:00<00:00,  5.11it/s] 
Total PDFs processed: 3
Total chunks extracted: 248
Generating embeddings: 100%|█████...████████████████████████████████████████████| 3/3 [00:01<00:00,  2.58it/s] 
Embeddings generated for 3 chunks.
Chunk 0 embedding length: 3072
Chunk 0 content (first 100 chars): IEEE INTERNET OF THINGS JOURNAL, VOL. 10, NO. 16, 15 AUGUST 2023
14569
Toward Extra Large-Scale MIMO
Chunk 0 metadata: {'title': 'Han et al. - 2023 - Toward Extra Large-Scale MIMO New Channel Propert.pdf', 'chunk index': 0}
Chunk 1 embedding length: 3072
Chunk 1 content (first 100 chars):  of publication 5 May 2023; date of current ver-
sion 8 August 2023. This work was supported in part
Chunk 1 metadata: {'title': 'Han et al. - 2023 - Toward Extra Large-Scale MIMO New Channel Propert.pdf', 'chunk index': 1}
Chunk 2 embedding length: 3072
Chunk 2 content (first 100 chars): Terms—Distributed
processing,
extra
large-scale
multiple-input–multiple-output (MIMO), low-cost arch
Chunk 2 metadata: {'title': 'Han et al. - 2023 - Toward Extra Large-Scale MIMO New Channel Propert.pdf', 'chunk index': 2}
Embedding generation completed.
```



