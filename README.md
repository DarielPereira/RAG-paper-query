# RAG Paper Query System

This project implements a **Retrieval-Augmented Generation (RAG)** pipeline for querying research papers.  
It allows ingestion of PDFs, splitting them into chunks (word-based or token-based). Chunks are embedded and stored in 
vector database storage (cloud Azure AI search or local ChromaDB). Queries can be made to retrieve relevant chunks 
and generate answers using OpenAI's language models.
A Streamlit UI will be added in later steps.

---

## 🚀 Current Features
- Extract text from PDFs using [PyMuPDF](https://pymupdf.readthedocs.io/).
- Flexible chunking:
  - **Word-based** (simple, quick).
  - **Token-based** using [tiktoken](https://github.com/openai/tiktoken) (preferred for embeddings).
- Embedding generation.
- Store in vector DB (Azure AI search and ChromaDB)
- Retrieval from DB and query answer.

---

## Features To Do
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
### Embedding Generation (it incorporates ingestion and chunking)
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
### Vector DB Storage (it incorporates ingestion, chunking and embedding generation)    
To test storage in vector DB:
1. Ensure you have set your API keys in the environment:
   - For Azure AI Search, make sure to set the key and endpoint (otherwise it would fallback to local chromaDB).
2. Run the vector DB storage script:
- For storing '--limit' first '--chuking_type'-based chunks at '--data_dir
```bash
python src/vector_db.py --chunking_type tokens --limit 10 --data_dir ./data 
```
3. Output:
- Storage details are printed for each chunk.
- Example:
```console
Starting vector database pipeline...
Local ChromaDB collection ready
Using local ChromaDB as vector database.
Chunking type: tokens
Ingesting PDFs: 100%|███████████...█████████████████████████████████████████| 3/3 [00:00<00:00,  4.98it/s]
Total PDFs processed: 3
Extracted 1478 from PDFs.
Processing 10 chunks for embedding and upsert.
Generating embeddings: 100%|██████████...████████████████████████████████| 10/10 [00:01<00:00,  5.17it/s]
Generated embeddings for 10 chunks.
Upserted 10 documents to ChromaDB.
Upsert completed.
```
### Querying (it assumes you have already stored chunks in vector DB)
To test querying from vector DB:
1. Ensure you have set your API keys in the environment:
   - For Azure AI Search, make sure to set the key and endpoint (otherwise it would fallback to local chromaDB).
   - Make sure you have already ingested, chunked, embedded and stored some PDFs in the vector DB in .chroma folder.
   - Make sure you have set your OpenAI API key in the environment for generating answers.
2. Run the query script:
- For answering the query in '--query' and retrieving top '--top_k' reference chunks
```bash
python src/query.py --query "What are the benefits of extra large-scale MIMO?" --top_k 3
```
3. Output:
- Retrieved chunks and generated answer are printed.
- Example:
```console
Local ChromaDB collection ready
Using local ChromaDB as vector database.

RAG Answer for query: 'tell me about massive mimo'

Massive MIMO (Multiple Input Multiple Output) refers to a technology that employs a large number of antennas at the base station to serve multiple users simultaneously, enhancing spectral efficiency and improving signal quality. It is particularly effective in dense urban environments and is often integrated with distributed antenna systems to optimize performance.

Key aspects of massive MIMO include:

1. **User Capacity**: More antennas at the user side can accommodate more users, which is crucial for high-demand scenarios (Feng et al., IEEE Communications Letters).
2. **Beamforming Techniques**: Techniques like sparse multicast beamforming and multicell beamforming are utilized to manage interference and improve signal strength (Feng et al., IEEE Trans. Wireless Commun.; Zhao et al., IEEE Trans. Signal Process.).
3. **Joint Optimization**: The optimization of precoding matrices and selection of active remote radio heads (RRHs) is essential for minimizing network power consumption (NPC) and maximizing utility in MIMO systems (Feng et al., IEEE Communications Letters).

For further details, refer to the works of Feng, Ge, and Lu on hierarchical transmission optimization, as well as Zhao, Quek, and Lei on coordinated multipoint transmission.

References:
1. Pan et al. - 2017 - Joint Precoding and RRH Selection for User-Centric.pdf (chunk 1674)
Snippet:  “Hierarchical transmission optimization for
massively dense distributed antenna systems,” IEEE Communications
Letters, vol. 19, no. 4, pp...

2. Pan et al. - 2017 - Joint Precoding and RRH Selection for User-Centric.pdf (chunk 1673)
Snippet: 34] W. Feng, N. Ge, and J. Lu, “Hierarchical transmission optimization for
massively dense distributed antenna systems,” IEEE...

(...)

10. Pan et al. - 2017 - Joint Precoding and RRH Selection for User-Centric.pdf (chunk 1637)
Snippet:  MIMO applications,” International Journal of Antennas and
Propagation, vol. 2014, 2014.
[26] K. Wang...
```

