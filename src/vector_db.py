import os

import numpy as np
import requests
from dotenv import load_dotenv
from pathlib import Path
import random

# Load environment variables
load_dotenv()

# --------------------------
# Config
# --------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # project root path
DATA_DIR = PROJECT_ROOT/"data"    # path to data folder containing the PDF files
CHROMADB_DIR = PROJECT_ROOT/"chroma"  # path to local ChromaDB storage

# --------------------------
# Azure AI Search Setup
# --------------------------
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SimpleField,
    SearchableField,
    SearchFieldDataType,
    SearchField,
    VectorSearch,
    VectorSearchProfile,
    HnswAlgorithmConfiguration
)
from azure.search.documents import SearchClient

# --------------------------
# Local DB config
# --------------------------
import chromadb
from chromadb.config import Settings

# --------------------------
# vector DB Class
# --------------------------

class VectorDB:
    def __init__(self):
        self.use_azure = False
        self.use_chromadb = False

        if self._azure_service_available():
            # Endpoint reachable → use Azure
            self._init_azure()
            self.use_azure = True
            print("Using Azure AI Search as vector database.")
        else:
            # Fallback to local ChromaDB
            self._init_chromadb()
            self.use_chromadb = True
            print("Using local ChromaDB as vector database.")

    # Azure AI Search availability check
    def _azure_service_available(self):
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        api_key = os.getenv("AZURE_SEARCH_API_KEY")
        if not endpoint or not api_key:
            return False
        try:
            url = f"{endpoint}/indexes?api-version=2023-07-01-Preview"
            headers = {"api-key": api_key}
            r = requests.get(url, headers=headers, timeout=5)
            return r.status_code == 200
        except requests.RequestException:
            return False

    # Initialize Azure AI Search
    def _init_azure(self):
        endpoint = os.getenv("AZURE_SEARCH_ENDPOINT")
        api_key = os.getenv("AZURE_SEARCH_API_KEY")
        index_name = os.getenv("AZURE_SEARCH_INDEX_NAME")
        if not endpoint or not api_key or not index_name:
            raise ValueError("Please set AZURE_SEARCH_ENDPOINT, AZURE_SEARCH_API_KEY, and AZURE_SEARCH_INDEX_NAME "
                             "in your environment variables.")
        # Connect to Azure
        credential = AzureKeyCredential(api_key)
        index_client = SearchIndexClient(endpoint=endpoint, credential=credential)

        # Create index
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SearchField(
                name="embedding",
                type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                searchable=True,
                vector_search_dimensions=3072,  # embedding size
                vector_search_profile_name="default",
            ),
            SearchableField(name="metadata_title", type=SearchFieldDataType.String, filterable=True),
            SimpleField(name="metadata_chunk_index", type=SearchFieldDataType.Int32, filterable=True),
            SimpleField(name="metadata_chunking_type", type=SearchFieldDataType.String, filterable=True),
        ]

        vector_search = VectorSearch(
            algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
            profiles=[VectorSearchProfile(name="default", algorithm_configuration_name="hnsw")]
        )

        index = SearchIndex(name=index_name, fields=fields, vector_search=vector_search)

        # Create or update index
        index_client.create_or_update_index(index)
        print(f"Azure index '{index_name}' ready")

        # Create search client
        self.search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=credential)

    # Initialize local ChromaDB
    def _init_chromadb(self):
        self.client = chromadb.PersistentClient(path=str(CHROMADB_DIR), settings=Settings(allow_reset=True))
        self.collection = self.client.get_or_create_collection(name="rag_collection")
        print("Local ChromaDB collection ready")


    # Add documents to the vector DB
    def upsert_documents(self, docs):
        """
        docs: List of dicts with keys 'id', 'content', 'embedding', 'metadata_title',
        'metadata_chunk_index', 'metadata_chunking_type
        '"""
        if self.use_azure:
            actions = [{"id": doc["id"],
                        "content": doc["content"],
                        "embedding": doc["embedding"],
                        "metadata_title": doc.get("metadata_title"),
                        "metadata_chunk_index": doc.get("metadata_chunk_index"),
                        "metadata_chunking_type": doc.get("metadata_chunking_type")} for doc in docs]
            self.search_client.upload_documents(documents=actions)
            print(f"Upserted {len(docs)} documents to Azure AI Search.")
        elif self.use_chromadb:
            batch_size = 100
            total = len(docs)
            for i in range(0, total, batch_size):
                batch = docs[i:i + batch_size]

                ids = [doc["id"] for doc in batch]
                embeddings = [doc["embedding"] for doc in batch]
                metadatas = [{"title": doc.get("metadata_title"),
                              "chunk_index": doc.get("metadata_chunk_index"),
                              "chunking_type": doc.get("metadata_chunking_type")} for doc in batch]
                documents = [doc["content"] for doc in batch]

                self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=documents)
            print(f"Upserted {len(docs)} documents to ChromaDB.")
        else:
            raise RuntimeError("No vector database initialized.")

    # Query the vector DB
    def query_vector(self, vector, top_k=5):
        """
                Query the vector DB and return a uniform list of results:
                Each result is a dict:
                    {
                        "id": str,
                        "content": str,
                        "metadata": dict,
                        "score": float
                    }
                """
        results_list = []

        if self.use_azure:
            results = self.search_client.search(
                search_text="*",
                vectors=[vector],
                top=top_k
            )
            for r in results:
                results_list.append({
                    "id": r["id"],
                    "content": r["content"],
                    "metadata": {
                        "title": r.get("metadata_title"),
                        "chunk_index": r.get("metadata_chunk_index"),
                        "method": r.get("metadata_method")
                    },
                    "score": r["@search.score"] if "@search.score" in r else None
                })

        elif self.use_chromadb:
            query_results = self.collection.query(query_embeddings=[vector], n_results=top_k)
            for i in range(len(query_results["ids"][0])):
                results_list.append({
                    "id": query_results["ids"][0][i],
                    "content": query_results["documents"][0][i],
                    "metadata": query_results["metadatas"][0][i],
                    "score": query_results["distances"][0][i]
                })

        else:
            raise RuntimeError("No vector DB available")

        return results_list


# -------------------------------
# Test/Run
# -------------------------------
if __name__ == "__main__":
    import uuid
    import argparse
    from embeddings import embed_chunks
    from ingestion import ingest_pdfs

    parser = argparse.ArgumentParser(
        description = "Ingest PDFs, generate embeddings, and store in vector DB."
    )
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
    parser.add_argument(
        "--limit",
        type = int,
        default = None,
        help = "Number of documents to process for testing."
    )
    args = parser.parse_args()

    print("\n Starting vector database pipeline...")

    # 1. Init DB
    db = VectorDB()

    # 2. Ingest PDFs
    chunks = ingest_pdfs(data_dir=args.data_dir, chunking_type=args.chunking_type)
    print(f"Extracted {len(chunks)} from PDFs.")

    # 3. Take a random limited number for testing (if specified)
    if args.limit is not None:
        sample_chunks = random.sample(chunks, args.limit)
    else:
        sample_chunks = chunks

    print(f"Processing {len(sample_chunks)} chunks for embedding and upsert.")

    # 4. Embed chunks
    embedded_chunks = embed_chunks(sample_chunks)
    print(f"Generated embeddings for {len(embedded_chunks)} chunks.")

    # 5. Prepare documents for upsert
    docs_to_upsert = [
        {
            "id": str(uuid.uuid4()),
            "content": chunk["content"],
            "embedding": chunk["embedding"],
            "metadata_title": chunk["metadata"]["title"],
            "metadata_chunk_index": chunk["metadata"]["chunk index"],
            "metadata_chunking_type": args.chunking_type
        } for chunk in embedded_chunks
    ]

    # 6. Upsert to vector DB
    db.upsert_documents(docs_to_upsert)
    print("Upsert completed.")

