import os
import argparse
from dotenv import load_dotenv
from pathlib import Path
from typing import List, Dict, Any

# -------------------------
# Config
# -------------------------
load_dotenv()               # Load environment variables from .env file

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent   # project root path
DATA_DIR = PROJECT_ROOT/"data"    # path to data folder containing the PDF files

# Import local modules
from embeddings import generate_embedding
from vector_db import VectorDB

# Azure OpenAi config
from openai import OpenAI

AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")

# Validate environment variables
if not AZURE_ENDPOINT or not AZURE_API_KEY or not AZURE_CHAT_DEPLOYMENT:
    raise ValueError("Please set AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, and AZURE_OPENAI_CHAT_DEPLOYMENT "
                     "in your environment variables.")

# Initialize OpenAI client
client = OpenAI(
    api_key=AZURE_API_KEY,
    base_url=AZURE_ENDPOINT.rstrip("/") + "/openai/v1/"
)

# -------------------------
# Query engine class
# -------------------------
class QueryEngine:
    """Query engine for the RAG paper system"""

    def __init__(self):
        self.vector_db = VectorDB()
        if not (self.vector_db.use_azure or self.vector_db.use_chromadb):
            raise RuntimeError("No vector database available. Please check the configuration.")


    def query(self, user_query: str, top_k: int = 5) -> List[Dict[str, any]]:
        query_vector = generate_embedding(user_query)
        results = self.vector_db.query_vector(query_vector, top_k=top_k)
        return results


    def generate_rag_answer(self, user_query:str, top_k:int = 5) -> Dict[str, Any]:
        """
        Generate a RAG answer using the retrieved chunks.
        Returns:
            {
                "answer": str,
                "references": List[dict]  # same as query result
            }
        """

        retrieved_chunks = self.query(user_query, top_k=top_k)

        # Create context from retrieved chunks
        context_text = "\n\n".join([chunk["content"] for chunk in retrieved_chunks])

        prompt = (
            "You are an expert summarizer. Answer the user's question based on the following context:\n\n"
            f"{context_text}\n\n"
            f"User question: {user_query}\n"
            "Provide a concise answer and mention the sources where appropriate."
        )

        response = client.chat.completions.create(
            model=AZURE_CHAT_DEPLOYMENT,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500
        )

        answer_text = response.choices[0].message.content
        return {
            "answer": answer_text,
            "references": retrieved_chunks
        }

def main():
    parser = argparse.ArgumentParser(description="Query the RAG paper system.")
    parser.add_argument(
        "--query",
        type=str,
        default="tell me about Tony Q. S. Quek",
        help="The user query string."
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Number of top relevant chunks to retrieve."
    )

    args = parser.parse_args()

    query_engine = QueryEngine()
    rag_result = query_engine.generate_rag_answer(args.query, top_k=args.top_k)

    print(f"\nRAG Answer for query: '{args.query}'\n")
    print(rag_result["answer"])
    print("\nReferences:")
    for i, ref in enumerate(rag_result["references"]):
        print(f"{i+1}. {ref['metadata'].get('title','Unknown')} (chunk {ref['metadata'].get('chunk_index','?')})")
        print(f"Snippet: {ref['content'][:200]}...\n")

# ------------------------------
# Test/Run
# ------------------------------
if __name__ == "__main__":
    main()


