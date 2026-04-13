import os
import httpx
from pathlib import Path
from typing import Tuple, List

import chromadb
from pypdf import PdfReader
from groq import Groq

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "voice_rag_docs"
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


class JinaEmbeddingFunction:
    """Calls Jina AI's embedding API — free tier, no model loaded in memory."""

    def __init__(self, api_key: str):
        self._api_key = api_key

    def __call__(self, input: List[str]) -> List[List[float]]:
        import time
        for attempt in range(3):
            with httpx.Client(timeout=60.0) as client:
                response = client.post(
                    "https://api.jina.ai/v1/embeddings",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"input": input, "model": "jina-embeddings-v2-base-en"},
                )
            if response.status_code == 429:
                time.sleep(2 ** attempt)
                continue
            response.raise_for_status()
            return [item["embedding"] for item in response.json()["data"]]
        response.raise_for_status()


chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
embedding_fn = JinaEmbeddingFunction(api_key=os.environ.get("JINA_API_KEY"))
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_fn,
)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap
    return [c for c in chunks if len(c) > 50]  # filter tiny chunks


def ingest_pdf(file_path: str, display_name: str) -> int:
    """Extract text from PDF, chunk it, and store in ChromaDB."""
    reader = PdfReader(file_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    chunks = _chunk_text(full_text)

    ids = [f"{display_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": display_name, "chunk": i} for i in range(len(chunks))]

    # Upsert so re-uploading same file doesn't duplicate
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)

    return len(chunks)


def query_rag(question: str, n_results: int = 4) -> Tuple[str, List[str]]:
    """Retrieve relevant chunks and generate an answer with Groq."""
    results = collection.query(query_texts=[question], n_results=n_results)

    if not results["documents"] or not results["documents"][0]:
        return "I couldn't find relevant information in the uploaded documents. Please upload a PDF first.", []

    docs = results["documents"][0]
    sources = list({m["source"] for m in results["metadatas"][0]})
    context = "\n\n---\n\n".join(docs)

    system_prompt = (
        "You are a helpful voice assistant that answers questions based strictly on "
        "the provided document context. Keep answers concise and conversational — "
        "you are speaking out loud, so avoid bullet points or markdown. "
        "If the answer isn't in the context, say so clearly."
    )

    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=400,
        temperature=0.3,
    )

    answer = response.choices[0].message.content.strip()
    return answer, sources
