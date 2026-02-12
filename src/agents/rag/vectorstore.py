"""Vector store implementation using ChromaDB with local embeddings."""

import os
from pathlib import Path
from typing import Optional

import chromadb
from ..logging_config import get_logger

logger = get_logger(__name__)
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer


# Global embedding model instance (loaded once)
_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    """Get or initialize the embedding model.
    
    Uses sentence-transformers/all-MiniLM-L6-v2:
    - Fast (runs on CPU)
    - 384-dimensional embeddings
    - Good for code and technical text
    
    Returns:
        Initialized SentenceTransformer model
    """
    global _embedding_model
    
    if _embedding_model is None:
        logger.info("Loading embedding model (first time only)...")
        _embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    return _embedding_model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed a list of text documents.
    
    Args:
        texts: List of text strings to embed
    
    Returns:
        List of embedding vectors (each is a list of floats)
    """
    model = get_embedding_model()
    embeddings = model.encode(texts, show_progress_bar=len(texts) > 50)
    return embeddings.tolist()


def get_vectorstore(persist_dir: str | Path) -> chromadb.Collection:
    """Get or create a ChromaDB collection for code chunks.
    
    Args:
        persist_dir: Directory to persist the ChromaDB database
    
    Returns:
        ChromaDB Collection instance
    """
    persist_dir = Path(persist_dir)
    persist_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize ChromaDB client with persistence
    client = chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        )
    )
    
    # Get or create collection
    # Using cosine similarity for semantic search
    collection = client.get_or_create_collection(
        name="code_chunks",
        metadata={"hnsw:space": "cosine"},
    )
    
    return collection


def add_chunks_to_store(
    collection: chromadb.Collection,
    chunks: list[dict],
) -> None:
    """Add code chunks to the vector store.
    
    Args:
        collection: ChromaDB collection
        chunks: List of chunk dictionaries with keys:
            - id: Unique identifier
            - content: Text content to embed
            - metadata: Dict with path, chunk_type, symbols, start_line, end_line
    """
    if not chunks:
        return
    
    # Extract components
    ids = [c["id"] for c in chunks]
    documents = [c["content"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    
    # Generate embeddings
    embeddings = embed_documents(documents)
    
    # Add to collection (upsert to handle updates)
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )


def query_store(
    collection: chromadb.Collection,
    query_text: str,
    n_results: int = 10,
    where: dict | None = None,
    where_document: dict | None = None,
) -> dict:
    """Query the vector store for similar chunks.
    
    Args:
        collection: ChromaDB collection
        query_text: Query text to find similar chunks
        n_results: Number of results to return
        where: Metadata filter (e.g., {"chunk_type": "function"})
        where_document: Document content filter
    
    Returns:
        Query results with ids, documents, metadatas, distances
    """
    # Embed query
    query_embedding = embed_documents([query_text])[0]
    
    # Query collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        where_document=where_document,
    )
    
    return results


def delete_chunks_by_path(collection: chromadb.Collection, file_path: str) -> None:
    """Delete all chunks for a specific file path.
    
    Args:
        collection: ChromaDB collection
        file_path: Path to filter by
    """
    try:
        collection.delete(where={"path": file_path})
    except Exception:
        # Collection might be empty or path might not exist
        pass
