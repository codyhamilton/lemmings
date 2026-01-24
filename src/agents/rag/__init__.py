"""RAG (Retrieval-Augmented Generation) system for agent context management.

This module provides semantic code search and context retrieval capabilities
to replace slow shell-based tools with fast vector similarity search.
"""

from .vectorstore import get_vectorstore, embed_documents
from .chunker import CodeChunk, chunk_file
from .indexer import build_index, update_index, get_index_stats
from .retriever import retrieve, retrieve_for_requirement

__all__ = [
    "get_vectorstore",
    "embed_documents",
    "CodeChunk",
    "chunk_file",
    "build_index",
    "update_index",
    "get_index_stats",
    "retrieve",
    "retrieve_for_requirement",
]
