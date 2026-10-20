import os
import chromadb
from chromadb.utils import embedding_functions


def get_collection(db_path: str = "./chroma_db", collection_name: str = "Company_docs"):
    """
    Initializes a persistent ChromaDB client with cosine space distance.
    Cosine distance ensures similarity scores map predictably between 0.0 and 1.0.
    """
    client = chromadb.PersistentClient(path=db_path)
    emb_fn = embedding_functions.DefaultEmbeddingFunction()

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )
    return collection


def index_chunks(collection, chunks: list[str], category: str = "general"):
    """
    Batches and inserts chunks into ChromaDB with deterministic IDs.
    """
    if not chunks:
        return

    ids = [f"doc_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"category": category, "chunk_index": i} for i in range(len(chunks))]

    collection.upsert(documents=chunks, metadatas=metadatas, ids=ids)
