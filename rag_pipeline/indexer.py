import hashlib
import logging
from typing import Optional
import chromadb
from chromadb.api import ClientAPI
from chromadb.utils import embedding_functions
from config import settings

logger = logging.getLogger(__name__)


def get_chroma_client() -> ClientAPI:
    """
    Returns an embedded client or HTTP service client based on configuration.
    """
    if settings.CHROMA_MODE == "http":
        logger.info(
            f"Connecting to ChromaDB service at {settings.CHROMA_HOST}:{settings.CHROMA_PORT}"
        )
        headers = {}
        if settings.CHROMA_AUTH_TOKEN:
            headers["X-Chroma-Token"] = settings.CHROMA_AUTH_TOKEN
        return chromadb.HttpClient(
            host=settings.CHROMA_HOST, port=settings.CHROMA_PORT, headers=headers
        )

    logger.info(f"Initializing local persistent ChromaDB at {settings.CHROMA_DB_PATH}")
    return chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)


def get_or_create_collection(client: ClientAPI):
    """
    Configures collection with explicitly tuned HNSW index settings.
    """
    emb_fn = embedding_functions.DefaultEmbeddingFunction()

    metadata = {
        "hnsw:space": settings.HNSW_SPACE,
        "hnsw:construction_ef": settings.HNSW_CONSTRUCTION_EF,
        "hnsw:M": settings.HNSW_M,
        "hnsw:search_ef": settings.HNSW_SEARCH_EF,
    }

    return client.get_or_create_collection(
        name=settings.COLLECTION_NAME, embedding_function=emb_fn, metadata=metadata
    )


def batch_upsert_chunks(
    collection, chunks: list[str], source_name: str, category: str = "general"
) -> int:
    """
    Upserts chunks in deterministic, idempotent batches using MD5 content hashes.
    """
    if not chunks:
        return 0

    total_chunks = len(chunks)
    batch_size = settings.INGEST_BATCH_SIZE

    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]

        # Deterministic IDs prevent duplication across pipeline runs
        ids = [
            hashlib.md5(f"{source_name}_{chunk}".encode("utf-8")).hexdigest()
            for chunk in batch
        ]
        metadatas = [
            {
                "source": source_name,
                "category": category,
                "batch_index": i + idx,
            }
            for idx in range(len(batch))
        ]

        collection.upsert(documents=batch, metadatas=metadatas, ids=ids)
        logger.debug(f"Upserted batch {i // batch_size + 1} ({len(batch)} chunks)")

    logger.info(f"Successfully indexed {total_chunks} chunks from '{source_name}'.")
    return total_chunks
