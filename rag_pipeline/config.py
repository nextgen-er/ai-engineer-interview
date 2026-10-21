from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ChromaDB Service Configuration
    CHROMA_MODE: Literal["embedded", "http"] = "embedded"
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_AUTH_TOKEN: str = ""
    CHROMA_DB_PATH: str = "./chroma_db"
    COLLECTION_NAME: str = "production_docs"

    # HNSW Index Tuning
    HNSW_SPACE: Literal["cosine", "l2", "ip"] = "cosine"
    HNSW_CONSTRUCTION_EF: int = 128
    HNSW_M: int = 16
    HNSW_SEARCH_EF: int = 64

    # Chunking Configuration
    CHUNK_SIZE_TOKENS: int = 120
    CHUNK_OVERLAP_TOKENS: int = 30

    # Retrieval & Guardrail
    VECTOR_TOP_K: int = 5
    RERANKER_MODEL_NAME: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    CONFIDENCE_THRESHOLD: float = 0.10

    # Ingestion Batching
    INGEST_BATCH_SIZE: int = 200

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
