import hashlib
import json
import time
from typing import Any, Dict, Optional
from pydantic import BaseModel


class IdempotencyRecord(BaseModel):
    idempotency_key: str
    result: Any
    created_at: float


class IdempotencyManager:
    def __init__(self, ttl_seconds: int = 300):
        self._store: Dict[str, IdempotencyRecord] = {}
        self.ttl = ttl_seconds

    def generate_key(
        self, tool_name: str, user_id: str, payload: Dict[str, Any]
    ) -> str:
        # Sort keys to ensure stable serialization regardless of key ordering
        canonical_payload = json.dumps(payload, sort_keys=True, default=str)
        raw = f"{tool_name}:{user_id}:{canonical_payload}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> Optional[Any]:
        record = self._store.get(key)
        if not record:
            return None
        if (time.time() - record.created_at) > self.ttl:
            del self._store[key]
            return None
        return record.result

    def save(self, key: str, result: Any) -> None:
        self._store[key] = IdempotencyRecord(
            idempotency_key=key, result=result, created_at=time.time()
        )
