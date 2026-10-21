from enum import Enum
from typing import Set
from pydantic import BaseModel, Field


class Scope(str, Enum):
    READ = "tools:read"
    WRITE = "tools:write"
    FINANCIAL = "tools:financial"
    ADMIN = "tools:admin"


class SecurityContext(BaseModel):
    user_id: str
    session_id: str
    granted_scopes: Set[Scope] = Field(default_factory=set)

    def has_scope(self, required_scope: Scope) -> bool:
        return required_scope in self.granted_scopes
