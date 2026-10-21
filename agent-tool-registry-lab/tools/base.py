from abc import ABC, abstractmethod
from typing import Type, Dict, Any
from pydantic import BaseModel
from security.permissions import Scope


class BaseTool(ABC):
    name: str
    description: str
    required_scope: Scope
    is_idempotent: bool = False
    args_schema: Type[BaseModel]

    def get_json_schema(self) -> Dict[str, Any]:
        """Returns standard JSON Schema compatible with OpenAI/Anthropic tool schemas."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.args_schema.model_json_schema(),
        }

    @abstractmethod
    def execute(self, validated_args: BaseModel) -> Any:
        pass
