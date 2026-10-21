from typing import Dict, Any, Type
from pydantic import ValidationError
from tools.base import BaseTool
from security.permissions import SecurityContext
from security.idempotency import IdempotencyManager
from security.injection_guard import PromptInjectionDefense


class ToolRegistry:
    def __init__(self):
        self._registry: Dict[str, BaseTool] = {}
        self.idempotency_mgr = IdempotencyManager()

    def register(self, tool: BaseTool) -> None:
        self._registry[tool.name] = tool

    def get_tool(self, name: str) -> BaseTool:
        if name not in self._registry:
            raise ValueError(f"Tool '{name}' is not registered.")
        return self._registry[name]

    def export_all_schemas(self) -> list:
        return [tool.get_json_schema() for tool in self._registry.values()]

    def run_pipeline(
        self, tool_name: str, raw_arguments: Dict[str, Any], context: SecurityContext
    ) -> Dict[str, Any]:
        """
        Executes the guarded tool invocation pipeline.
        """
        tool = self.get_tool(tool_name)

        # Step 1: Permission Scoping Guard
        if not context.has_scope(tool.required_scope):
            return {
                "status": "PERMISSION_DENIED",
                "error": f"SecurityContext for '{context.user_id}' lacks required scope '{tool.required_scope.value}'.",
            }

        # Step 2: Input Validation via JSON / Pydantic Schema
        try:
            validated_args = tool.args_schema.model_validate(raw_arguments)
        except ValidationError as e:
            return {"status": "VALIDATION_ERROR", "error": e.errors()}

        # Step 3: Idempotency Check
        idempotency_key = None
        if tool.is_idempotent:
            idempotency_key = self.idempotency_mgr.generate_key(
                tool.name, context.user_id, validated_args.model_dump()
            )
            cached_result = self.idempotency_mgr.get(idempotency_key)
            if cached_result is not None:
                return {
                    "status": "SUCCESS_CACHED",
                    "result": cached_result,
                    "idempotency_key": idempotency_key,
                }

        # Step 4: Tool Execution
        try:
            raw_output = tool.execute(validated_args)
        except Exception as exc:
            return {"status": "EXECUTION_FAILURE", "error": str(exc)}

        # Step 5: Idempotency Persistence
        if tool.is_idempotent and idempotency_key:
            self.idempotency_mgr.save(idempotency_key, raw_output)

        # Step 6: Indirect Prompt Injection Defense & Data Isolation
        output_str = str(raw_output)
        is_suspicious, reason = PromptInjectionDefense.inspect_untrusted_content(
            output_str
        )

        encapsulated_data = PromptInjectionDefense.sanitize_and_encapsulate(
            raw_output, tool.name
        )

        return {
            "status": "SUCCESS",
            "result": raw_output,
            "llm_safe_payload": encapsulated_data,
            "security_warning": reason if is_suspicious else None,
        }
