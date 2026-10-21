import json
from registry import ToolRegistry
from tools.implementations import TransferFundsTool, FetchExternalDocTool
from security.permissions import SecurityContext, Scope


def print_result(scenario: str, result: dict):
    print(f"\n{'=' * 20} {scenario} {'=' * 20}")
    print(json.dumps(result, indent=2, default=str))


def main():
    registry = ToolRegistry()
    registry.register(TransferFundsTool())
    registry.register(FetchExternalDocTool())

    # User 1: Read-only permissions
    guest_ctx = SecurityContext(
        user_id="user_guest_101", session_id="sess_1", granted_scopes={Scope.READ}
    )

    # User 2: Financial permissions
    finance_ctx = SecurityContext(
        user_id="user_finance_202",
        session_id="sess_2",
        granted_scopes={Scope.READ, Scope.FINANCIAL},
    )

    # Scenario 1: Permission Violation
    res1 = registry.run_pipeline(
        tool_name="transfer_funds",
        raw_arguments={"recipient_account": "ACC-1234", "amount": 50.0},
        context=guest_ctx,
    )
    print_result("1. PERMISSION DENIED (Guest attempting financial transfer)", res1)

    # Scenario 2: Schema Validation Error
    res2 = registry.run_pipeline(
        tool_name="transfer_funds",
        raw_arguments={"recipient_account": "INVALID_FORMAT", "amount": -100.0},
        context=finance_ctx,
    )
    print_result("2. SCHEMA VALIDATION FAILURE (Negative amount + bad regex)", res2)

    # Scenario 3: Idempotency Success & Duplicate Call
    payload = {"recipient_account": "ACC-5678", "amount": 250.0, "currency": "USD"}
    res3_first = registry.run_pipeline("transfer_funds", payload, finance_ctx)
    print_result("3a. IDEMPOTENT EXECUTION (First Run)", res3_first)

    res3_replay = registry.run_pipeline("transfer_funds", payload, finance_ctx)
    print_result("3b. IDEMPOTENT EXECUTION (Replay Cached Run)", res3_replay)

    # Scenario 4: Indirect Prompt Injection Attack Detected
    res4 = registry.run_pipeline(
        tool_name="fetch_external_doc",
        raw_arguments={"document_id": "malicious_doc"},
        context=guest_ctx,
    )
    print_result("4. INDIRECT INJECTION DETECTED & ENCAPSULATED", res4)


if __name__ == "__main__":
    main()
