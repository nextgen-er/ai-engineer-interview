import pytest
from registry import ToolRegistry
from tools.implementations import TransferFundsTool, FetchExternalDocTool
from security.permissions import SecurityContext, Scope


@pytest.fixture
def test_setup():
    registry = ToolRegistry()
    registry.register(TransferFundsTool())
    registry.register(FetchExternalDocTool())

    context = SecurityContext(
        user_id="tester",
        session_id="sess_test",
        granted_scopes={Scope.READ, Scope.FINANCIAL},
    )
    return registry, context


def test_permission_enforcement():
    registry = ToolRegistry()
    registry.register(TransferFundsTool())
    read_only_ctx = SecurityContext(
        user_id="u1", session_id="s1", granted_scopes={Scope.READ}
    )

    result = registry.run_pipeline(
        "transfer_funds", {"recipient_account": "ACC-1234", "amount": 10}, read_only_ctx
    )
    assert result["status"] == "PERMISSION_DENIED"


def test_validation_failure(test_setup):
    registry, ctx = test_setup
    result = registry.run_pipeline(
        "transfer_funds", {"recipient_account": "ACC-1234", "amount": 999999}, ctx
    )
    assert result["status"] == "VALIDATION_ERROR"


def test_idempotency(test_setup):
    registry, ctx = test_setup
    args = {"recipient_account": "ACC-1234", "amount": 100.0}

    first = registry.run_pipeline("transfer_funds", args, ctx)
    second = registry.run_pipeline("transfer_funds", args, ctx)

    assert first["status"] == "SUCCESS"
    assert second["status"] == "SUCCESS_CACHED"
    assert first["result"] == second["result"]


def test_injection_guard(test_setup):
    registry, ctx = test_setup
    result = registry.run_pipeline(
        "fetch_external_doc", {"document_id": "malicious_doc"}, ctx
    )

    assert result["status"] == "SUCCESS"
    assert result["security_warning"] is not None
    assert "<untrusted_tool_data" in result["llm_safe_payload"]
