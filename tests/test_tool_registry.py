from langchain_core.tools import tool

from career_agent.contracts import ToolPermission
from career_agent.tools import ToolRegistry


@tool
def read_profile() -> str:
    """Read the current user's profile."""
    return "profile"


@tool
def send_application() -> str:
    """Send a job application."""
    return "sent"


def test_registry_filters_tools_by_permission() -> None:
    registry = ToolRegistry()
    registry.register(read_profile, permission=ToolPermission.READ)
    registry.register(send_application, permission=ToolPermission.EXTERNAL)

    assert [item.name for item in registry.allowed({ToolPermission.READ})] == ["read_profile"]
    assert [item.name for item in registry.allowed({ToolPermission.EXTERNAL})] == ["send_application"]


def test_registry_rejects_duplicate_names() -> None:
    registry = ToolRegistry()
    registry.register(read_profile)

    try:
        registry.register(read_profile)
    except ValueError as exc:
        assert "already registered" in str(exc)
    else:
        raise AssertionError("Expected duplicate registration to fail")
