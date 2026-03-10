"""
Shared test fixtures for VLM Agent Gateway tests.
"""

import pytest

from vlm_agent_gateway.models import Agent, AgentResult


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing."""
    return Agent(
        model="test-model",
        endpoint="http://test.local/v1/chat/completions",
        api_key="test-key",
        provider="openai",
    )


@pytest.fixture
def mock_successful_result():
    """Create a mock successful agent result."""
    return AgentResult(
        agent_id="test-123",
        model="test-model",
        provider="openai",
        content="Test response content",
        raw_response={"choices": [{"message": {"content": "Test response content"}}]},
        latency_ms=100.0,
    )


@pytest.fixture
def mock_failed_result():
    """Create a mock failed agent result."""
    return AgentResult(
        agent_id="test-456",
        model="test-model",
        provider="openai",
        content="",
        raw_response={},
        latency_ms=50.0,
        error="Connection timeout",
    )


@pytest.fixture
def sample_image_b64():
    """1x1 red pixel JPEG as base64."""
    # Minimal valid JPEG
    return (
        "/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRof"
        "Hh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwh"
        "MjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAAR"
        "CAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAn/xAAUEAEAAAAAAAAAAAAAAAAA"
        "AAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oADAMB"
        "AAIRAxEAPwCwAB//2Q=="
    )
