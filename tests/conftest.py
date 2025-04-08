"""
Pytest configuration and fixtures
"""

import sys
import os
import pytest
from typing import Dict, Any
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
from types import SimpleNamespace
from datetime import datetime

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "ai: mark test as AI-specific"
    )
    config.addinivalue_line(
        "markers", "cost: test that incurs API costs"
    )
    config.addinivalue_line(
        "markers", "security: mark test as security-related"
    )
    config.addinivalue_line(
        "markers", "ethics: mark test as ethics-related"
    )
    config.addinivalue_line(
        "markers", "performance: mark test as performance-related"
    )
    config.addinivalue_line(
        "markers", "complex: mark test as complex scenario"
    )

@pytest.fixture(autouse=True)
def check_cost_usage():
    """Monitor and limit API credit usage per test"""
    start_credits = get_api_credits()
    yield
    used = start_credits - get_api_credits()
    if used > 0.05:  # $0.05 per test
        pytest.fail(f"Test cost ${used:.4f} exceeded limit")

def get_api_credits() -> float:
    """Get current API credits - to be implemented"""
    return 0.0  # Placeholder

@pytest.fixture
def mock_openai_response() -> Dict[str, Any]:
    """Mock OpenAI API response"""
    return {
        "choices": [
            {
                "message": {
                    "content": "Mocked response content",
                    "role": "assistant"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150
        }
    }

@pytest.fixture
def mock_openai(monkeypatch):
    async def mock_create(**kwargs):
        api_key = kwargs.get('api_key', '')
        if api_key == "sk-test-key-for-testing-only":
            raise ValueError("Invalid API key")
            
        messages = kwargs.get('messages', [])
        if not messages:
            raise ValueError("No messages provided")
        
        prompt = messages[-1].get('content', '')
        if not prompt:
            raise ValueError("Empty prompt")
            
        response = SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=f"Mock response to: {prompt}",
                        role="assistant"
                    ),
                    finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=len(prompt.split()),
                completion_tokens=20,
                total_tokens=len(prompt.split()) + 20
            ),
            model="gpt-4"
        )
        return response

    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(side_effect=mock_create)
    monkeypatch.setattr("openai.AsyncClient", MagicMock(return_value=mock_client))
    return mock_client

@pytest.fixture
def context_manager():
    """Fixture for context management testing"""
    from app.utils.context import ContextManager
    return ContextManager() 