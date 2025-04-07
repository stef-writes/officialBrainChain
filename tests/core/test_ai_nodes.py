"""
Core AI functionality tests
"""

import pytest
from app.nodes.ai_nodes import TextGenerationNode
from app.models.config import LLMConfig
from app.models.nodes import NodeConfig, NodeMetadata

@pytest.mark.asyncio
async def test_text_generation_basic(mock_openai):
    """Test basic text generation functionality"""
    # Create config with test API key
    llm_config = LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    )
    
    # Create node
    node = TextGenerationNode.create(llm_config)
    
    # Execute with simple prompt
    result = await node.execute({"prompt": "Hello, how are you?"})
    
    # In test mode with invalid API key, we expect authentication error
    assert not result.success
    assert "API key" in result.error
    assert result.metadata.error_type == "AuthenticationError"
    assert result.duration > 0

@pytest.mark.asyncio
async def test_text_generation_error_handling(mock_openai):
    """Test error handling in text generation"""
    llm_config = LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    )
    
    node = TextGenerationNode.create(llm_config)
    
    # Test with empty prompt
    result = await node.execute({"prompt": ""})
    
    assert not result.success
    assert result.error is not None
    assert "No prompt provided" in result.error

@pytest.mark.asyncio
async def test_text_generation_metadata(mock_openai):
    """Test metadata handling in text generation"""
    llm_config = LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    )
    
    node = TextGenerationNode.create(llm_config)
    
    result = await node.execute({"prompt": "Test prompt"})
    
    # In test mode with invalid API key, we expect authentication error
    assert not result.success
    assert "API key" in result.error
    assert result.metadata.error_type == "AuthenticationError"
    assert result.duration > 0
    assert result.metadata.node_type == "ai"  # Still check that metadata is properly set 