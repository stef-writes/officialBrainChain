"""
Tests for token usage tracking
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from app.nodes.text_generation import TextGenerationNode
from app.models.config import LLMConfig
from app.models.node_models import NodeConfig, NodeMetadata, UsageMetadata
from app.utils.context import ContextManager

async def async_return(result):
    """Helper function to create an async return value."""
    return result

@pytest.mark.asyncio
async def test_token_usage_tracking():
    """Test that token usage is properly tracked via the decorator."""
    # Create a mock context manager
    context_manager = ContextManager()
    
    # Create a node with the context manager
    llm_config = LLMConfig(
        model="gpt-4",
        api_key="sk-test-key",
        temperature=0.7
    )
    
    config = NodeConfig(
        metadata=NodeMetadata(
            node_id="test-node",
            node_type="ai",
            version="1.0.0"
        ),
        llm_config=llm_config,
        templates=[]
    )
    
    node = TextGenerationNode(config)
    node.context = context_manager  # Attach the context manager to the node
    
    # Mock the OpenAI client response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Test response"
    mock_response.usage = MagicMock()
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 20
    mock_response.usage.total_tokens = 30
    
    # Create a mock client
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = async_return(mock_response)
    
    # Mock the client property
    with patch('app.nodes.text_generation.TextGenerationNode.client', new_callable=PropertyMock) as mock_client_prop:
        mock_client_prop.return_value = mock_client
        
        # Execute the node
        result = await node.execute({"prompt": "Test prompt"})
        
        # Verify the result
        assert result.success
        assert result.output == "Test response"
        assert result.usage is not None
        assert result.usage.prompt_tokens == 10
        assert result.usage.completion_tokens == 20
        assert result.usage.total_tokens == 30
        assert result.usage.api_calls == 1
        assert result.usage.model == "gpt-4"
        assert result.usage.node_id == "test-node"
        
        # Verify that the usage was tracked in the context manager
        usage_stats = context_manager.get_usage_stats()
        assert "test-node" in usage_stats
        assert usage_stats["test-node"].prompt_tokens == 10
        assert usage_stats["test-node"].completion_tokens == 20
        assert usage_stats["test-node"].total_tokens == 30
        
        # Verify that the token count was updated
        assert context_manager._token_counts["test-node"] == 30

@pytest.mark.asyncio
async def test_token_usage_aggregation():
    """Test that token usage is properly aggregated across multiple nodes."""
    # Create a context manager
    context_manager = ContextManager()
    
    # Create multiple nodes with the same context manager
    nodes = []
    for i in range(3):
        llm_config = LLMConfig(
            model="gpt-4",
            api_key="sk-test-key",
            temperature=0.7
        )
        
        config = NodeConfig(
            metadata=NodeMetadata(
                node_id=f"test-node-{i}",
                node_type="ai",
                version="1.0.0"
            ),
            llm_config=llm_config,
            templates=[]
        )
        
        node = TextGenerationNode(config)
        node.context = context_manager  # Attach the context manager to the node
        nodes.append(node)
    
    # Mock the OpenAI client response for each node
    for i, node in enumerate(nodes):
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = f"Test response {i}"
        mock_response.usage = MagicMock()
        mock_response.usage.prompt_tokens = 10 * (i + 1)
        mock_response.usage.completion_tokens = 20 * (i + 1)
        mock_response.usage.total_tokens = 30 * (i + 1)
        
        # Create a mock client
        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = async_return(mock_response)
        
        # Mock the client property
        with patch('app.nodes.text_generation.TextGenerationNode.client', new_callable=PropertyMock) as mock_client_prop:
            mock_client_prop.return_value = mock_client
            
            # Execute the node
            result = await node.execute({"prompt": f"Test prompt {i}"})
            
            # Verify the result
            assert result.success
            assert result.output == f"Test response {i}"
            assert result.usage is not None
            assert result.usage.prompt_tokens == 10 * (i + 1)
            assert result.usage.completion_tokens == 20 * (i + 1)
            assert result.usage.total_tokens == 30 * (i + 1)
            assert result.usage.api_calls == 1
            assert result.usage.model == "gpt-4"
            assert result.usage.node_id == f"test-node-{i}"
    
    # Verify that the usage was tracked for all nodes
    usage_stats = context_manager.get_usage_stats()
    assert len(usage_stats) == 3
    
    # Verify the total token counts
    total_prompt_tokens = sum(usage.prompt_tokens for usage in usage_stats.values())
    total_completion_tokens = sum(usage.completion_tokens for usage in usage_stats.values())
    total_tokens = sum(usage.total_tokens for usage in usage_stats.values())
    
    assert total_prompt_tokens == 60  # 10 + 20 + 30
    assert total_completion_tokens == 120  # 20 + 40 + 60
    assert total_tokens == 180  # 30 + 60 + 90 