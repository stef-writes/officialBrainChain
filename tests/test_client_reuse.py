import pytest
from unittest.mock import patch, MagicMock
from app.nodes.text_generation import TextGenerationNode
from app.models.config import LLMConfig
from app.models.node_models import NodeConfig, NodeMetadata

@pytest.mark.asyncio
async def test_client_reuse():
    """Test that the OpenAI client is reused across node instances."""
    # Create two different API keys
    api_key1 = "sk-test-key-1"
    api_key2 = "sk-test-key-2"
    
    # Create LLM configs with different API keys
    llm_config1 = LLMConfig(
        model="gpt-4",
        api_key=api_key1,
        temperature=0.7
    )
    
    llm_config2 = LLMConfig(
        model="gpt-4",
        api_key=api_key2,
        temperature=0.7
    )
    
    # Create node configs
    config1 = NodeConfig(
        metadata=NodeMetadata(
            node_id="test-node-1",
            node_type="ai",
            version="1.0.0"
        ),
        llm_config=llm_config1,
        templates=[]
    )
    
    config2 = NodeConfig(
        metadata=NodeMetadata(
            node_id="test-node-2",
            node_type="ai",
            version="1.0.0"
        ),
        llm_config=llm_config2,
        templates=[]
    )
    
    # Create node instances
    node1 = TextGenerationNode(config1)
    node2 = TextGenerationNode(config2)
    node3 = TextGenerationNode(config1)  # Same API key as node1
    
    # Mock the AsyncOpenAI constructor
    with patch('app.nodes.text_generation.AsyncOpenAI') as mock_openai:
        # Configure the mock to return different instances based on API key
        mock_client1 = MagicMock()
        mock_client2 = MagicMock()
        
        # Set up side_effect to return different mocks based on API key
        def side_effect(**kwargs):
            if kwargs.get('api_key') == api_key1:
                return mock_client1
            elif kwargs.get('api_key') == api_key2:
                return mock_client2
            return MagicMock()
        
        mock_openai.side_effect = side_effect
        
        # Access the client property for each node
        client1 = node1.client
        client2 = node2.client
        client3 = node3.client
        
        # Verify that AsyncOpenAI was called only twice (once for each unique API key)
        assert mock_openai.call_count == 2
        
        # Verify that the same client is returned for nodes with the same API key
        assert client1 is client3
        assert client1 is not client2
        
        # Verify that the correct API keys were used
        mock_openai.assert_any_call(api_key=api_key1)
        mock_openai.assert_any_call(api_key=api_key2) 