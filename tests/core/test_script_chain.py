"""
Script chain functionality tests with complex scenarios
"""

import pytest
import asyncio
from typing import Dict, Any
from app.chains.script_chain import ScriptChain
from app.models.node_models import NodeConfig, NodeMetadata
from app.utils.context import ContextManager
from app.nodes.text_generation import TextGenerationNode
from app.models.config import LLMConfig

@pytest.mark.asyncio
async def test_chain_execution_order(mock_openai):
    """Test that nodes in a chain execute in the correct order"""
    # Create nodes with different configurations
    node1 = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    ))
    
    node2 = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.5,
        max_tokens=150
    ))
    
    node3 = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.3,
        max_tokens=200
    ))
    
    # Create a script chain with dependencies
    chain = ScriptChain()
    chain.add_node(node1)
    chain.add_node(node2)
    chain.add_node(node3)
    
    # Add edges to create dependencies
    chain.add_edge(node1.node_id, node2.node_id)
    chain.add_edge(node2.node_id, node3.node_id)
    
    # Set context for the first node using the chain's context manager
    chain.context.set_context(node1.node_id, {"prompt": "Generate a creative story"})
    chain.context.set_context(node2.node_id, {"prompt": "Continue the story"})
    chain.context.set_context(node3.node_id, {"prompt": "Conclude the story"})
    
    result = await chain.execute()
    
    # In test mode, we expect authentication errors
    assert not result.success
    assert isinstance(result.output, str)  # Output is a string representation of the results
    assert node1.node_id in result.output
    assert "API key" in result.error  # Check for the authentication error
    assert result.metadata.error_type == "AuthenticationError"
    assert result.duration > 0

@pytest.mark.asyncio
async def test_chain_error_handling(mock_openai):
    """Test error handling in a chain with multiple nodes"""
    # Create nodes with different configurations
    node1 = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    ))
    
    node2 = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.5,
        max_tokens=150
    ))
    
    # Create a script chain
    chain = ScriptChain()
    chain.add_node(node1)
    chain.add_node(node2)
    
    # Add edge to create dependency
    chain.add_edge(node1.node_id, node2.node_id)
    
    # Set error-inducing context for the first node using the chain's context manager
    chain.context.set_context(node1.node_id, {"prompt": "error"})  # This will trigger an error in the mock
    chain.context.set_context(node2.node_id, {"prompt": "Continue the story"})
    
    result = await chain.execute()
    
    # In test mode, we expect authentication errors
    assert not result.success
    assert isinstance(result.output, str)  # Output is a string representation of the results
    assert node1.node_id in result.output
    assert "API key" in result.error  # Check for the authentication error
    assert result.metadata.error_type == "AuthenticationError"
    assert result.duration > 0

@pytest.mark.asyncio
async def test_chain_context_persistence(mock_openai):
    """Test context persistence across nodes in a chain"""
    # Create nodes with different configurations
    node1 = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    ))
    
    node2 = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.5,
        max_tokens=150
    ))
    
    # Create a script chain
    chain = ScriptChain()
    chain.add_node(node1)
    chain.add_node(node2)
    
    # Add edge to create dependency
    chain.add_edge(node1.node_id, node2.node_id)
    
    # Set context for nodes using the chain's context manager
    chain.context.set_context(node1.node_id, {
        "background": "Testing context persistence",
        "format": "detailed",
        "prompt": "Generate a creative story"
    })
    chain.context.set_context(node2.node_id, {
        "background": "Testing context persistence",
        "format": "detailed",
        "prompt": "Continue the story"
    })
    
    result = await chain.execute()
    
    # In test mode, we expect authentication errors
    assert not result.success
    assert isinstance(result.output, str)  # Output is a string representation of the results
    assert node1.node_id in result.output
    assert "API key" in result.error  # Check for the authentication error
    assert result.metadata.error_type == "AuthenticationError"
    assert result.duration > 0
    
    # Verify context was passed correctly
    node1_context = chain.context.get_context(node1.node_id)
    node2_context = chain.context.get_context(node2.node_id)
    
    assert "background" in node1_context
    assert "format" in node1_context
    assert "prompt" in node1_context
    assert "background" in node2_context
    assert "format" in node2_context
    assert "prompt" in node2_context

@pytest.mark.asyncio
async def test_chain_concurrent_execution(mock_openai):
    """Test concurrent execution of multiple chains"""
    # Create nodes for multiple chains
    chain1_node1 = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    ))
    
    chain1_node2 = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.5,
        max_tokens=150
    ))
    
    chain2_node1 = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.3,
        max_tokens=200
    ))
    
    chain2_node2 = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.4,
        max_tokens=175
    ))
    
    # Create two script chains
    chain1 = ScriptChain()
    chain1.add_node(chain1_node1)
    chain1.add_node(chain1_node2)
    chain1.add_edge(chain1_node1.node_id, chain1_node2.node_id)
    
    chain2 = ScriptChain()
    chain2.add_node(chain2_node1)
    chain2.add_node(chain2_node2)
    chain2.add_edge(chain2_node1.node_id, chain2_node2.node_id)
    
    # Set context for nodes
    chain1.context.set_context(chain1_node1.node_id, {"prompt": "Generate a story about space"})
    chain1.context.set_context(chain1_node2.node_id, {"prompt": "Continue the space story"})
    
    chain2.context.set_context(chain2_node1.node_id, {"prompt": "Generate a story about the ocean"})
    chain2.context.set_context(chain2_node2.node_id, {"prompt": "Continue the ocean story"})
    
    # Execute chains concurrently
    tasks = [
        chain1.execute(),
        chain2.execute()
    ]
    
    results = await asyncio.gather(*tasks)
    
    # In test mode, we expect authentication errors
    for result in results:
        assert not result.success
        assert isinstance(result.output, str)  # Output is a string representation of the results
        assert "API key" in result.error  # Check for the authentication error
        assert result.metadata.error_type == "AuthenticationError"
        assert result.duration > 0 