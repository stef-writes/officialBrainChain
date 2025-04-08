"""
Test suite for ScriptChain using real TextGenerationNode implementations
"""

import pytest
import asyncio
from datetime import datetime
from typing import Dict, Any, List
from app.chains.script_chain import ScriptChain
from app.models.node_models import (
    NodeConfig,
    NodeMetadata,
    NodeExecutionResult,
    UsageMetadata
)
from app.models.config import LLMConfig, MessageTemplate
from app.nodes.text_generation import TextGenerationNode
from app.utils.context import ContextManager
import os

# Get API key from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    pytest.skip("OPENAI_API_KEY not set", allow_module_level=True)

@pytest.fixture
def script_chain():
    """Create a fresh ScriptChain instance for each test"""
    return ScriptChain(max_context_tokens=1000)

@pytest.fixture
def llm_config():
    """Create LLM configuration for nodes"""
    return LLMConfig(
        model="gpt-4",
        api_key=OPENAI_API_KEY,
        temperature=0.7,
        max_tokens=100
    )

@pytest.fixture
def text_generation_node(llm_config):
    """Create a text generation node"""
    return TextGenerationNode.create(llm_config)

@pytest.mark.asyncio
async def test_simple_chain_execution(script_chain, text_generation_node):
    """Test execution of a simple chain with a text generation node"""
    # Add node to chain
    script_chain.add_node(text_generation_node)
    
    # Set initial context with prompt
    script_chain.context.set_context(text_generation_node.node_id, {
        "prompt": "What is 2+2? Answer in one word."
    })
    
    # Execute chain
    result = await script_chain.execute()
    
    # Verify execution
    assert result.success
    assert result.output is not None
    assert "four" in result.output.lower()
    assert result.usage is not None
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0

@pytest.mark.asyncio
async def test_chain_with_multiple_nodes(script_chain, llm_config):
    """Test chain execution with multiple text generation nodes"""
    # Create nodes
    node1 = TextGenerationNode.create(llm_config)
    node2 = TextGenerationNode.create(llm_config)
    node3 = TextGenerationNode.create(llm_config)
    
    # Add nodes to chain
    script_chain.add_node(node1)
    script_chain.add_node(node2)
    script_chain.add_node(node3)
    
    # Add edges to create dependencies
    script_chain.add_edge(node1.node_id, node2.node_id)
    script_chain.add_edge(node2.node_id, node3.node_id)
    
    # Set initial context for all nodes
    script_chain.context.set_context(node1.node_id, {
        "prompt": "What is 2+2? Answer in one word."
    })
    script_chain.context.set_context(node2.node_id, {
        "prompt": "Take the previous answer and add 2 to it. Answer in one word."
    })
    script_chain.context.set_context(node3.node_id, {
        "prompt": "Take the previous answer and multiply it by 2. Answer in one word."
    })
    
    # Execute chain
    result = await script_chain.execute()
    
    # Verify execution
    assert result.success
    assert result.output is not None
    assert result.usage is not None
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0
    assert result.usage.api_calls == 3  # One call per node

@pytest.mark.asyncio
async def test_context_optimization(script_chain, llm_config):
    """Test context optimization with large text"""
    # Create nodes
    node1 = TextGenerationNode.create(llm_config)
    node2 = TextGenerationNode.create(llm_config)
    
    # Add nodes to chain
    script_chain.add_node(node1)
    script_chain.add_node(node2)
    script_chain.add_edge(node1.node_id, node2.node_id)
    
    # Set initial context with large text
    large_text = "test " * 1000  # Create large text
    script_chain.context.set_context(node1.node_id, {
        "prompt": large_text
    })
    script_chain.context.set_context(node2.node_id, {
        "prompt": "Summarize the previous text in one sentence."
    })
    
    # Execute chain
    result = await script_chain.execute()
    
    # Verify context optimization
    assert result.success
    context = script_chain.context.get_context(node2.node_id)
    assert len(str(context)) < 1000  # Verify context was optimized

@pytest.mark.asyncio
async def test_error_handling(script_chain, llm_config):
    """Test error handling with invalid API key"""
    # Create node with invalid API key
    invalid_config = LLMConfig(
        model="gpt-4",  # Using supported model
        api_key="sk-test-" + "1" * 45,  # Create a properly formatted but invalid key
        temperature=0.7,
        max_tokens=100
    )
    error_node = TextGenerationNode.create(invalid_config)
    
    # Add node to chain
    script_chain.add_node(error_node)
    
    # Set initial context
    script_chain.context.set_context(error_node.node_id, {
        "prompt": "What is 2+2?"
    })
    
    # Execute chain
    result = await script_chain.execute()
    
    # Verify error handling
    assert not result.success
    assert result.error is not None
    assert result.metadata.error_type == "AuthenticationError"
    assert "incorrect api key provided" in result.error.lower()
    assert "invalid_api_key" in result.error.lower()

@pytest.mark.asyncio
async def test_retry_mechanism(script_chain, llm_config):
    """Test retry mechanism with rate limiting"""
    from app.utils.retry import AsyncRetry
    
    # Configure retry
    script_chain.retry = AsyncRetry(max_retries=2, delay=0.1)
    
    # Create nodes that will trigger rate limits
    nodes = [TextGenerationNode.create(llm_config) for _ in range(5)]
    
    # Add nodes to chain
    for i, node in enumerate(nodes):
        script_chain.add_node(node)
        if i > 0:
            script_chain.add_edge(nodes[i-1].node_id, node.node_id)
        # Set context for each node
        script_chain.context.set_context(node.node_id, {
            "prompt": f"What is {i}+2? Answer in one word."
        })
    
    # Execute chain
    result = await script_chain.execute()
    
    # Verify retry behavior
    assert result.success
    assert result.usage is not None
    assert result.usage.api_calls > 0

@pytest.mark.asyncio
async def test_workflow_execution(script_chain, llm_config):
    """Test execute_workflow method with real nodes"""
    # Create workflow nodes
    nodes = [TextGenerationNode.create(llm_config) for _ in range(3)]
    node_ids = [node.node_id for node in nodes]
    
    # Set context for each node
    for i, node_id in enumerate(node_ids):
        script_chain.context.set_context(node_id, {
            "prompt": f"What is {i}+2? Answer in one word."
        })
    
    # Execute workflow
    results = await script_chain.execute_workflow(nodes, node_ids)
    
    # Verify workflow execution
    assert isinstance(results, dict)
    assert len(results) == 3
    assert all(node_id in results for node_id in node_ids)
    assert all(isinstance(result, NodeExecutionResult) for result in results.values())
    assert all(result.success for result in results.values())
    assert all(isinstance(result.output, str) for result in results.values()) 