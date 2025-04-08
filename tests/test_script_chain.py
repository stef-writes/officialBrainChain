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
from app.utils.debug_callback import DebugCallback
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
    assert "authentication failed" in result.error.lower()
    assert "api key" in result.error.lower()

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

@pytest.mark.asyncio
async def test_script_chain_callbacks():
    """Test that callbacks are properly invoked during chain execution."""
    # Create a debug callback
    debug_callback = DebugCallback()
    
    # Create a simple chain with two nodes
    chain = ScriptChain(callbacks=[debug_callback])
    
    # Add nodes
    node1 = TextGenerationNode(
        node_id="node1",
        config=NodeConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            messages=[
                MessageTemplate(
                    role="user",
                    content="Generate a short poem about {topic}"
                )
            ]
        )
    )
    
    node2 = TextGenerationNode(
        node_id="node2",
        config=NodeConfig(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=100,
            messages=[
                MessageTemplate(
                    role="user",
                    content="Summarize this poem: {output}"
                )
            ]
        )
    )
    
    chain.add_node(node1)
    chain.add_node(node2)
    chain.add_edge("node1", "node2")
    
    # Execute the chain
    result = await chain.execute()
    
    # Verify the result
    assert result.success
    assert result.output is not None
    assert result.error is None
    
    # Verify that all callback events were logged
    assert len(debug_callback.events) >= 6  # At least chain start, 2 node starts, 2 node completes, chain end
    
    # Verify chain start event
    chain_start = next(e for e in debug_callback.events if e["event"] == "chain_start")
    assert chain_start["chain_id"].startswith("chain_")
    assert chain_start["config"]["node_count"] == 2
    assert "node1" in chain_start["config"]["execution_order"]
    assert "node2" in chain_start["config"]["execution_order"]
    
    # Verify node start events
    node1_start = next(e for e in debug_callback.events if e["event"] == "node_start" and e["node_id"] == "node1")
    assert node1_start["config"]["model"] == "gpt-3.5-turbo"
    
    node2_start = next(e for e in debug_callback.events if e["event"] == "node_start" and e["node_id"] == "node2")
    assert node2_start["config"]["model"] == "gpt-3.5-turbo"
    
    # Verify node complete events
    node1_complete = next(e for e in debug_callback.events if e["event"] == "node_complete" and e["node_id"] == "node1")
    assert node1_complete["result"].success
    assert node1_complete["result"].output is not None
    
    node2_complete = next(e for e in debug_callback.events if e["event"] == "node_complete" and e["node_id"] == "node2")
    assert node2_complete["result"].success
    assert node2_complete["result"].output is not None
    
    # Verify context update events
    context_updates = [e for e in debug_callback.events if e["event"] == "context_update"]
    assert len(context_updates) >= 2  # At least one update per node
    
    # Verify chain end event
    chain_end = next(e for e in debug_callback.events if e["event"] == "chain_end")
    assert chain_end["result"]["success"]
    assert chain_end["result"]["output"] is not None
    assert chain_end["result"]["error"] is None
    assert chain_end["result"]["usage"] is not None

@pytest.mark.asyncio
async def test_callback_events(script_chain, llm_config):
    """Test that all callback events are properly triggered during chain execution."""
    # Create a debug callback
    debug_callback = DebugCallback()
    script_chain.callbacks = [debug_callback]
    
    # Create nodes
    node1 = TextGenerationNode.create(llm_config)
    node2 = TextGenerationNode.create(llm_config)
    
    # Add nodes to chain
    script_chain.add_node(node1)
    script_chain.add_node(node2)
    script_chain.add_edge(node1.node_id, node2.node_id)
    
    # Set initial context
    script_chain.context.set_context(node1.node_id, {
        "prompt": "Generate a short poem about nature."
    })
    script_chain.context.set_context(node2.node_id, {
        "prompt": "Summarize this poem: {output}"
    })
    
    # Execute chain
    result = await script_chain.execute()
    
    # Verify chain execution was successful
    assert result.success
    assert result.output is not None
    
    # Verify that all callback events were logged
    events = debug_callback.events
    assert len(events) >= 6  # At least chain start, 2 node starts, 2 node completes, chain end
    
    # Verify chain start event
    chain_start = next(e for e in events if e["event"] == "chain_start")
    assert chain_start["chain_id"].startswith("chain_")
    assert chain_start["config"]["node_count"] == 2
    assert "execution_order" in chain_start["config"]
    
    # Verify node start events
    node1_start = next(e for e in events if e["event"] == "node_start" and e["node_id"] == node1.node_id)
    assert node1_start["config"].llm_config.model == llm_config.model
    
    node2_start = next(e for e in events if e["event"] == "node_start" and e["node_id"] == node2.node_id)
    assert node2_start["config"].llm_config.model == llm_config.model
    
    # Verify node complete events
    node1_complete = next(e for e in events if e["event"] == "node_complete" and e["node_id"] == node1.node_id)
    assert node1_complete["result"].success
    assert node1_complete["result"].output is not None
    assert node1_complete["result"].usage is not None
    
    node2_complete = next(e for e in events if e["event"] == "node_complete" and e["node_id"] == node2.node_id)
    assert node2_complete["result"].success
    assert node2_complete["result"].output is not None
    assert node2_complete["result"].usage is not None
    
    # Verify context update events
    context_updates = [e for e in events if e["event"] == "context_update"]
    assert len(context_updates) >= 2  # At least one update per node
    
    # Verify chain end event
    chain_end = next(e for e in events if e["event"] == "chain_end")
    assert chain_end["result"]["success"]
    assert chain_end["result"]["output"] is not None
    assert chain_end["result"]["error"] is None
    assert chain_end["result"]["usage"] is not None 