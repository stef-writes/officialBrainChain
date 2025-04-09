"""
Tests for the enhanced ScriptChain implementation.
"""

import pytest
import asyncio
from typing import Dict, Any
from app.chains.script_chain import ScriptChain
from app.models.node_models import NodeConfig, NodeExecutionResult
from app.utils.callbacks import LoggingCallback, MetricsCallback, DebugCallback

@pytest.mark.asyncio
async def test_script_chain_initialization(script_chain: ScriptChain):
    """Test ScriptChain initialization with different configurations."""
    assert script_chain.concurrency_level == 2
    assert script_chain.retry_policy['max_retries'] == 2
    assert script_chain.retry_policy['delay'] == 0.1
    assert script_chain.retry_policy['backoff'] == 1.5

@pytest.mark.asyncio
async def test_add_node(script_chain: ScriptChain, test_nodes: Dict[str, NodeConfig]):
    """Test adding nodes to the chain."""
    # Add nodes
    for node in test_nodes.values():
        script_chain.add_node(node)
    
    # Verify nodes were added
    assert len(script_chain.nodes) == 3
    assert "node1" in script_chain.nodes
    assert "node2" in script_chain.nodes
    assert "node3" in script_chain.nodes

@pytest.mark.asyncio
async def test_validate_workflow(script_chain: ScriptChain, test_nodes: Dict[str, NodeConfig]):
    """Test workflow validation."""
    # Add valid nodes
    for node in test_nodes.values():
        script_chain.add_node(node)
    
    # Validate should pass
    assert script_chain.validate_workflow() is True
    
    # Add node with cyclic dependency
    cyclic_node = NodeConfig(
        id="cyclic",
        type="llm",
        model="gpt-4",
        prompt="Test",
        level=0,
        dependencies=["node3"]  # Creates a cycle
    )
    script_chain.add_node(cyclic_node)
    
    # Validate should fail
    assert script_chain.validate_workflow() is False

@pytest.mark.asyncio
async def test_execution_levels(script_chain: ScriptChain, test_nodes: Dict[str, NodeConfig]):
    """Test execution level calculation."""
    # Add nodes
    for node in test_nodes.values():
        script_chain.add_node(node)
    
    # Calculate levels
    levels = script_chain._calculate_execution_levels()
    
    # Verify levels
    assert len(levels) == 3
    assert "node1" in levels[0]
    assert "node2" in levels[1]
    assert "node3" in levels[2]

@pytest.mark.asyncio
async def test_parallel_execution(script_chain: ScriptChain, test_nodes: Dict[str, NodeConfig]):
    """Test parallel execution of nodes at the same level."""
    # Add nodes at the same level
    node1 = test_nodes["node1"]
    node2 = NodeConfig(
        id="parallel_node",
        type="llm",
        model="gpt-4",
        prompt="Test",
        level=0
    )
    script_chain.add_node(node1)
    script_chain.add_node(node2)
    
    # Execute
    result = await script_chain.execute()
    
    # Verify both nodes executed
    assert result.success
    assert "node1" in result.node_results
    assert "parallel_node" in result.node_results

@pytest.mark.asyncio
async def test_error_handling(script_chain: ScriptChain):
    """Test error handling during execution."""
    # Add node that will fail
    failing_node = NodeConfig(
        id="failing_node",
        type="llm",
        model="gpt-4",
        prompt="Test",
        level=0
    )
    script_chain.add_node(failing_node)
    
    # Mock node execution to fail
    async def mock_execute(*args, **kwargs):
        raise Exception("Test error")
    
    script_chain.execute_node = mock_execute
    
    # Execute
    result = await script_chain.execute()
    
    # Verify error handling
    assert not result.success
    assert "failing_node" in result.node_results
    assert result.node_results["failing_node"].error is not None

@pytest.mark.asyncio
async def test_callback_integration(script_chain: ScriptChain, test_nodes: Dict[str, NodeConfig], callbacks: Dict[str, Any]):
    """Test callback integration during execution."""
    # Add callbacks
    for callback in callbacks.values():
        script_chain.add_callback(callback)
    
    # Add nodes
    for node in test_nodes.values():
        script_chain.add_node(node)
    
    # Execute
    result = await script_chain.execute()
    
    # Verify callbacks were triggered
    assert result.success
    metrics = callbacks["metrics"].get_metrics()
    assert "chains" in metrics
    assert len(metrics["chains"]) > 0
    
    debug_events = callbacks["debug"].get_events()
    assert len(debug_events) > 0

@pytest.mark.asyncio
async def test_retry_mechanism(script_chain: ScriptChain):
    """Test retry mechanism for failed nodes."""
    # Add node that will fail twice then succeed
    retry_node = NodeConfig(
        id="retry_node",
        type="llm",
        model="gpt-4",
        prompt="Test",
        level=0
    )
    script_chain.add_node(retry_node)
    
    # Track attempts
    attempts = 0
    
    # Mock node execution to fail twice then succeed
    async def mock_execute(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise Exception("Test error")
        return NodeExecutionResult(success=True)
    
    script_chain.execute_node = mock_execute
    
    # Execute
    result = await script_chain.execute()
    
    # Verify retry behavior
    assert result.success
    assert attempts == 3
    assert "retry_node" in result.node_results
    assert result.node_results["retry_node"].success 