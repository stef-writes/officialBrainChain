"""
Tests for the enhanced ScriptChain implementation.
"""

import pytest
import asyncio
from typing import Dict, Any
from app.chains.script_chain import ScriptChain, ExecutionLevel
from app.models.node_models import NodeConfig, NodeExecutionResult, NodeMetadata, UsageMetadata
from app.utils.callbacks import LoggingCallback, MetricsCallback
from app.models.config import LLMConfig
from datetime import datetime

@pytest.mark.asyncio
async def test_script_chain_initialization(script_chain: ScriptChain):
    """Test ScriptChain initialization."""
    assert script_chain.max_context_tokens == 1000
    assert script_chain.concurrency_level == 2
    assert script_chain.nodes == {}
    assert script_chain.dependencies == {}
    assert script_chain.execution_levels == {}
    assert script_chain.llm_config.model == "gpt-4"
    assert script_chain.llm_config.api_key == "test-key"
    assert script_chain.llm_config.temperature == 0.7
    assert script_chain.llm_config.max_tokens == 500
    assert script_chain.llm_config.max_context_tokens == 1000

@pytest.mark.asyncio
async def test_add_node(script_chain: ScriptChain):
    """Test adding nodes to the chain."""
    # Create test nodes
    nodes = {
        "node1": NodeConfig(
            id="node1",
            type="llm",
            model="gpt-4",
            prompt="Test prompt 1",
            level=0
        ),
        "node2": NodeConfig(
            id="node2",
            type="llm",
            model="gpt-4",
            prompt="Test prompt 2",
            level=1,
            dependencies=["node1"]
        ),
        "node3": NodeConfig(
            id="node3",
            type="llm",
            model="gpt-4",
            prompt="Test prompt 3",
            level=2,
            dependencies=["node2"]
        )
    }
    
    # Add nodes
    for node in nodes.values():
        script_chain.add_node(node)
    
    # Verify nodes were added
    assert len(script_chain.nodes) == 3
    assert "node1" in script_chain.nodes
    assert "node2" in script_chain.nodes
    assert "node3" in script_chain.nodes

@pytest.mark.asyncio
async def test_validate_workflow(script_chain: ScriptChain):
    """Test workflow validation."""
    # Add valid nodes
    nodes = {
        "node1": NodeConfig(
            id="node1",
            type="llm",
            model="gpt-4",
            prompt="Test prompt 1",
            level=0
        ),
        "node2": NodeConfig(
            id="node2",
            type="llm",
            model="gpt-4",
            prompt="Test prompt 2",
            level=1,
            dependencies=["node1"]
        )
    }
    
    for node in nodes.values():
        script_chain.add_node(node)
    
    # Validate should pass
    assert script_chain.validate_workflow() is True
    
    # Add orphan node
    orphan_node = NodeConfig(
        id="orphan",
        type="llm",
        model="gpt-4",
        prompt="Test",
        level=0
    )
    script_chain.add_node(orphan_node)
    
    # Validate should still pass but log warning
    assert script_chain.validate_workflow() is True

@pytest.mark.asyncio
async def test_execution_levels(script_chain: ScriptChain):
    """Test execution level calculation."""
    # Add nodes
    nodes = {
        "node1": NodeConfig(
            id="node1",
            type="llm",
            model="gpt-4",
            prompt="Test prompt 1",
            level=0
        ),
        "node2": NodeConfig(
            id="node2",
            type="llm",
            model="gpt-4",
            prompt="Test prompt 2",
            level=1,
            dependencies=["node1"]
        ),
        "node3": NodeConfig(
            id="node3",
            type="llm",
            model="gpt-4",
            prompt="Test prompt 3",
            level=2,
            dependencies=["node2"]
        )
    }
    
    for node in nodes.values():
        script_chain.add_node(node)
    
    # Calculate levels
    levels = script_chain._calculate_execution_levels()
    
    # Verify levels
    assert len(levels) == 3
    assert "node1" in levels[0].node_ids
    assert "node2" in levels[1].node_ids
    assert "node3" in levels[2].node_ids

@pytest.mark.asyncio
async def test_parallel_execution(script_chain: ScriptChain):
    """Test parallel execution of nodes at the same level."""
    # Add nodes at the same level
    node1 = NodeConfig(
        id="node1",
        type="llm",
        model="gpt-4",
        prompt="Test prompt 1",
        level=0
    )
    node2 = NodeConfig(
        id="parallel_node",
        type="llm",
        model="gpt-4",
        prompt="Test",
        level=0
    )
    script_chain.add_node(node1)
    script_chain.add_node(node2)
    
    # Mock execution results
    async def mock_execute(*args, **kwargs):
        return NodeExecutionResult(
            success=True,
            output={"result": "success"},
            metadata=NodeMetadata(
                node_id=args[1],
                node_type="llm",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow()
            )
        )
    
    script_chain.execute_node = mock_execute
    
    # Execute
    result = await script_chain.execute()
    
    # Verify both nodes executed
    assert result.success
    assert result.output is not None
    assert len(result.output) == 2

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
    assert result.error is not None
    assert "Test error" in str(result.error)

@pytest.mark.asyncio
async def test_callback_integration(script_chain: ScriptChain, callbacks: Dict[str, Any]):
    """Test callback integration during execution."""
    # Add callbacks
    for callback in callbacks.values():
        script_chain.add_callback(callback)
    
    # Add test node
    node = NodeConfig(
        id="test_node",
        type="llm",
        model="gpt-4",
        prompt="Test",
        level=0
    )
    script_chain.add_node(node)
    
    # Mock successful execution
    async def mock_execute(*args, **kwargs):
        return NodeExecutionResult(
            success=True,
            output={"result": "success"},
            metadata=NodeMetadata(
                node_id="test_node",
                node_type="llm",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow()
            ),
            usage=UsageMetadata(
                prompt_tokens=10,
                completion_tokens=20,
                total_tokens=30,
                cost=0.01,
                model="gpt-4",
                node_id="test_node"
            )
        )
    
    script_chain.execute_node = mock_execute
    
    # Execute
    result = await script_chain.execute()
    
    # Verify callbacks were triggered
    assert result.success
    metrics = callbacks["metrics"].get_metrics()
    assert len(metrics) > 0
    assert any(chain_id for chain_id in metrics.keys())

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
        return NodeExecutionResult(
            success=True,
            output={"result": "success"},
            metadata=NodeMetadata(
                node_id="retry_node",
                node_type="llm",
                start_time=datetime.utcnow(),
                end_time=datetime.utcnow()
            )
        )
    
    script_chain.execute_node = mock_execute
    
    # Execute
    result = await script_chain.execute()
    
    # Verify retry behavior
    assert result.success
    assert attempts == 3
    assert result.output is not None 