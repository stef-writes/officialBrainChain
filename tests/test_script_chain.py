"""
Tests for the enhanced ScriptChain implementation.
"""

import pytest
import asyncio
from typing import Dict, Any
from app.chains.script_chain import ScriptChain, ExecutionLevel
from app.models.node_models import (
    NodeConfig, 
    NodeExecutionResult, 
    NodeMetadata, 
    UsageMetadata,
    ContextFormat,
    ContextRule,
    InputMapping
)
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

@pytest.mark.asyncio
async def test_add_node_with_context_rules(script_chain: ScriptChain):
    """Test adding nodes with context rules and format specifications."""
    # Create test node with context rules
    node = NodeConfig(
        id="test_node",
        type="llm",
        model="gpt-4",
        prompt="Test prompt with {input1} and {input2}",
        level=0,
        context_rules={
            "input1": ContextRule(
                include=True,
                format=ContextFormat.TEXT,
                required=True
            ),
            "input2": ContextRule(
                include=True,
                format=ContextFormat.JSON,
                max_tokens=100
            )
        },
        format_specifications={
            "input1": {"prefix": "Input 1: "},
            "input2": {"indent": 2}
        }
    )
    
    # Add node
    script_chain.add_node(node)
    
    # Verify node was added with context rules
    assert "test_node" in script_chain.nodes
    assert script_chain.nodes["test_node"].config.context_rules == node.context_rules
    assert script_chain.nodes["test_node"].config.format_specifications == node.format_specifications

@pytest.mark.asyncio
async def test_input_validation(script_chain: ScriptChain):
    """Test input validation with context rules."""
    # Create test node with required input
    node = NodeConfig(
        id="validation_node",
        type="llm",
        model="gpt-4",
        prompt="Test prompt",
        level=0,
        context_rules={
            "required_input": ContextRule(
                include=True,
                required=True
            )
        }
    )
    
    script_chain.add_node(node)
    
    # Test with missing required input
    result = await script_chain.nodes["validation_node"].validate_inputs({})
    assert not result
    
    # Test with valid input
    result = await script_chain.nodes["validation_node"].validate_inputs({
        "required_input": "test value"
    })
    assert result

@pytest.mark.asyncio
async def test_context_formatting(script_chain: ScriptChain):
    """Test context formatting with different formats."""
    # Create test node with format specifications
    node = NodeConfig(
        id="format_node",
        type="llm",
        model="gpt-4",
        prompt="Test prompt",
        level=0,
        context_rules={
            "text_input": ContextRule(format=ContextFormat.TEXT),
            "json_input": ContextRule(format=ContextFormat.JSON),
            "markdown_input": ContextRule(format=ContextFormat.MARKDOWN),
            "code_input": ContextRule(format=ContextFormat.CODE)
        }
    )
    
    script_chain.add_node(node)
    
    # Test different input formats
    inputs = {
        "text_input": "plain text",
        "json_input": {"key": "value"},
        "markdown_input": "# Heading\nContent",
        "code_input": "def test(): pass"
    }
    
    # Verify formatting
    formatted = await script_chain.nodes["format_node"].prepare_prompt(inputs)
    assert "plain text" in formatted
    assert '"key": "value"' in formatted
    assert "# Heading" in formatted
    assert "def test(): pass" in formatted

@pytest.mark.asyncio
async def test_parallel_execution_with_context(script_chain: ScriptChain):
    """Test parallel execution with context management."""
    # Add nodes at the same level with context dependencies
    node1 = NodeConfig(
        id="node1",
        type="llm",
        model="gpt-4",
        prompt="Test prompt 1",
        level=0
    )
    node2 = NodeConfig(
        id="node2",
        type="llm",
        model="gpt-4",
        prompt="Test prompt 2 with {node1_output}",
        level=0,
        context_rules={
            "node1_output": ContextRule(
                include=True,
                format=ContextFormat.TEXT
            )
        }
    )
    
    script_chain.add_node(node1)
    script_chain.add_node(node2)
    
    # Mock execution results
    async def mock_execute(*args, **kwargs):
        node_id = args[1]
        if node_id == "node1":
            return NodeExecutionResult(
                success=True,
                output="node1 output",
                metadata=NodeMetadata(
                    node_id=node_id,
                    node_type="llm",
                    version="1.0.0"
                )
            )
        else:
            return NodeExecutionResult(
                success=True,
                output="node2 output",
                metadata=NodeMetadata(
                    node_id=node_id,
                    node_type="llm",
                    version="1.0.0"
                )
            )
    
    script_chain.execute_node = mock_execute
    
    # Execute
    result = await script_chain.execute()
    
    # Verify execution and context
    assert result.success
    assert result.output is not None
    assert len(result.output) == 2
    assert "node1" in result.output
    assert "node2" in result.output 