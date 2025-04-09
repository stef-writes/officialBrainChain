"""
Tests for the callback system implementation.
"""

import pytest
import asyncio
from typing import Dict, Any
from app.utils.callbacks import LoggingCallback, MetricsCallback
from app.models.node_models import NodeConfig, NodeExecutionResult

@pytest.mark.asyncio
async def test_logging_callback():
    """Test LoggingCallback functionality."""
    callback = LoggingCallback()
    
    # Test chain events
    callback.on_chain_start("test_chain", {"input": "test"})
    callback.on_chain_end("test_chain", {"output": "test"})
    
    # Test node events
    callback.on_node_start("test_chain", "node1", {"input": "test"})
    callback.on_node_end("test_chain", "node1", {"output": "test"})
    callback.on_node_error("test_chain", "node1", Exception("Test error"))

@pytest.mark.asyncio
async def test_metrics_callback():
    """Test MetricsCallback functionality."""
    callback = MetricsCallback()
    
    # Test chain events
    callback.on_chain_start("test_chain", {"input": "test"})
    callback.on_chain_end("test_chain", {"output": "test"})
    
    # Test node events
    callback.on_node_start("test_chain", "node1", {"input": "test"})
    callback.on_node_end("test_chain", "node1", {"output": "test"})
    
    # Get metrics
    metrics = callback.get_metrics()
    
    # Verify metrics
    assert "test_chain" in metrics
    assert metrics["test_chain"]["inputs"] == {"input": "test"}
    assert metrics["test_chain"]["outputs"] == {"output": "test"}
    assert metrics["test_chain"]["success"] is True
    assert "nodes" in metrics["test_chain"]
    assert "node1" in metrics["test_chain"]["nodes"]
    assert metrics["test_chain"]["nodes"]["node1"]["inputs"] == {"input": "test"}
    assert metrics["test_chain"]["nodes"]["node1"]["outputs"] == {"output": "test"}
    assert metrics["test_chain"]["nodes"]["node1"]["success"] is True

@pytest.mark.asyncio
async def test_callback_error_handling():
    """Test error handling in callbacks."""
    callback = LoggingCallback()
    
    # Test with invalid data
    callback.on_chain_start("test_chain", None)
    callback.on_chain_end("test_chain", None)
    callback.on_node_start("test_chain", "node1", None)
    callback.on_node_end("test_chain", "node1", None)
    callback.on_node_error("test_chain", "node1", Exception("Test error"))

@pytest.mark.asyncio
async def test_metrics_export():
    """Test metrics export functionality."""
    callback = MetricsCallback()
    
    # Generate some metrics
    callback.on_chain_start("test_chain", {"input": "test"})
    callback.on_chain_end("test_chain", {"output": "test"})
    
    # Export metrics
    callback.export_metrics("test_metrics.json")
    
    # Verify file was created
    import os
    assert os.path.exists("test_metrics.json")
    os.remove("test_metrics.json")

@pytest.mark.asyncio
async def test_callback_chain_events():
    """Test callback chain event handling."""
    callback = MetricsCallback()
    
    # Test chain lifecycle
    callback.on_chain_start("test_chain", {"input": "test"})
    callback.on_node_start("test_chain", "node1", {"input": "test"})
    callback.on_node_end("test_chain", "node1", {"output": "test"})
    callback.on_chain_end("test_chain", {"output": "test"})
    
    # Get metrics
    metrics = callback.get_metrics()
    
    # Verify chain metrics
    chain_metrics = metrics["test_chain"]
    assert chain_metrics["inputs"] == {"input": "test"}
    assert chain_metrics["outputs"] == {"output": "test"}
    assert chain_metrics["success"] is True
    assert "node1" in chain_metrics["nodes"]
    assert chain_metrics["nodes"]["node1"]["success"] is True

@pytest.mark.asyncio
async def test_callback_node_events():
    """Test callback node event handling."""
    callback = MetricsCallback()
    
    # Initialize chain metrics first
    callback.on_chain_start("test_chain", {"input": "test"})
    
    # Test node lifecycle
    callback.on_node_start("test_chain", "node1", {"input": "test"})
    callback.on_node_end("test_chain", "node1", {"output": "test"})
    
    # Get metrics
    metrics = callback.get_metrics()
    
    # Verify node metrics
    assert "test_chain" in metrics
    assert "nodes" in metrics["test_chain"]
    assert "node1" in metrics["test_chain"]["nodes"]
    node_metrics = metrics["test_chain"]["nodes"]["node1"]
    assert node_metrics["inputs"] == {"input": "test"}
    assert node_metrics["outputs"] == {"output": "test"}
    assert node_metrics["success"] is True 