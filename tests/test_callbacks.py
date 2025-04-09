"""
Tests for the callback system implementation.
"""

import pytest
import asyncio
from typing import Dict, Any
from app.utils.callbacks import LoggingCallback, MetricsCallback, DebugCallback
from app.models.node_models import NodeConfig, NodeExecutionResult

@pytest.mark.asyncio
async def test_logging_callback():
    """Test LoggingCallback functionality."""
    callback = LoggingCallback()
    
    # Test chain events
    await callback.on_chain_start("test_chain", {"total_nodes": 2})
    await callback.on_chain_end("test_chain", {"success": True, "duration": 1.0})
    
    # Test node events
    await callback.on_node_start("test_chain", {"node_id": "node1", "level": 0})
    await callback.on_node_end("test_chain", {"node_id": "node1", "level": 0})
    await callback.on_node_error("test_chain", {
        "node_id": "node1",
        "level": 0,
        "error": {"error": "Test error"}
    })

@pytest.mark.asyncio
async def test_metrics_callback():
    """Test MetricsCallback functionality."""
    callback = MetricsCallback()
    
    # Test chain events
    await callback.on_chain_start("test_chain", {"total_nodes": 2})
    await callback.on_chain_end("test_chain", {"success": True, "duration": 1.0})
    
    # Test node events
    await callback.on_node_start("test_chain", {"node_id": "node1", "level": 0})
    await callback.on_node_end("test_chain", {
        "node_id": "node1",
        "level": 0,
        "result": {"usage": {"total_tokens": 100}}
    })
    
    # Get metrics
    metrics = callback.get_metrics()
    
    # Verify metrics
    assert "chains" in metrics
    assert "test_chain" in metrics["chains"]
    assert metrics["chains"]["test_chain"]["success"] is True
    assert metrics["chains"]["test_chain"]["duration"] > 0
    assert "nodes" in metrics["chains"]["test_chain"]
    assert "node1" in metrics["chains"]["test_chain"]["nodes"]
    assert metrics["chains"]["test_chain"]["nodes"]["node1"]["usage"]["total_tokens"] == 100

@pytest.mark.asyncio
async def test_debug_callback():
    """Test DebugCallback functionality."""
    callback = DebugCallback()
    
    # Test chain events
    await callback.on_chain_start("test_chain", {"total_nodes": 2})
    await callback.on_chain_end("test_chain", {"success": True, "duration": 1.0})
    
    # Test node events
    await callback.on_node_start("test_chain", {"node_id": "node1", "level": 0})
    await callback.on_node_end("test_chain", {"node_id": "node1", "level": 0})
    
    # Test context events
    await callback.on_context_update("node1", {"data": "test"}, {"version": "1.0"})
    
    # Test vector store events
    await callback.on_vector_store_op("store", "node1", "test content", 0.8)
    
    # Get events
    events = callback.get_events()
    
    # Verify events
    assert len(events) > 0
    assert any(e["type"] == "chain_start" for e in events)
    assert any(e["type"] == "chain_end" for e in events)
    assert any(e["type"] == "node_start" for e in events)
    assert any(e["type"] == "node_end" for e in events)
    assert any(e["type"] == "context_update" for e in events)
    assert any(e["type"] == "vector_store_op" for e in events)

@pytest.mark.asyncio
async def test_callback_error_handling():
    """Test error handling in callbacks."""
    callback = LoggingCallback()
    
    # Test with invalid data
    await callback.on_chain_start("test_chain", None)
    await callback.on_chain_end("test_chain", None)
    await callback.on_node_start("test_chain", None)
    await callback.on_node_end("test_chain", None)
    await callback.on_node_error("test_chain", None)

@pytest.mark.asyncio
async def test_metrics_export():
    """Test metrics export functionality."""
    callback = MetricsCallback()
    
    # Generate some metrics
    await callback.on_chain_start("test_chain", {"total_nodes": 2})
    await callback.on_chain_end("test_chain", {"success": True, "duration": 1.0})
    
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
    await callback.on_chain_start("test_chain", {"total_nodes": 2})
    await callback.on_node_start("test_chain", {"node_id": "node1", "level": 0})
    await callback.on_node_end("test_chain", {"node_id": "node1", "level": 0})
    await callback.on_chain_end("test_chain", {"success": True, "duration": 1.0})
    
    # Get metrics
    metrics = callback.get_metrics()
    
    # Verify chain metrics
    chain_metrics = metrics["chains"]["test_chain"]
    assert chain_metrics["node_count"] == 2
    assert chain_metrics["success"] is True
    assert chain_metrics["duration"] > 0
    assert "node1" in chain_metrics["nodes"]
    assert chain_metrics["nodes"]["node1"]["success"] is True

@pytest.mark.asyncio
async def test_callback_node_events():
    """Test callback node event handling."""
    callback = MetricsCallback()
    
    # Test node lifecycle
    await callback.on_node_start("test_chain", {"node_id": "node1", "level": 0})
    await callback.on_node_end("test_chain", {
        "node_id": "node1",
        "level": 0,
        "result": {"usage": {"total_tokens": 100}}
    })
    
    # Get metrics
    metrics = callback.get_metrics()
    
    # Verify node metrics
    node_metrics = metrics["chains"]["test_chain"]["nodes"]["node1"]
    assert node_metrics["level"] == 0
    assert node_metrics["success"] is True
    assert node_metrics["usage"]["total_tokens"] == 100 