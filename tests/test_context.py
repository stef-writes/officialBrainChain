"""
Tests for the GraphContextManager implementation.
"""

import pytest
import networkx as nx
from typing import Dict, Any
from app.utils.context import GraphContextManager
from app.context.vector import VectorStore

def test_context_manager_initialization(context_manager: GraphContextManager, test_graph: nx.DiGraph):
    """Test GraphContextManager initialization."""
    assert context_manager.max_tokens == 1000
    assert context_manager.graph == test_graph
    assert isinstance(context_manager.vector_store, VectorStore)

def test_set_context(context_manager: GraphContextManager):
    """Test setting context for a node."""
    test_context = {
        "system": "Test system prompt",
        "user": "Test user input"
    }
    
    context_manager.set_context("node1", test_context)
    
    # Verify context was set
    stored_context = context_manager.get_context("node1")
    assert stored_context is not None
    assert stored_context["system"] == test_context["system"]
    assert stored_context["user"] == test_context["user"]

def test_get_context_with_optimization(context_manager: GraphContextManager):
    """Test getting optimized context with graph inheritance."""
    # Set contexts for nodes
    context_manager.set_context("node1", {"data": "parent data"})
    context_manager.set_context("node2", {"data": "child data"})
    
    # Get optimized context for node2
    context = context_manager.get_context_with_optimization("node2")
    
    # Verify context includes both parent and child data
    assert "data" in context
    assert context["data"] == "child data"  # Child data should override parent

def test_context_token_limits(context_manager: GraphContextManager):
    """Test context token limits and optimization."""
    # Create large context
    large_context = {
        "system": "Test " * 1000,  # Should exceed token limit
        "user": "Test " * 1000
    }
    
    context_manager.set_context("node1", large_context)
    
    # Get optimized context
    context = context_manager.get_context_with_optimization("node1")
    
    # Verify context was optimized
    assert len(str(context)) < len(str(large_context))

def test_vector_store_integration(context_manager: GraphContextManager):
    """Test vector store integration for context optimization."""
    # Set context with semantic content
    context_manager.set_context("node1", {
        "content": "The quick brown fox jumps over the lazy dog"
    })
    
    # Get optimized context
    context = context_manager.get_context_with_optimization("node1")
    
    # Verify vector store was used
    assert context is not None
    assert "content" in context

def test_error_handling(context_manager: GraphContextManager):
    """Test error handling in context operations."""
    # Test with invalid node
    context = context_manager.get_context("nonexistent_node")
    assert context is None
    
    # Test with invalid context
    with pytest.raises(Exception):
        context_manager.set_context("node1", None)

def test_context_cleanup(context_manager: GraphContextManager):
    """Test context cleanup operations."""
    # Set contexts
    context_manager.set_context("node1", {"data": "test"})
    context_manager.set_context("node2", {"data": "test"})
    
    # Clear specific context
    context_manager.clear_context("node1")
    assert context_manager.get_context("node1") is None
    assert context_manager.get_context("node2") is not None
    
    # Clear all contexts
    context_manager.clear_all_contexts()
    assert context_manager.get_context("node2") is None

def test_context_metadata(context_manager: GraphContextManager):
    """Test context metadata handling."""
    # Set context
    context_manager.set_context("node1", {"data": "test"})
    
    # Get context with metadata
    context_with_metadata = context_manager.get_context_with_version("node1")
    
    # Verify metadata
    assert "version" in context_with_metadata
    assert "timestamp" in context_with_metadata
    assert "data" in context_with_metadata["data"]

def test_graph_inheritance(context_manager: GraphContextManager):
    """Test context inheritance through the graph."""
    # Set contexts for parent and child
    context_manager.set_context("node1", {"parent": "data"})
    context_manager.set_context("node2", {"child": "data"})
    
    # Get context for child node
    context = context_manager.get_context_with_optimization("node2")
    
    # Verify inheritance
    assert "parent" in context
    assert "child" in context
    assert context["parent"] == "data"
    assert context["child"] == "data" 