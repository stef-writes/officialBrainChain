import pytest
import numpy as np
from pathlib import Path
import json
import tempfile
import os
from app.context.vector import VectorStore
from app.utils.context import ContextManager

@pytest.fixture
def temp_store():
    """Create a temporary vector store for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        storage_path = Path(temp_dir) / "test_vector_store.json"
        store = VectorStore(storage_path=storage_path)
        yield store
        # Cleanup is handled by the context manager

def test_add_context(temp_store):
    """Test adding a context to the store."""
    text = "The quick brown fox jumps over the lazy dog"
    metadata = {"node_id": "test1"}
    
    temp_store.add_context(text, metadata)
    
    assert len(temp_store.vectors) == 1
    assert len(temp_store.contexts) == 1
    assert temp_store.contexts[0]["text"] == text
    assert temp_store.contexts[0]["source"] == "test"
    assert temp_store.contexts[0]["id"] == "123"
    assert "timestamp" in temp_store.contexts[0]

def test_find_similar(temp_store):
    """Test finding similar contexts."""
    text = "The quick brown fox jumps over the lazy dog"
    metadata = {"node_id": "test1"}
    temp_store.add_context(text, metadata)
    results = temp_store.find_similar("Tell me about foxes", top_k=2)
    assert len(results) > 0

def test_persistence(temp_store):
    """Test that contexts persist between store instances."""
    # Add a context
    text = "This is a persistent test"
    metadata = {"test": "persistence"}
    temp_store.add_context(text, metadata)
    
    # Create a new store with the same path
    new_store = VectorStore(storage_path=temp_store.storage_path)
    
    # Check that the context was loaded
    assert len(new_store.vectors) == 1
    assert len(new_store.contexts) == 1
    assert new_store.contexts[0]["text"] == text
    assert new_store.contexts[0]["test"] == "persistence"

def test_empty_store(temp_store):
    """Test behavior of an empty store"""
    results = temp_store.find_similar("test query", top_k=3)
    assert results == []

def test_multiple_contexts(temp_store):
    """Test adding multiple contexts."""
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "A journey of a thousand miles begins with a single step",
        "To be or not to be, that is the question",
        "All that glitters is not gold",
        "The only thing we have to fear is fear itself"
    ]
    for i, text in enumerate(texts):
        metadata = {"node_id": f"test{i+1}"}
        temp_store.add_context(text, metadata)
    assert len(temp_store.vectors) == 5

@pytest.mark.asyncio
async def test_vector_storage_and_retrieval(temp_store):
    """Test vector storage and retrieval"""
    text = "AI pipeline management"
    metadata = {"node_id": "test1"}
    temp_store.add_context(text, metadata)
    results = temp_store.find_similar("AI pipeline management", top_k=2)
    assert len(results) > 0

@pytest.mark.asyncio
async def test_vector_context_injection(temp_store):
    """Test vector context injection"""
    context_manager = ContextManager()
    context_manager.set_context("node1", {"key": "value"})
    ctx = context_manager.get_context_with_optimization("node2")
    assert "vector_node1" in ctx 