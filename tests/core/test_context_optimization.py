"""
Unit tests for ContextManager
Tests focus on basic context management and token-aware optimization
"""

import pytest
from app.utils.context import ContextManager

def test_basic_context_operations():
    """Test basic context operations"""
    context_manager = ContextManager(max_context_tokens=100)
    
    # Test setting and getting context
    context = {"key": "value", "number": 42}
    context_manager.set_context("node1", context)
    
    result = context_manager.get_context("node1")
    assert result == context
    
    # Test clearing context
    context_manager.clear_context("node1")
    with pytest.raises(ValueError):
        context_manager.get_context("node1")

def test_token_limit_handling():
    """Test handling of token limits"""
    context_manager = ContextManager(max_context_tokens=10)  # Very small token limit
    
    # Create context that exceeds token limit
    long_text = "This is a very long text that will definitely exceed our small token limit by using many words and adding more content to ensure we go over the limit"
    context = {
        "short": "Keep this",
        "long": long_text * 2  # Make it even longer by repeating
    }
    
    context_manager.set_context("node1", context)
    result = context_manager.get_context("node1")
    
    # Short text should be preserved
    assert result["short"] == "Keep this"
    
    # Long text should be truncated
    assert result["long"].endswith("...")
    assert len(result["long"]) < len(long_text)

def test_invalid_context():
    """Test handling of invalid context"""
    context_manager = ContextManager()
    
    # Test with non-dict context
    with pytest.raises(ValueError):
        context_manager.set_context("node1", "not a dict")
    
    # Test with missing node
    with pytest.raises(ValueError):
        context_manager.get_context("nonexistent") 