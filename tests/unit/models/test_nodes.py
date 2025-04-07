"""
Unit tests for node models
"""

import pytest
from datetime import datetime
from app.models.nodes import NodeMetadata

def test_node_metadata_creation():
    """Test basic NodeMetadata creation"""
    metadata = NodeMetadata(
        node_id="test-node",
        node_type="ai"
    )
    
    assert metadata.node_id == "test-node"
    assert metadata.node_type == "ai"
    assert metadata.version == "1.0.0"
    assert isinstance(metadata.created_at, datetime)
    assert isinstance(metadata.modified_at, datetime)

def test_node_metadata_optional_fields():
    """Test NodeMetadata with optional fields"""
    metadata = NodeMetadata(
        node_id="test-node",
        node_type="ai",
        owner="test-user",
        description="Test node description"
    )
    
    assert metadata.owner == "test-user"
    assert metadata.description == "Test node description"

def test_node_metadata_version_validation():
    """Test version format validation"""
    # Valid version
    metadata = NodeMetadata(
        node_id="test-node",
        node_type="ai",
        version="1.0.0"
    )
    assert metadata.version == "1.0.0"
    
    # Invalid version should raise error
    with pytest.raises(ValueError):
        NodeMetadata(
            node_id="test-node",
            node_type="ai",
            version="invalid"
        ) 