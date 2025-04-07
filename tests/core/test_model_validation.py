"""
Model validation and error handling tests
"""

import pytest
from pydantic import ValidationError
from app.models.config import LLMConfig, MessageTemplate, AppConfig
from app.models.nodes import NodeConfig, NodeMetadata, NodeExecutionResult

def test_llm_config_validation():
    """Test LLM configuration validation"""
    # Test valid configuration
    valid_config = LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000
    )
    assert valid_config.model == "gpt-4"
    assert valid_config.temperature == 0.7
    assert valid_config.max_tokens == 1000
    
    # Test invalid model
    with pytest.raises(ValidationError):
        LLMConfig(
            api_key="sk-test-key-for-testing-only",
            model="invalid-model",
            temperature=0.7,
            max_tokens=1000
        )
    
    # Test invalid temperature
    with pytest.raises(ValidationError):
        LLMConfig(
            api_key="sk-test-key-for-testing-only",
            model="gpt-4",
            temperature=1.5,  # Should be between 0 and 1
            max_tokens=1000
        )
    
    # Test invalid max_tokens
    with pytest.raises(ValidationError):
        LLMConfig(
            api_key="sk-test-key-for-testing-only",
            model="gpt-4",
            temperature=0.7,
            max_tokens=0  # Should be greater than 0
        )
    
    # Test invalid API key format
    with pytest.raises(ValidationError):
        LLMConfig(
            api_key="invalid-key",
            model="gpt-4",
            temperature=0.7,
            max_tokens=1000
        )

def test_message_template_validation():
    """Test message template validation"""
    # Test valid template
    valid_template = MessageTemplate(
        role="system",
        content="You are a helpful assistant. Background: {background}",
        version="1.0.0",
        min_model_version="gpt-4"
    )
    assert valid_template.role == "system"
    assert valid_template.version == "1.0.0"
    assert valid_template.min_model_version == "gpt-4"
    
    # Test invalid role
    with pytest.raises(ValidationError):
        MessageTemplate(
            role="invalid-role",
            content="Test content",
            version="1.0.0",
            min_model_version="gpt-4"
        )
    
    # Test invalid version format
    with pytest.raises(ValidationError):
        MessageTemplate(
            role="system",
            content="Test content",
            version="1.0",  # Should be semantic version
            min_model_version="gpt-4"
        )
    
    # Test invalid model version
    with pytest.raises(ValidationError):
        MessageTemplate(
            role="system",
            content="Test content",
            version="1.0.0",
            min_model_version="invalid-model"
        )

def test_node_config_validation():
    """Test node configuration validation"""
    # Test valid configuration
    valid_metadata = NodeMetadata(
        node_id="test-node",
        node_type="ai",
        version="1.0.0",
        description="Test node"
    )
    
    valid_llm_config = LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000
    )
    
    valid_config = NodeConfig(
        metadata=valid_metadata,
        llm_config=valid_llm_config
    )
    assert valid_config.metadata.node_id == "test-node"
    assert valid_config.metadata.node_type == "ai"
    assert valid_config.llm_config.model == "gpt-4"
    
    # Test self-reference in dependencies
    with pytest.raises(ValidationError):
        NodeConfig(
            metadata=valid_metadata,
            llm_config=valid_llm_config,
            dependencies=["test-node"]  # Cannot depend on itself
        )
    
    # Test invalid timeout
    with pytest.raises(ValidationError):
        NodeConfig(
            metadata=valid_metadata,
            llm_config=valid_llm_config,
            timeout=0  # Should be greater than 0
        )

def test_node_execution_result_validation():
    """Test node execution result validation"""
    # Test valid result
    valid_result = NodeExecutionResult(
        success=True,
        output="Test output",
        metadata={
            "node_id": "test_node",
            "node_type": "ai",
            "description": "Test execution",
            "usage": {"total_tokens": 100}
        },
        duration=1.5
    )
    assert valid_result.success
    assert valid_result.output == "Test output"
    assert valid_result.metadata.node_id == "test_node"
    assert valid_result.metadata.node_type == "ai"
    assert valid_result.duration == 1.5
    
    # Test error result
    error_result = NodeExecutionResult(
        success=False,
        error="Test error",
        metadata={
            "node_id": "test_node",
            "node_type": "ai",
            "error_type": "ValueError"
        },
        duration=0.5
    )
    assert not error_result.success
    assert error_result.error == "Test error"
    assert error_result.metadata.error_type == "ValueError"
    assert error_result.duration == 0.5
    
    # Test invalid duration
    with pytest.raises(ValidationError):
        NodeExecutionResult(
            success=True,
            output="Test output",
            metadata={
                "node_id": "test_node",
                "node_type": "ai"
            },
            duration=-1.0  # Should be non-negative
        ) 