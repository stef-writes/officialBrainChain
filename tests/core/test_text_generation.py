"""
Tests for text generation functionality
"""

import pytest
from app.models.config import LLMConfig, MessageTemplate
from app.models.node_models import NodeConfig, NodeMetadata
from app.nodes.text_generation import TextGenerationNode

@pytest.mark.asyncio
async def test_text_generation_with_templates():
    """Test text generation with template handling"""
    # Create LLM config
    llm_config = LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    )
    
    # Create templates
    templates = [
        MessageTemplate(
            role="system",
            content="You are a helpful assistant. Background: {background}",
            version="1.0.0",
            min_model_version="gpt-4"
        ),
        MessageTemplate(
            role="user",
            content="Please analyze: {prompt}",
            version="1.0.0",
            min_model_version="gpt-4"
        )
    ]
    
    # Create node config
    node_config = NodeConfig(
        metadata=NodeMetadata(
            node_id="test_node",
            node_type="ai",
            version="1.0.0",
            description="Test node"
        ),
        llm_config=llm_config,
        templates=templates,
        input_schema={
            "prompt": "str",
            "background": "str"
        }
    )
    
    # Create node
    node = TextGenerationNode(node_config)
    
    # Test template retrieval
    system_template = node.get_template("system")
    assert system_template.role == "system"
    assert "You are a helpful assistant" in system_template.content
    
    user_template = node.get_template("user")
    assert user_template.role == "user"
    assert "Please analyze" in user_template.content
    
    # Test message building
    messages = node._build_messages(
        inputs={"prompt": "test prompt"},
        context={"background": "test background"}
    )
    
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert "test background" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "test prompt" in messages[1]["content"]
    
    # Test with missing template
    with pytest.raises(ValueError) as excinfo:
        node.get_template("assistant")
    assert "No template found for role: assistant" in str(excinfo.value)
    
    # Test execution (will fail with auth error, but that's expected)
    result = await node.execute({
        "prompt": "test prompt",
        "background": "test background"  # Match the template's {background} variable
    })
    assert not result.success
    assert "API key" in result.error
    assert result.metadata.error_type == "AuthenticationError"

@pytest.mark.asyncio
async def test_lifecycle_hooks_and_validation():
    """Test node lifecycle hooks and input validation"""
    # Create node with input schema
    llm_config = LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    )
    
    node_config = NodeConfig(
        metadata=NodeMetadata(
            node_id="validation_test",
            node_type="ai",
            version="1.0.0",
            description="Validation test node"
        ),
        llm_config=llm_config,
        input_schema={
            "prompt": "str",
            "temperature": "float",
            "max_tokens": "int"
        }
    )
    
    node = TextGenerationNode(node_config)
    
    # Test input validation success
    valid_context = {
        "prompt": "test prompt",
        "temperature": 0.7,
        "max_tokens": 100
    }
    assert await node.validate_input(valid_context)
    
    # Test input validation failure - wrong type
    invalid_context = {
        "prompt": "test prompt",
        "temperature": "not a float",
        "max_tokens": 100
    }
    assert not await node.validate_input(invalid_context)
    
    # Test input validation failure - missing required field
    incomplete_context = {
        "temperature": 0.7,
        "max_tokens": 100
    }
    assert not await node.validate_input(incomplete_context)
    
    # Test pre_execute with invalid input
    with pytest.raises(ValueError) as excinfo:
        await node.pre_execute(invalid_context)
    assert "Input validation failed" in str(excinfo.value)
    
    # Test full execution with validation
    result = await node.execute(valid_context)
    assert not result.success  # Will fail with auth error, but should pass validation
    assert result.metadata.error_type == "AuthenticationError"  # Expected error
    
    # Test execution with invalid input
    result = await node.execute(invalid_context)
    assert not result.success
    assert "validation failed" in result.error.lower() 