import pytest
from pydantic import ValidationError
from app.models.config import MessageTemplate, parse_model_version

def test_parse_model_version():
    """Test model version parsing"""
    assert parse_model_version("gpt-4") == "4.0.0"
    assert parse_model_version("gpt-4-turbo") == "4.1.0"
    assert parse_model_version("gpt-4-32k") == "4.0.1"
    
    with pytest.raises(ValueError, match="Unsupported model"):
        parse_model_version("invalid-model")

def test_message_template_validation():
    """Test message template validation"""
    # Valid template
    template = MessageTemplate(
        role="system",
        content="Process this: {input}",
        version="1.0.0",
        min_model_version="gpt-4"
    )
    assert template.role == "system"
    assert template.version == "1.0.0"
    
    # Invalid role
    with pytest.raises(ValidationError, match="Invalid role"):
        MessageTemplate(
            role="invalid",
            content="test",
            version="1.0.0"
        )
    
    # Invalid version format
    with pytest.raises(ValidationError, match=r"String should match pattern.*"):
        MessageTemplate(
            role="system",
            content="test",
            version="1.0"
        )
    
    # Invalid model version
    with pytest.raises(ValidationError, match="Unsupported model version"):
        MessageTemplate(
            role="system",
            content="test",
            version="1.0.0",
            min_model_version="invalid-model"
        )

def test_template_formatting():
    """Test template content formatting"""
    template = MessageTemplate(
        role="user",
        content="Process this: {input}",
        version="1.0.0"
    )
    
    # Successful formatting
    assert template.format(input="test") == "Process this: test"
    
    # Missing key falls back to unformatted content
    assert template.format() == "Process this: {input}"

def test_model_compatibility():
    """Test model version compatibility validation"""
    from app.models.node_models import NodeConfig, LLMConfig, NodeMetadata
    
    # Create a NodeConfig with GPT-4
    config = NodeConfig(
        metadata=NodeMetadata(
            node_id="test-node",
            node_type="ai",
            version="1.0.0"
        ),
        llm_config=LLMConfig(
            model="gpt-4",
            temperature=0.7,
            api_key="sk-test-key"
        ),
        templates=[
            MessageTemplate(
                role="system",
                content="test",
                version="1.0.0",
                min_model_version="gpt-4"  # Compatible
            )
        ]
    )
    
    # Should fail with incompatible version
    with pytest.raises(ValueError, match="Model gpt-4 is too old for template requiring gpt-4-turbo"):
        NodeConfig(
            metadata=NodeMetadata(
                node_id="test-node",
                node_type="ai",
                version="1.0.0"
            ),
            llm_config=LLMConfig(
                model="gpt-4",
                temperature=0.7,
                api_key="sk-test-key"
            ),
            templates=[
                MessageTemplate(
                    role="system",
                    content="test",
                    version="1.0.0",
                    min_model_version="gpt-4-turbo"  # Requires newer version
                )
            ]
        ) 