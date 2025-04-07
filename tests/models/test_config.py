"""
Tests for configuration models
"""

import pytest
from app.models.config import MessageTemplate, LLMConfig, AppConfig

def test_message_template_validation():
    """Test validation for MessageTemplate model"""
    # Test valid template
    template = MessageTemplate(
        role="system",
        content="This is a {variable} template",
        version="1.0.0",
        min_model_version="gpt-4"
    )
    assert template.role == "system"
    assert template.version == "1.0.0"
    assert template.min_model_version == "gpt-4"
    
    # Test invalid role
    with pytest.raises(ValueError) as excinfo:
        MessageTemplate(
            role="invalid",
            content="This is a template",
            version="1.0.0",
            min_model_version="gpt-4"
        )
    assert "Valid roles: system, user, assistant" in str(excinfo.value)
    
    # Test invalid version format
    with pytest.raises(ValueError) as excinfo:
        MessageTemplate(
            role="system",
            content="This is a template",
            version="1.0",
            min_model_version="gpt-4"
        )
    assert "String should match pattern" in str(excinfo.value)
    
    # Test invalid model version
    with pytest.raises(ValueError) as excinfo:
        MessageTemplate(
            role="system",
            content="This is a template",
            version="1.0.0",
            min_model_version="invalid-model"
        )
    assert "Unsupported model version" in str(excinfo.value)

def test_llm_config_validation():
    """Test validation for LLMConfig model"""
    # Test valid config
    config = LLMConfig(
        model="gpt-4",
        api_key="sk-test-key-for-testing-only",
        temperature=0.7,
        max_tokens=1000
    )
    assert config.model == "gpt-4"
    assert config.temperature == 0.7
    
    # Test invalid model
    with pytest.raises(ValueError) as excinfo:
        LLMConfig(
            model="invalid-model",
            api_key="sk-test-key-for-testing-only",
            temperature=0.7
        )
    assert "Unsupported model" in str(excinfo.value)
    
    # Test invalid API key format
    with pytest.raises(ValueError) as excinfo:
        LLMConfig(
            model="gpt-4",
            api_key="invalid_key",
            temperature=0.7
        )
    assert "API keys must start with 'sk-'" in str(excinfo.value)
    
    # Test API key length
    with pytest.raises(ValueError) as excinfo:
        LLMConfig(
            model="gpt-4",
            api_key="sk-123",
            temperature=0.7
        )
    assert "51 characters long" in str(excinfo.value)
    
    # Test temperature bounds
    with pytest.raises(ValueError) as excinfo:
        LLMConfig(
            model="gpt-4",
            api_key="sk-test-key-for-testing-only",
            temperature=1.5
        )
    assert "Input should be less than or equal to 1" in str(excinfo.value)
    
    # Test negative temperature
    with pytest.raises(ValueError) as excinfo:
        LLMConfig(
            model="gpt-4",
            api_key="sk-test-key-for-testing-only",
            temperature=-0.1
        )
    assert "Input should be greater than or equal to 0" in str(excinfo.value)
    
    # Test max_tokens validation
    with pytest.raises(ValueError) as excinfo:
        LLMConfig(
            model="gpt-4",
            api_key="sk-test-key-for-testing-only",
            max_tokens=0
        )
    assert "Input should be greater than 0" in str(excinfo.value)

def test_app_config_validation():
    """Test validation for AppConfig model"""
    # Test valid config
    config = AppConfig(
        version="1.0.0",
        environment="development",
        log_level="INFO"
    )
    assert config.version == "1.0.0"
    assert config.environment == "development"
    assert config.log_level == "INFO"
    
    # Test invalid version format
    with pytest.raises(ValueError) as excinfo:
        AppConfig(
            version="1.0",
            environment="development",
            log_level="INFO"
        )
    assert "Version must use semantic format" in str(excinfo.value)
    
    # Test invalid environment
    with pytest.raises(ValueError) as excinfo:
        AppConfig(
            version="1.0.0",
            environment="invalid",
            log_level="INFO"
        )
    assert "Invalid environment" in str(excinfo.value)
    
    # Test invalid log level
    with pytest.raises(ValueError) as excinfo:
        AppConfig(
            version="1.0.0",
            environment="development",
            log_level="INVALID"
        )
    assert "Invalid log level" in str(excinfo.value) 