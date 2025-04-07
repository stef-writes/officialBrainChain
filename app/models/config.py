"""
Prompt template system with version tracking
"""

import re
from typing import Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict
from app.utils.logging import logger

class MessageTemplate(BaseModel):
    """Template for message generation"""
    role: str = Field(..., description="Message role (system, user, assistant)")
    content: str = Field(..., description="Message content template")
    version: str = Field("1.0.0", pattern=r"^\d+\.\d+\.\d+$",
                        description="Template version")
    min_model_version: str = Field("gpt-4", description="Minimum required model version")
    
    def format(self, **kwargs) -> str:
        """Format template with provided values, using defaults for missing keys"""
        try:
            return self.content.format(**kwargs)
        except KeyError as e:
            # Return unformatted content if formatting fails
            logger.warning(f"Missing template key {e}, using unformatted content")
            return self.content

    model_config = ConfigDict(extra="forbid")

    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate message role"""
        valid_roles = ['system', 'user', 'assistant']
        if v not in valid_roles:
            raise ValueError(f"Invalid role. Valid roles: {', '.join(valid_roles)}")
        return v

    @field_validator('version')
    @classmethod
    def validate_version_format(cls, v: str) -> str:
        """Validate version format"""
        if not re.match(r'^\d+\.\d+\.\d+$', v):
            raise ValueError("Version must use semantic format (e.g., 1.2.3)")
        return v

    @field_validator('min_model_version')
    @classmethod
    def validate_model_version(cls, v: str) -> str:
        """Validate model version"""
        valid_models = [
            'gpt-4',
            'gpt-4-turbo',
            'gpt-4-32k'
        ]
        if v not in valid_models:
            raise ValueError(f"Unsupported model version. Valid options: {', '.join(valid_models)}")
        return v

class LLMConfig(BaseModel):
    """Configuration for language model parameters"""
    model: str = Field("gpt-4", description="Model identifier")
    api_key: str = Field(..., description="API key for model access")
    temperature: float = Field(0.7, ge=0, le=1, description="Sampling temperature")
    max_tokens: int = Field(1000, gt=0, description="Maximum tokens to generate")
    top_p: float = Field(1.0, ge=0, le=1, description="Nucleus sampling parameter")
    frequency_penalty: float = Field(0.0, ge=-2, le=2, 
                                   description="Frequency penalty parameter")
    presence_penalty: float = Field(0.0, ge=-2, le=2,
                                  description="Presence penalty parameter")

    model_config = ConfigDict(extra="forbid")

    @field_validator('model')
    @classmethod
    def validate_model(cls, v: str) -> str:
        """Validate model identifier"""
        valid_models = [
            'gpt-4',
            'gpt-4-turbo',
            'gpt-4-32k'
        ]
        if v not in valid_models:
            raise ValueError(f"Unsupported model. Valid options: {', '.join(valid_models)}")
        return v

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key format"""
        # Allow test API keys that start with 'sk-test-'
        if v.startswith('sk-test-'):
            return v
            
        # Validate production API keys
        if not v.startswith('sk-') or len(v) != 51:
            raise ValueError("API keys must start with 'sk-' and be 51 characters long. Check your OpenAI credentials.")
        return v

class AppConfig(BaseModel):
    """Application configuration"""
    version: str = Field(..., description="Application version")
    environment: str = Field("development", description="Runtime environment")
    debug: bool = Field(False, description="Debug mode flag")
    api_version: str = Field("v1", description="API version")
    log_level: str = Field("INFO", description="Logging level")
    
    model_config = ConfigDict(extra="forbid")
    
    @field_validator('version')
    @classmethod
    def validate_version(cls, v: str) -> str:
        """Validate semantic version format"""
        if not re.match(r'^\d+\.\d+\.\d+$', v):
            raise ValueError("Version must use semantic format (e.g., 1.2.3)")
        return v
    
    @field_validator('environment')
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Validate environment setting"""
        valid_envs = ['development', 'testing', 'staging', 'production']
        if v.lower() not in valid_envs:
            raise ValueError(f"Invalid environment. Valid options: {', '.join(valid_envs)}")
        return v.lower()
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate logging level"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        v = v.upper()
        if v not in valid_levels:
            raise ValueError(f"Invalid log level. Valid options: {', '.join(valid_levels)}")
        return v