"""
Prompt template system with version tracking
"""

import re
from typing import Optional
from packaging import version
from pydantic import BaseModel, Field, field_validator, ConfigDict

def parse_model_version(model_name: str) -> str:
    """Convert model name to semantic version string.
    
    Args:
        model_name: Name of the model (e.g., 'gpt-4', 'gpt-4-turbo')
        
    Returns:
        Semantic version string (e.g., '4.0.0', '4.1.0')
        
    Raises:
        ValueError: If model name cannot be parsed
    """
    if model_name == "gpt-4":
        return "4.0.0"
    elif model_name == "gpt-4-turbo":
        return "4.1.0"
    elif model_name == "gpt-4-32k":
        return "4.0.1"
    else:
        raise ValueError(f"Unsupported model: {model_name}")

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
            print(f"Warning: Missing template key {e}, using unformatted content")
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

    def is_compatible_with_model(self, model_name: str) -> bool:
        """Check if template is compatible with given model version.
        
        Args:
            model_name: Name of the model to check compatibility with
            
        Returns:
            True if model meets minimum version requirement, False otherwise
        """
        try:
            model_ver = version.parse(parse_model_version(model_name))
            min_ver = version.parse(parse_model_version(self.min_model_version))
            return model_ver >= min_ver
        except ValueError:
            return False

    def __init__(self, **data):
        super().__init__(**data)
        # Validate model version compatibility during initialization
        if not self.is_compatible_with_model(data.get('min_model_version', 'gpt-4')):
            raise ValueError(f"Model {data.get('min_model_version')} is too old for this template")

class LLMConfig(BaseModel):
    """Configuration for language models"""
    model: str = Field(..., description="Model identifier (e.g., gpt-4)")
    temperature: float = Field(0.7, ge=0.0, le=1.0, description="Sampling temperature")
    max_tokens: int = Field(500, gt=0, description="Maximum tokens to generate")
    max_context_tokens: int = Field(4000, gt=0, description="Maximum context window size")
    api_key: str = Field(..., description="API key for the model service")

    @field_validator('api_key')
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Validate API key format."""
        if v.startswith('test-'):  # Allow test keys
            return v
        if not (v.startswith('sk-') and len(v) == 51) and not v.startswith('sk-proj-'):
            raise ValueError("API keys must start with 'sk-' and be 51 characters long, or start with 'sk-proj-' for project-specific keys")
        return v

    model_config = ConfigDict(extra="forbid")

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