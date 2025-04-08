"""
Concrete node implementations using the data models
"""

import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from openai import AsyncOpenAI, OpenAI
from openai import APIError, RateLimitError, Timeout
import logging
import uuid

# Existing imports
from app.models.node_models import (
    NodeConfig, 
    NodeExecutionResult,
    NodeExecutionRecord,
    NodeMetadata,
    UsageMetadata
)
from app.models.config import LLMConfig, MessageTemplate
from app.utils.context import ContextManager
from app.utils.retry import AsyncRetry
from app.nodes.base import BaseNode
from app.utils.logging import logger

logger = logging.getLogger(__name__)

class TextGenerationNode(BaseNode):
    """Node for text generation using OpenAI's API"""
    
    def __init__(self, config: NodeConfig):
        """Initialize the text generation node.
        
        Args:
            config: Node configuration
        """
        super().__init__(config)
        self.llm_config = config.llm_config
        self.templates = config.templates
        self._client = None
    
    @property
    def client(self) -> OpenAI:
        """Get the OpenAI client instance."""
        if self._client is None:
            self._client = OpenAI(api_key=self.llm_config.api_key)
        return self._client
    
    @classmethod
    def create(cls, llm_config: LLMConfig) -> 'TextGenerationNode':
        """Create a new text generation node with the given LLM config."""
        config = NodeConfig(
            metadata=NodeMetadata(
                node_id=f"text_generation_{uuid.uuid4().hex[:8]}",
                node_type="ai",
                version="1.0.0",
                description="Text generation using OpenAI"
            ),
            llm_config=llm_config,
            templates=[]
        )
        return cls(config)
    
    async def execute(self, context: Dict[str, Any]) -> NodeExecutionResult:
        """Execute text generation with the given context."""
        start_time = datetime.utcnow()
        
        try:
            # Validate prompt
            prompt = context.get("prompt")
            if not prompt:
                raise ValueError("No prompt provided in context")
            
            # Create messages
            messages = []
            
            # Add system message if templates exist
            if self.templates:
                for template in self.templates:
                    if template.role == "system":
                        messages.append({
                            "role": "system",
                            "content": template.content.format(**context)
                        })
            
            # Add user message
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Call OpenAI API
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_config.model,
                    messages=messages,
                    temperature=self.llm_config.temperature,
                    max_tokens=self.llm_config.max_tokens
                )
                
                # Extract response content
                output = response.choices[0].message.content
                
                # Create successful result
                return NodeExecutionResult(
                    success=True,
                    output=output,
                    error=None,
                    metadata=NodeMetadata(
                        node_id=self.node_id,
                        node_type=self.node_type,
                        version=self.config.metadata.version,
                        description=self.config.metadata.description,
                        error_type=None,
                        timestamp=datetime.utcnow()
                    ),
                    duration=(datetime.utcnow() - start_time).total_seconds(),
                    timestamp=datetime.utcnow(),
                    usage=UsageMetadata(
                        prompt_tokens=response.usage.prompt_tokens,
                        completion_tokens=response.usage.completion_tokens,
                        total_tokens=response.usage.total_tokens,
                        api_calls=1,
                        model=self.llm_config.model
                    )
                )
                
            except Exception as e:
                # Handle API errors
                error_type = "APIError"
                error_message = str(e)
                
                if "API key" in error_message:
                    error_type = "AuthenticationError"
                elif "rate limit" in error_message.lower():
                    error_type = "RateLimitError"
                elif "timeout" in error_message.lower():
                    error_type = "TimeoutError"
                
                return NodeExecutionResult(
                    success=False,
                    output=None,
                    error=error_message,
                    metadata=NodeMetadata(
                        node_id=self.node_id,
                        node_type=self.node_type,
                        version=self.config.metadata.version,
                        description=self.config.metadata.description,
                        error_type=error_type,
                        timestamp=datetime.utcnow()
                    ),
                    duration=(datetime.utcnow() - start_time).total_seconds(),
                    timestamp=datetime.utcnow()
                )
                
        except ValueError as e:
            # Handle validation errors
            return NodeExecutionResult(
                success=False,
                output=None,
                error=str(e),
                metadata=NodeMetadata(
                    node_id=self.node_id,
                    node_type=self.node_type,
                    version=self.config.metadata.version,
                    description=self.config.metadata.description,
                    error_type="ValueError",
                    timestamp=datetime.utcnow()
                ),
                duration=(datetime.utcnow() - start_time).total_seconds(),
                timestamp=datetime.utcnow()
            )

    @property
    def input_keys(self) -> List[str]:
        """Get list of input keys from schema"""
        return list(self.config.input_schema.keys())
    
    @property
    def output_keys(self) -> List[str]:
        """Get list of output keys from schema"""
        return list(self.config.output_schema.keys())

    def _init_template(self, role: str) -> MessageTemplate:
        """Validate and initialize versioned templates"""
        template = next(
            t for t in self.config.templates 
            if t.role == role and t.version == self.config.metadata.version
        )
        if template.min_model_version > self.config.llm_config.model:
            raise ValueError(
                f"Template {template.version} requires model version "
                f"{template.min_model_version}, but using {self.config.llm_config.model}"
            )
        return template

    def _build_messages(self, inputs: Dict, context: Dict) -> List[Dict]:
        """Construct messages with versioned templates"""
        return [
            self._init_template('system').format(
                context=context.get("background", ""),
                instructions=inputs.get("instructions", "")
            ),
            self._init_template('user').format(
                query=inputs.get("query", ""),
                **inputs.get("additional_args", {})
            )
        ]

    def _process_response(self, response: Dict) -> Tuple[str, Dict]:
        """Extract and validate response content"""
        if not response.choices:
            raise ValueError("Empty response from API")
        
        content = response.choices[0].message.content
        usage = {
            "prompt_tokens": int(response.usage.prompt_tokens),
            "completion_tokens": int(response.usage.completion_tokens),
            "total_tokens": int(response.usage.total_tokens)
        }
        
        if not content.strip():
            raise ValueError("Empty content in response")
        
        return content.strip(), usage

    def _format_error(self, error: Exception) -> str:
        """Create user-friendly error messages"""
        if isinstance(error, APIError):
            return "API service unavailable. Please try again later."
        if isinstance(error, RateLimitError):
            return "Rate limit exceeded. Please adjust your request rate."
        if isinstance(error, Timeout):
            return f"Request timed out after {self.config.timeout}s"
        return str(error)

    def _update_execution_stats(self, result: NodeExecutionResult):
        """Update node execution statistics"""
        self.execution_record.executions += 1
        
        if result.success:
            self.execution_record.successes += 1
            tokens = result.metadata.get("total_tokens", 0)
            self.execution_record.token_usage[self.config.llm_config.model] = \
                self.execution_record.token_usage.get(self.config.llm_config.model, 0) + tokens
        else:
            self.execution_record.failures += 1
        
        # Update average duration (fixed calculation)
        self.execution_record.avg_duration = (
            (self.execution_record.avg_duration * (self.execution_record.executions - 1) 
            + result.duration) 
            / self.execution_record.executions
        )
        
        self.execution_record.last_executed = datetime.utcnow()