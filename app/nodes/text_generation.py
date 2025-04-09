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
from app.utils.context import GraphContextManager
from app.utils.retry import AsyncRetry
from app.nodes.base import BaseNode
from app.utils.logging import logger
from app.utils.tracking import track_usage

logger = logging.getLogger(__name__)

class OpenAIErrorHandler:
    """Centralized error handling for OpenAI API calls"""
    
    @classmethod
    def classify_error(cls, error: Exception) -> str:
        """Classify OpenAI API errors into standardized error types.
        
        Args:
            error: The exception to classify
            
        Returns:
            Standardized error type string
        """
        # Check for specific error types
        error_map = {
            APIError: "APIError",
            RateLimitError: "RateLimitError",
            Timeout: "TimeoutError"
        }
        
        # Check for specific error messages
        error_message = str(error).lower()
        if "api key" in error_message or "authentication" in error_message:
            return "AuthenticationError"
        elif "rate limit" in error_message:
            return "RateLimitError"
        elif "timeout" in error_message:
            return "TimeoutError"
        
        # Return mapped error type or default
        return error_map.get(type(error), "UnknownError")
    
    @classmethod
    def format_error_message(cls, error: Exception) -> str:
        """Create user-friendly error messages.
        
        Args:
            error: The exception to format
            
        Returns:
            User-friendly error message
        """
        error_type = cls.classify_error(error)
        
        if error_type == "APIError":
            return "API service unavailable. Please try again later."
        elif error_type == "RateLimitError":
            return "Rate limit exceeded. Please adjust your request rate."
        elif error_type == "TimeoutError":
            return f"Request timed out. Please try again."
        elif error_type == "AuthenticationError":
            return "Authentication failed. Please check your API key."
        
        return str(error)

class TextGenerationNode(BaseNode):
    """Node for text generation using OpenAI's API"""
    
    # Shared client pool for all instances
    _client_pool = {}
    
    def __init__(self, config: NodeConfig, context_manager: GraphContextManager):
        """Initialize the node with configuration and context manager.
        
        Args:
            config: Node configuration
            context_manager: Context manager for handling node context
        """
        super().__init__(config)
        self.context_manager = context_manager
        self._client = None
        self.llm_config = config.llm_config
        self.templates = config.templates
    
    @property
    def client(self) -> AsyncOpenAI:
        """Get the OpenAI client instance from the shared pool.
        
        Returns:
            AsyncOpenAI client instance
        """
        if self.llm_config.api_key not in self._client_pool:
            self._client_pool[self.llm_config.api_key] = AsyncOpenAI(
                api_key=self.llm_config.api_key
            )
        return self._client_pool[self.llm_config.api_key]
    
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
    
    @track_usage
    async def execute(self, context: Dict[str, Any]) -> NodeExecutionResult:
        """Execute text generation with the given context."""
        start_time = datetime.utcnow()
        
        try:
            # Run pre-execute hook
            context = await self.pre_execute(context)
            
            # Retrieve optimized context using LangChain
            node_context = self.context_manager.get_context_with_optimization(self.node_id)
            context.update(node_context)
            
            # Validate prompt
            prompt = context.get("prompt")
            if not prompt:
                raise ValueError("No prompt provided in context")
            
            # Create messages
            messages = self._build_messages(context, context)
            
            # Call OpenAI API
            try:
                response = await self.client.chat.completions.create(
                    model=self.llm_config.model,
                    messages=messages,
                    temperature=self.llm_config.temperature,
                    max_tokens=self.llm_config.max_tokens
                )
                
                # Extract response content
                output = response.choices[0].message.content
                
                # Create successful result
                result = NodeExecutionResult(
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
                        model=self.llm_config.model,
                        node_id=self.node_id  # Add node_id to usage metadata
                    )
                )
                
                # Run post-execute hook
                return await self.post_execute(result)
                
            except Exception as e:
                # Use centralized error handling
                error_type = OpenAIErrorHandler.classify_error(e)
                error_message = OpenAIErrorHandler.format_error_message(e)
                
                result = NodeExecutionResult(
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
                
                # Run post-execute hook even for errors
                return await self.post_execute(result)
                
        except ValueError as e:
            # Handle validation errors
            result = NodeExecutionResult(
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
            
            # Run post-execute hook for validation errors
            return await self.post_execute(result)

    @property
    def input_keys(self) -> List[str]:
        """Get list of input keys from schema"""
        return list(self.config.input_schema.keys())
    
    @property
    def output_keys(self) -> List[str]:
        """Get list of output keys from schema"""
        return list(self.config.output_schema.keys())

    def get_template(self, role: str) -> MessageTemplate:
        """Get a template by role.
        
        Args:
            role: The role of the template to retrieve
            
        Returns:
            The matching MessageTemplate
            
        Raises:
            ValueError: If no template is found for the given role
        """
        try:
            return next(t for t in self.templates if t.role == role)
        except StopIteration:
            raise ValueError(f"No template found for role: {role}")

    def _build_messages(self, inputs: Dict, context: Dict) -> List[Dict]:
        """Construct messages with templates"""
        messages = []
        
        # Add system message if template exists
        try:
            system_template = self.get_template('system')
            # Combine inputs and context for template formatting
            format_args = {**inputs, **context}
            messages.append({
                "role": "system",
                "content": system_template.content.format(**format_args)
            })
        except ValueError:
            # No system template, continue without it
            pass
            
        # Add user message
        try:
            user_template = self.get_template('user')
            # Combine inputs and context for template formatting
            format_args = {**inputs, **context}
            messages.append({
                "role": "user",
                "content": user_template.content.format(**format_args)
            })
        except ValueError:
            # Fall back to direct prompt if no user template
            messages.append({
                "role": "user",
                "content": inputs.get("query", inputs.get("prompt", ""))
            })
            
        return messages

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
        return OpenAIErrorHandler.format_error_message(error)

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