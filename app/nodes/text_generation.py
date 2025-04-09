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
import asyncio
import json
import traceback

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
from app.nodes.base import BaseNode
from app.utils.logging import logger
from app.utils.tracking import track_usage

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage
)

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
    """Node for text generation using LLMs"""
    
    def __init__(
        self,
        config: NodeConfig,
        context_manager: Optional[GraphContextManager] = None,
        llm_config: Optional[LLMConfig] = None
    ):
        """Initialize text generation node.
        
        Args:
            config: Node configuration
            context_manager: Optional context manager
            llm_config: Optional LLM configuration
        """
        super().__init__(config)
        self.context_manager = context_manager
        self.llm_config = llm_config or LLMConfig()
        self.type = "llm"
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=self.llm_config.model,
            temperature=self.llm_config.temperature,
            max_tokens=self.llm_config.max_tokens
        )
        
        logger.debug(
            f"Initialized TextGenerationNode with model={self.llm.model_name}"
        )
    
    async def execute(self, context: Optional[Dict] = None) -> NodeExecutionResult:
        """Execute text generation.
        
        Args:
            context: Optional execution context
            
        Returns:
            NodeExecutionResult containing generated text and metadata
        """
        start_time = datetime.utcnow()
        try:
            # Prepare messages
            messages = self._prepare_messages(context)
            
            # Generate text
            response = await self.llm.agenerate([messages])
            generation = response.generations[0][0]
            
            # Extract usage statistics
            usage = response.llm_output.get("token_usage", {})
            
            return NodeExecutionResult(
                success=True,
                output={"text": generation.text},
                metadata=NodeMetadata(
                    node_id=self.config.id,
                    node_type=self.type,
                    start_time=start_time,
                    end_time=datetime.utcnow()
                ),
                usage=UsageMetadata(
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    cost=self._calculate_cost(usage),
                    model=self.llm.model_name,
                    node_id=self.config.id
                )
            )
            
        except Exception as e:
            logger.error(f"Text generation failed: {str(e)}")
            return NodeExecutionResult(
                success=False,
                error=str(e),
                metadata=NodeMetadata(
                    node_id=self.config.id,
                    node_type=self.type,
                    start_time=start_time,
                    end_time=datetime.utcnow(),
                    error_type=e.__class__.__name__,
                    error_traceback=traceback.format_exc()
                )
            )
    
    def _prepare_messages(self, context: Optional[Dict] = None) -> List:
        """Prepare messages for LLM.
        
        Args:
            context: Optional context dictionary
            
        Returns:
            List of messages for the LLM
        """
        messages = []
        
        # Add system message if present
        if self.config.system_message:
            messages.append(SystemMessage(content=self.config.system_message))
            
        # Add context if present
        if context and context.get("messages"):
            for msg in context["messages"]:
                if msg["role"] == "user":
                    messages.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    messages.append(AIMessage(content=msg["content"]))
                    
        # Add prompt
        if self.config.prompt:
            messages.append(HumanMessage(content=self.config.prompt))
            
        return messages
    
    def _calculate_cost(self, usage: Dict[str, int]) -> float:
        """Calculate cost based on token usage.
        
        Args:
            usage: Token usage dictionary
            
        Returns:
            Estimated cost in USD
        """
        # Cost per 1K tokens (approximate)
        costs = {
            "gpt-4": {"prompt": 0.03, "completion": 0.06},
            "gpt-3.5-turbo": {"prompt": 0.002, "completion": 0.002}
        }
        
        model_costs = costs.get(self.llm.model_name, {"prompt": 0, "completion": 0})
        
        prompt_cost = (
            usage.get("prompt_tokens", 0) * model_costs["prompt"] / 1000
        )
        completion_cost = (
            usage.get("completion_tokens", 0) * model_costs["completion"] / 1000
        )
        
        return prompt_cost + completion_cost

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