"""
Concrete node implementations using the data models
"""

import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
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
    UsageMetadata,
    ContextRule
)
from app.models.config import LLMConfig, MessageTemplate
from app.utils.context import GraphContextManager
from app.nodes.base import BaseNode
from app.utils.logging import logger
from app.utils.tracking import calculate_cost
from app.utils.callbacks import ScriptChainCallback

from langchain_community.chat_models import ChatOpenAI
from langchain_core.messages import (
    HumanMessage,
    SystemMessage,
    AIMessage,
    BaseMessage
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
        context_manager: GraphContextManager,
        llm_config: LLMConfig,
        callbacks: Optional[List[ScriptChainCallback]] = None
    ):
        """Initialize text generation node.
        
        Args:
            config: Node configuration
            context_manager: Optional context manager
            llm_config: Optional LLM configuration
            callbacks: Optional list of node callbacks
        """
        super().__init__(config)
        self.context_manager = context_manager
        self.llm_config = llm_config
        self.callbacks = callbacks or []
        self.type = "llm"
        self.metadata = config.metadata
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model_name=llm_config.model,
            openai_api_key=llm_config.api_key,
            temperature=llm_config.temperature,
            max_tokens=llm_config.max_tokens
        )
        
        logger.debug(
            f"Initialized TextGenerationNode with model={self.llm.model_name}"
        )
    
    async def execute(self, inputs: Optional[Dict[str, Any]] = None) -> NodeExecutionResult:
        """Execute the node using configured templates and track usage/cost."""
        start_time = time.time()
        inputs = inputs or {}
        node_start_metadata = self.metadata # Capture initial metadata

        # Note: context passed to _build_messages might need refinement depending
        # on how conversation history or other context is managed.
        # For now, using inputs for template formatting.
        context_for_formatting = inputs 

        try:
            # Build messages using templates from config
            # Pass only inputs for formatting for now.
            message_dicts = self._build_messages(inputs=inputs, context={}) 

            # Convert dicts to LangChain BaseMessage objects
            lc_messages: List[BaseMessage] = []
            for msg in message_dicts:
                role = msg.get("role")
                content = msg.get("content", "")
                if role == "system":
                    lc_messages.append(SystemMessage(content=content))
                elif role == "user":
                    lc_messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    lc_messages.append(AIMessage(content=content))
                else:
                    logger.warning(f"Unknown message role '{role}' encountered in node {self.node_id}")
                    # Fallback to human message for unknown roles
                    lc_messages.append(HumanMessage(content=content))

            if not lc_messages:
                 raise ValueError("No messages were generated for LLM call.")

            # Execute LLM call
            # Note: Ensure self.llm.agenerate expects List[List[BaseMessage]]
            # If it expects just List[BaseMessage], remove the outer list brackets.
            # Based on previous read, it likely wants List[List[...]] for batching,
            # so we wrap our single message list in another list.
            response = await self.llm.agenerate([lc_messages])
            
            # Process response
            # Assuming the first result in the first generation is the one we want
            if not response.generations or not response.generations[0]:
                 raise ValueError("LLM response did not contain expected generations.")
            
            output = response.generations[0][0].text
            llm_output_data = response.llm_output or {}
            token_usage = llm_output_data.get("token_usage", {})
            
            # Calculate execution time
            end_time = time.time()
            execution_time = end_time - start_time

            # Calculate cost
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0)
            
            # Ensure total_tokens is consistent if possible, otherwise use sum
            if total_tokens == 0 and (prompt_tokens > 0 or completion_tokens > 0):
                 total_tokens = prompt_tokens + completion_tokens # Recalculate if total is missing

            # Call the imported calculate_cost function
            cost = calculate_cost(
                model_name=self.llm.model_name, # Pass the model name
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens
            )
            
            # Create usage metadata
            usage_meta = UsageMetadata(
                node_id=self.node_id, # Add node_id
                model=self.llm.model_name, # Add model name
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=cost,
                # api_calls: Assuming 1 call per execute for now
            )
            
            # Update NodeMetadata with execution details
            node_end_metadata = node_start_metadata.model_copy(update={
                 'start_time': datetime.fromtimestamp(start_time),
                 'end_time': datetime.fromtimestamp(end_time),
                 'duration': execution_time,
                 'timestamp': datetime.fromtimestamp(end_time) # Update timestamp to end time
            })

            # Create successful result
            result = NodeExecutionResult(
                success=True,
                output={"result": output}, # Wrap output in a dict consistently?
                metadata=node_end_metadata,
                usage=usage_meta,
                execution_time=execution_time,
                context_used=inputs # Store inputs as context used for now
            )
            
            # Call post_execute hook from base class
            return await self.post_execute(result)
            
        except Exception as e:
            end_time = time.time()
            execution_time = end_time - start_time
            error_msg = self._format_error(e) # Use the helper
            logger.error(f"Error executing node {self.node_id}: {error_msg}")
            logger.debug(traceback.format_exc())
            
            # Update NodeMetadata with error details
            node_error_metadata = node_start_metadata.model_copy(update={
                 'start_time': datetime.fromtimestamp(start_time),
                 'end_time': datetime.fromtimestamp(end_time),
                 'duration': execution_time,
                 'timestamp': datetime.fromtimestamp(end_time),
                 'error_type': e.__class__.__name__
            })

            # Create error result
            return NodeExecutionResult(
                success=False,
                error=error_msg,
                metadata=node_error_metadata,
                usage=None, # No usage data on error
                execution_time=execution_time,
                context_used=inputs
            )

    async def validate_input(self, context: Dict[str, Any]) -> bool:
        """Validate execution context against node requirements."""
        # context here is the dictionary passed by ScriptChain or BaseNode.pre_execute
        # Use context for validation checks
        for input_id, rule in self.config.context_rules.items():
            if rule.required and input_id not in context:
                logger.error(f"Node {self.node_id}: Required input '{input_id}' not found in context")
                return False
                
            if input_id in context and rule.max_tokens:
                # TODO: Implement token counting logic for specific input fields if needed
                # This requires a tokenizer compatible with the model
                logger.warning(f"Node {self.node_id}: max_tokens validation for input '{input_id}' not yet implemented.")
                pass # Placeholder for token counting
                
        # Add any other necessary validation based on self.config.input_schema etc.
        logger.debug(f"Node {self.node_id}: Input validation passed.")
        return True

    @property
    def input_keys(self) -> List[str]:
        """Get list of input keys from schema"""
        return list(self.config.input_schema.keys())
    
    @property
    def output_keys(self) -> List[str]:
        """Get list of output keys from schema"""
        return list(self.config.output_schema.keys())

    def get_template(self, role: str) -> Optional[MessageTemplate]:
        """Get a template by role. Returns None if not found."""
        # Assumes self.config.templates is the Dict[str, MessageTemplate]
        return self.config.templates.get(role)

    def _build_messages(self, inputs: Dict, context: Dict) -> List[Dict]:
        """Construct messages using templates defined in NodeConfig."""
        messages = []
        # Combine inputs and context for template formatting
        # Prioritize inputs over context in case of key collision
        format_args = {**context, **inputs} 

        # Add system message if template exists
        system_template = self.get_template('system')
        if system_template:
            try:
                content = system_template.content.format(**format_args)
                messages.append({"role": "system", "content": content})
            except KeyError as e:
                logger.warning(f"Node {self.node_id}: Missing key '{e}' for system template. Skipping.")
        else:
             # Optionally add a default system message if none is provided?
             # Or rely on the user template to carry the main instruction. Let's omit default for now.
             pass

        # Add user message
        user_template = self.get_template('user')
        if user_template:
            try:
                content = user_template.content.format(**format_args)
                messages.append({"role": "user", "content": content})
            except KeyError as e:
                logger.error(f"Node {self.node_id}: Missing key '{e}' for user template. Cannot proceed.")
                raise ValueError(f"Missing required key '{e}' for user template in node {self.node_id}") from e
        else:
            # Fallback if no user template? This might be an error condition.
            # Let's raise an error if no user template is defined, as it's usually essential.
            logger.error(f"Node {self.node_id}: No user template defined in NodeConfig.templates.")
            raise ValueError(f"User template not found for node {self.node_id}")
            
        # Add assistant message template if needed (e.g., for few-shot)
        assistant_template = self.get_template('assistant')
        if assistant_template:
             try:
                 content = assistant_template.content.format(**format_args)
                 messages.append({"role": "assistant", "content": content})
             except KeyError as e:
                 logger.warning(f"Node {self.node_id}: Missing key '{e}' for assistant template. Skipping.")

        logger.debug(f"Node {self.node_id}: Built messages: {messages}")
        return messages

    def _format_error(self, error: Exception) -> str:
        """Create user-friendly error messages using the handler."""
        return OpenAIErrorHandler.format_error_message(error)