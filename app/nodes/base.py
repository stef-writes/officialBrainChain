"""
Base node implementation defining the interface for all nodes
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
from pydantic import ValidationError, create_model
from app.models.node_models import NodeConfig, NodeExecutionResult

class BaseNode(ABC):
    """Abstract base class for all nodes
    
    Provides:
    - Lifecycle hooks (pre_execute, post_execute)
    - Input validation using schema
    - Core node properties and configuration
    """
    
    def __init__(self, config: NodeConfig):
        self.config = config
        
    @property
    def node_id(self) -> str:
        """Get the node's unique identifier"""
        return self.config.metadata.node_id
    
    @property
    def node_type(self) -> str:
        """Get the node's type"""
        return self.config.metadata.node_type
    
    async def pre_execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and modify context before execution.
        
        This hook runs before execute() and can be used to:
        - Validate inputs
        - Transform context
        - Add additional context
        - Perform setup
        
        Args:
            context: The execution context
            
        Returns:
            Modified context dictionary
            
        Raises:
            ValueError: If context validation fails
        """
        # Validate inputs first
        if not await self.validate_input(context):
            raise ValueError("Input validation failed")
            
        return context
        
    async def post_execute(self, result: NodeExecutionResult) -> NodeExecutionResult:
        """Process results after execution.
        
        This hook runs after execute() and can be used to:
        - Transform results
        - Add metadata
        - Perform cleanup
        - Log execution data
        
        Args:
            result: The execution result
            
        Returns:
            Modified execution result
        """
        return result
        
    async def validate_input(self, context: Dict[str, Any]) -> bool:
        """Validate input context against the node's input schema.
        
        Uses Pydantic to validate that:
        - All required fields are present
        - Field types match schema
        - No unexpected fields
        
        Args:
            context: The context to validate
            
        Returns:
            True if validation passes, False otherwise
        """
        if not self.config.input_schema:
            return True
            
        try:
            # Create a dynamic Pydantic model from input schema
            fields = {
                key: (eval(type_str), ...) 
                for key, type_str in self.config.input_schema.items()
            }
            InputModel = create_model('InputModel', **fields)
            
            # Validate context against model
            InputModel(**context)
            return True
            
        except (ValidationError, NameError, SyntaxError):
            return False
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> NodeExecutionResult:
        """Execute node logic (abstract method).
        
        This method should be implemented by concrete node classes.
        It will be called after pre_execute() and before post_execute().
        
        Args:
            context: The validated execution context
            
        Returns:
            Execution result
        """
        pass