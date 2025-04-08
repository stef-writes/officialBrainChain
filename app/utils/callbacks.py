from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from app.models.node_models import NodeConfig, NodeExecutionResult

class ScriptChainCallback(ABC):
    """Abstract base class for script chain callbacks."""
    
    @abstractmethod
    async def on_chain_start(self, chain_id: str, config: Dict[str, Any]) -> None:
        """Called when a chain starts execution.
        
        Args:
            chain_id: Unique identifier for the chain
            config: Chain configuration including:
                - node_count: Number of nodes in the chain
                - execution_order: List of node IDs in execution order
                - created_at: ISO format timestamp of chain creation
        """
        pass
    
    @abstractmethod
    async def on_chain_end(self, chain_id: str, result: Dict[str, Any]) -> None:
        """Called when a chain completes execution.
        
        Args:
            chain_id: Unique identifier for the chain
            result: Chain execution result including:
                - success: Whether the chain executed successfully
                - duration: Total execution time in seconds
                - node_results: Dictionary of node IDs to their execution results
                - error: Error message if execution failed
        """
        pass
    
    @abstractmethod
    async def on_node_start(self, node_id: str, config: NodeConfig) -> None:
        """Called when a node starts execution.
        
        Args:
            node_id: ID of the node starting execution
            config: Node configuration
        """
        pass
    
    @abstractmethod
    async def on_node_complete(self, node_id: str, result: NodeExecutionResult) -> None:
        """Called when a node completes execution.
        
        Args:
            node_id: ID of the completed node
            result: Node execution result
        """
        pass
    
    @abstractmethod
    async def on_node_error(self, node_id: str, error: str) -> None:
        """Called when a node encounters an error.
        
        Args:
            node_id: ID of the node that encountered the error
            error: Error message
        """
        pass
    
    @abstractmethod
    async def on_context_update(self, node_id: str, context: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """Called when a node's context is updated.
        
        Args:
            node_id: ID of the node whose context was updated
            context: Updated context data
            metadata: Context metadata including:
                - version: Unique version identifier
                - timestamp: ISO format timestamp of the update
        """
        pass 