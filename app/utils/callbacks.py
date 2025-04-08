from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from app.models.node_models import NodeConfig, NodeExecutionResult, UsageMetadata

class ScriptChainCallback(ABC):
    """Base class for ScriptChain callbacks."""
    
    @abstractmethod
    async def on_chain_start(self, chain_id: str, config: Dict[str, Any]) -> None:
        """Called when the chain starts execution."""
        pass
    
    @abstractmethod
    async def on_chain_end(self, chain_id: str, result: Dict[str, Any]) -> None:
        """Called when the chain completes execution."""
        pass
    
    @abstractmethod
    async def on_node_start(self, node_id: str, config: NodeConfig) -> None:
        """Called when a node starts execution."""
        pass
    
    @abstractmethod
    async def on_node_complete(
        self,
        node_id: str,
        result: NodeExecutionResult,
        usage: Optional[UsageMetadata] = None
    ) -> None:
        """Called when a node completes execution."""
        pass
    
    @abstractmethod
    async def on_node_error(
        self,
        node_id: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called when a node encounters an error."""
        pass
    
    @abstractmethod
    async def on_context_update(
        self,
        node_id: str,
        context: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Called when node context is updated."""
        pass 