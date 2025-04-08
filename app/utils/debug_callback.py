import logging
from typing import Any, Dict, Optional
from app.models.node_models import NodeConfig, NodeExecutionResult, UsageMetadata
from app.utils.callbacks import ScriptChainCallback

logger = logging.getLogger(__name__)

class DebugCallback(ScriptChainCallback):
    """A debug callback that logs all chain events."""
    
    def __init__(self):
        self.events = []
    
    async def on_chain_start(self, chain_id: str, config: Dict[str, Any]) -> None:
        """Log chain start."""
        event = {"event": "chain_start", "chain_id": chain_id, "config": config}
        self.events.append(event)
        logger.info(f"Chain {chain_id} starting with config: {config}")
    
    async def on_chain_end(self, chain_id: str, result: Dict[str, Any]) -> None:
        """Log chain completion."""
        event = {"event": "chain_end", "chain_id": chain_id, "result": result}
        self.events.append(event)
        logger.info(f"Chain {chain_id} completed with result: {result}")
    
    async def on_node_start(self, node_id: str, config: NodeConfig) -> None:
        """Log node start."""
        event = {"event": "node_start", "node_id": node_id, "config": config}
        self.events.append(event)
        logger.info(f"Node {node_id} starting with config: {config}")
    
    async def on_node_complete(
        self,
        node_id: str,
        result: NodeExecutionResult,
        usage: Optional[UsageMetadata] = None
    ) -> None:
        """Log node completion."""
        event = {"event": "node_complete", "node_id": node_id, "result": result, "usage": usage}
        self.events.append(event)
        logger.info(
            f"Node {node_id} completed with result: {result}, "
            f"usage: {usage}"
        )
    
    async def on_node_error(
        self,
        node_id: str,
        error: Exception,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log node error."""
        event = {"event": "node_error", "node_id": node_id, "error": str(error), "context": context}
        self.events.append(event)
        logger.error(
            f"Node {node_id} encountered error: {error}, "
            f"context: {context}"
        )
    
    async def on_context_update(
        self,
        node_id: str,
        context: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log context update."""
        event = {"event": "context_update", "node_id": node_id, "context": context, "metadata": metadata}
        self.events.append(event)
        logger.info(
            f"Node {node_id} context updated: {context}, "
            f"metadata: {metadata}"
        ) 