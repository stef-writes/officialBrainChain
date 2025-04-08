import logging
from typing import Dict, Any, List
from app.utils.callbacks import ScriptChainCallback
from app.models.node_models import NodeConfig, NodeExecutionResult

logger = logging.getLogger(__name__)

class DebugCallback(ScriptChainCallback):
    """Debug callback implementation that logs all chain events."""
    
    def __init__(self):
        """Initialize the debug callback."""
        self.events: List[Dict[str, Any]] = []
    
    async def on_chain_start(self, chain_id: str, config: Dict[str, Any]) -> None:
        """Log chain start event."""
        event = {
            "event": "chain_start",
            "chain_id": chain_id,
            "config": config
        }
        self.events.append(event)
        logger.info(f"Chain {chain_id} started with {config['node_count']} nodes")
    
    async def on_chain_end(self, chain_id: str, result: Dict[str, Any]) -> None:
        """Log chain end event."""
        event = {
            "event": "chain_end",
            "chain_id": chain_id,
            "result": result
        }
        self.events.append(event)
        if result["success"]:
            logger.info(f"Chain {chain_id} completed successfully in {result['duration']} seconds")
        else:
            logger.error(f"Chain {chain_id} failed: {result.get('error', 'Unknown error')}")
    
    async def on_node_start(self, node_id: str, config: NodeConfig) -> None:
        """Log node start event."""
        event = {
            "event": "node_start",
            "node_id": node_id,
            "config": config
        }
        self.events.append(event)
        logger.info(f"Node {node_id} started execution")
    
    async def on_node_complete(self, node_id: str, result: NodeExecutionResult) -> None:
        """Log node completion event."""
        event = {
            "event": "node_complete",
            "node_id": node_id,
            "result": result
        }
        self.events.append(event)
        if result.success:
            logger.info(f"Node {node_id} completed successfully")
        else:
            logger.error(f"Node {node_id} failed: {result.error}")
    
    async def on_node_error(self, node_id: str, error: str) -> None:
        """Log node error event."""
        event = {
            "event": "node_error",
            "node_id": node_id,
            "error": error
        }
        self.events.append(event)
        logger.error(f"Node {node_id} encountered an error: {error}")
    
    async def on_context_update(self, node_id: str, context: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """Log context update event."""
        event = {
            "event": "context_update",
            "node_id": node_id,
            "context": context,
            "metadata": metadata
        }
        self.events.append(event)
        logger.debug(f"Context updated for node {node_id} (version: {metadata['version']})")
    
    def get_events(self) -> List[Dict[str, Any]]:
        """Get all logged events.
        
        Returns:
            List of event dictionaries
        """
        return self.events 