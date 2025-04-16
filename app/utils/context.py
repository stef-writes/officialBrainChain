"""
Simplified context cache management for graph-based node execution.
Stores and retrieves node results.
"""

from typing import Dict, Any, Optional, Union
from app.models.node_models import NodeExecutionResult
import networkx as nx
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class GraphContextManager:
    """Manages a simple context cache for graph-based node execution results."""
    
    def __init__(
        self,
        graph: Optional[nx.DiGraph] = None,
        max_cache_size: int = 1000
    ):
        """Initialize the context manager cache.
        
        Args:
            graph: Optional directed graph (kept for potential future use or reference).
            max_cache_size: Maximum number of node results to keep in the cache.
        """
        self.graph = graph or nx.DiGraph()
        self._context_cache: Dict[str, Dict[str, Any]] = {}
        self._max_cache_size = max_cache_size
        self.logger = logger

        logger.info(f"GraphContextManager initialized (cache size: {self._max_cache_size}).")
            
    async def _cleanup_cache(self):
        """Clean up context cache if it exceeds maximum size, removing oldest entries."""
        if len(self._context_cache) > self._max_cache_size:
            # Remove oldest entries based on 'timestamp' key
            try:
                # Ensure items have a comparable timestamp
                items_to_sort = [
                    (k, v) for k, v in self._context_cache.items() 
                    if isinstance(v.get('timestamp'), datetime) 
                ]
                # Handle items potentially missing timestamp or having non-datetime type
                # Assign a default old timestamp to items missing it for sorting purposes
                default_timestamp = datetime.min 
                sorted_items = sorted(
                    items_to_sort,
                    key=lambda item: item[1].get('timestamp', default_timestamp)
                )
                # Rebuild cache with the newest items
                self._context_cache = dict(sorted_items[-self._max_cache_size:])
                self.logger.debug(f"Cleaned up context cache to {len(self._context_cache)} items.")
            except TypeError as e:
                 # Log error if timestamps are not comparable
                 self.logger.error(f"Error during cache cleanup due to incompatible timestamp types: {e}")
            
    async def get_context(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get raw context (cached result dictionary) for a node. Returns None if not found."""
        result = self._context_cache.get(node_id)
        if result is None:
            self.logger.debug(f"Context cache miss for node {node_id}.")
        else:
            self.logger.debug(f"Context cache hit for node {node_id}.")
        return result
        
    async def update_context(self, node_id: str, result: NodeExecutionResult):
        """Update context cache with results from a NodeExecutionResult."""
        if not isinstance(result, NodeExecutionResult):
             self.logger.warning(f"Attempted to update context for {node_id} with non-NodeExecutionResult type: {type(result)}. Skipping update.")
             return

        # Extract relevant info from the result to cache
        cached_data = {
            'output': result.output,
            'success': result.success,
            'error': result.error,
            'metadata': result.metadata.model_dump(mode='json') if result.metadata else None,
            'usage': result.usage.model_dump(mode='json') if result.usage else None,
            'timestamp': result.metadata.timestamp if result.metadata and result.metadata.timestamp else datetime.utcnow()
        }
        self._context_cache[node_id] = cached_data
        
        self.logger.debug(f"Updated context cache for node {node_id}")
        await self._cleanup_cache()

    async def clear_context(self, node_id: str):
        """Clear context cache for a specific node."""
        if node_id in self._context_cache:
            del self._context_cache[node_id]
            self.logger.debug(f"Cleared context cache for node {node_id}")

    async def clear_all_contexts(self):
        """Clear the entire context cache."""
        self._context_cache.clear()
        self.logger.info("Cleared all contexts")