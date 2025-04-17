"""
Simplified context cache management for graph-based node execution.
Stores and retrieves node results.

TODO: Implement token counting and vector store integration based on max_tokens and vector_store.
"""

from typing import Dict, Any, Optional, Union, List
from app.models.node_models import NodeExecutionResult
from app.vector.base import VectorStoreInterface
import networkx as nx
import logging
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class GraphContextManager:
    """Manages context cache for graph-based node execution results,
    integrating token limits and vector store capabilities."""

    def __init__(
        self,
        graph: Optional[nx.DiGraph] = None,
        max_tokens: int = 4000,
        vector_store: Optional[VectorStoreInterface] = None
    ):
        """Initialize the context manager cache.

        Args:
            graph: Directed graph representing the workflow. Required for parent prioritization.
            max_tokens: Maximum approximate tokens allowed in the context retrieved for a node.
            vector_store: Optional vector store instance for advanced context retrieval.
        """
        if graph is None:
            # Graph is essential for prioritizing parents in get_managed_context
            raise ValueError("Graph must be provided to GraphContextManager for prioritization.")
        self.graph = graph
        self._context_cache: Dict[str, Dict[str, Any]] = {}
        self.max_tokens = max_tokens
        self.vector_store = vector_store
        self.logger = logger

        logger.info(f"GraphContextManager initialized (max_tokens: {self.max_tokens}, vector_store: {'Provided' if self.vector_store else 'Not Provided'}).")
            
    def _estimate_tokens(self, data: Any, usage_info: Optional[Dict[str, Any]] = None) -> int:
        """Estimates token count for given data.

        Prioritizes usage info if available, otherwise approximates based on string length.

        Args:
            data: The data (e.g., node output) to estimate tokens for.
            usage_info: Optional dictionary containing usage metrics (e.g., {'total_tokens': 100}).

        Returns:
            Estimated token count.
        """
        # Priority 1: Use actual token count from usage if available
        if usage_info and isinstance(usage_info.get('total_tokens'), int) and usage_info['total_tokens'] > 0:
            return usage_info['total_tokens']

        # Priority 2: Approximate based on string length (crude approximation)
        try:
            # Convert complex objects to JSON string for more representative length
            if isinstance(data, (dict, list)):
                data_str = json.dumps(data)
            else:
                data_str = str(data)
            # Common approximation: ~4 characters per token
            return max(1, len(data_str) // 4)
        except Exception:
            # Fallback for unstringifiable objects
            return 100 # Return a default penalty value

    async def _cleanup_cache(self):
        """Clean up context cache. 
        NOTE: This implementation currently relies on item count, not token count,
        as _max_cache_size was removed. Needs update to use token limits.
        """
        cache_limit_for_cleanup = 1000
        if len(self._context_cache) > cache_limit_for_cleanup:
            try:
                items_to_sort = [
                    (k, v) for k, v in self._context_cache.items()
                    if isinstance(v.get('timestamp'), datetime)
                ]
                default_timestamp = datetime.min
                sorted_items = sorted(
                    items_to_sort,
                    key=lambda item: item[1].get('timestamp', default_timestamp)
                )
                self._context_cache = dict(sorted_items[-cache_limit_for_cleanup:])
                self.logger.debug(f"Cleaned up context cache to {len(self._context_cache)} items (temporary size limit).")
            except TypeError as e:
                 self.logger.error(f"Error during cache cleanup due to incompatible timestamp types: {e}")
            
    async def get_context(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get raw context (cached result dictionary) for a node. Returns None if not found."""
        result = self._context_cache.get(node_id)
        if result is None:
            self.logger.debug(f"Context cache miss for node {node_id}.")
        else:
            self.logger.debug(f"Context cache hit for node {node_id}.")
        return result
        
    async def get_managed_context(
        self,
        node_id: str,
        requested_dependencies: List[str]
    ) -> Dict[str, Any]:
        """
        Retrieves context from specified dependencies, respecting token limits.
        Prioritizes direct parents and truncates output if necessary.

        Args:
            node_id: The ID of the node requesting context.
            requested_dependencies: A list of node IDs whose output is requested.

        Returns:
            A dictionary where keys are dependency IDs and values are their
            (potentially truncated) outputs.
        """
        assembled_context: Dict[str, Any] = {}
        current_tokens: int = 0

        if node_id not in self.graph:
             self.logger.warning(f"Node {node_id} not found in graph for context retrieval.")
             return {} # Cannot determine parents if node isn't in graph

        try:
            direct_parents = set(self.graph.predecessors(node_id))
        except nx.NetworkXError:
            self.logger.warning(f"Could not get predecessors for node {node_id}.")
            direct_parents = set()

        # Create prioritized list: requested direct parents first, then other requested dependencies
        prioritized_deps = [dep for dep in requested_dependencies if dep in direct_parents]
        prioritized_deps.extend([dep for dep in requested_dependencies if dep not in direct_parents])

        self.logger.debug(f"Context request for node {node_id}. Prioritized deps: {prioritized_deps}")

        for dep_id in prioritized_deps:
            if current_tokens >= self.max_tokens:
                self.logger.debug(f"Max tokens {self.max_tokens} reached for {node_id}. Stopping context assembly.")
                break # Stop if token budget is already exceeded

            cached_result = await self.get_context(dep_id)

            if cached_result and cached_result.get('success'):
                output_data = cached_result.get('output')
                usage_data = cached_result.get('usage') # Fetch usage info if available

                if output_data is None:
                    self.logger.debug(f"Dependency {dep_id} has null output. Skipping.")
                    continue

                estimated_tokens = self._estimate_tokens(output_data, usage_data)
                remaining_tokens = self.max_tokens - current_tokens

                if estimated_tokens <= remaining_tokens:
                    # Fits entirely
                    assembled_context[dep_id] = output_data
                    current_tokens += estimated_tokens
                    self.logger.debug(f"Added full output of {dep_id} ({estimated_tokens} tokens) to context for {node_id}. Total: {current_tokens}/{self.max_tokens}")
                elif remaining_tokens > 10: # Only truncate if there's meaningful space left (e.g., > 10 tokens)
                    # Needs truncation
                    try:
                        # Convert complex objects to JSON string for truncation
                        if isinstance(output_data, (dict, list)):
                            output_str = json.dumps(output_data, indent=2) # Pretty print slightly
                        else:
                            output_str = str(output_data)

                        # Estimate characters needed (~4 chars/token)
                        chars_to_keep = max(0, remaining_tokens * 4)
                        truncated_output = output_str[:chars_to_keep] + "\n... [TRUNCATED]"

                        # Estimate tokens for the truncated part only
                        truncated_tokens = self._estimate_tokens(truncated_output)
                        
                        # Add the truncated string to the context
                        assembled_context[dep_id] = truncated_output
                        current_tokens += truncated_tokens # Add estimated tokens of truncated content
                        # Clamp to max_tokens to avoid slight overshoots from approximation
                        current_tokens = min(current_tokens, self.max_tokens) 
                        self.logger.debug(f"Added TRUNCATED output of {dep_id} ({truncated_tokens} tokens) to context for {node_id}. Total: {current_tokens}/{self.max_tokens}")
                        # Stop after truncating one item, as the budget is likely full
                        break 
                    except Exception as e:
                        self.logger.error(f"Error truncating output for {dep_id}: {e}. Skipping.", exc_info=True)
                else:
                    # Doesn't fit, and not enough space to truncate meaningfully
                    self.logger.debug(f"Output of {dep_id} ({estimated_tokens} tokens) doesn't fit in remaining space ({remaining_tokens} tokens) for {node_id}. Skipping.")
                    # Optionally break here if you want to stop as soon as one dependency doesn't fit
                    # break

            elif cached_result:
                 self.logger.debug(f"Dependency {dep_id} execution failed. Skipping context.")
            else:
                 self.logger.debug(f"No cached result found for dependency {dep_id}. Skipping context.")

        return assembled_context

    async def update_context(self, node_id: str, result: NodeExecutionResult):
        """Update context cache with results from a NodeExecutionResult."""
        if not isinstance(result, NodeExecutionResult):
             self.logger.warning(f"Attempted to update context for {node_id} with non-NodeExecutionResult type: {type(result)}. Skipping update.")
             return

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