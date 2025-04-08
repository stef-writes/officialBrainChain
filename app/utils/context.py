"""
Simple context management with token awareness and inheritance
"""

from typing import Dict, Any, Union, List, Optional, TYPE_CHECKING
import tiktoken
import json
from datetime import datetime
from uuid import uuid4
import logging

if TYPE_CHECKING:
    from app.models.node_models import UsageMetadata

logger = logging.getLogger(__name__)

class ContextManager:
    """Manages workflow context with token limit awareness"""
    
    def __init__(self, max_context_tokens: Optional[int] = None):
        """Initialize the context manager.
        
        Args:
            max_context_tokens: Optional maximum number of tokens allowed in context
        """
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.max_context_tokens = max_context_tokens
        self.contexts: Dict[str, Dict[str, Any]] = {}
        self.usage_stats: Dict[str, Dict[str, int]] = {}

    def set_context(self, node_id: str, data: Dict[str, Any]) -> None:
        """Set context for a node.
        
        Args:
            node_id: ID of the node
            data: Context data to set
        """
        self.contexts[node_id] = {
            "data": data,
            "version": uuid4().hex[:8],
            "timestamp": datetime.utcnow().isoformat()
        }
        logger.debug(f"Set context for node {node_id} with version {self.contexts[node_id]['version']}")

    def get_context(self, node_id: str) -> Dict[str, Any]:
        """Get context for a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Context data for the node
        """
        context = self.contexts.get(node_id, {})
        return context.get("data", {}) if isinstance(context, dict) else {}

    def get_context_with_version(self, node_id: str) -> Dict[str, Any]:
        """Get context with version information for a node.
        
        Args:
            node_id: ID of the node
            
        Returns:
            Context data with version information
        """
        return self.contexts.get(node_id, {
            "data": {},
            "version": None,
            "timestamp": None
        })

    def clear_context(self, node_id: str) -> None:
        """Clear context for a node.
        
        Args:
            node_id: ID of the node
        """
        if node_id in self.contexts:
            del self.contexts[node_id]
            logger.debug(f"Cleared context for node {node_id}")

    def get_usage_stats(self) -> Dict[str, Dict[str, int]]:
        """Get usage statistics for all nodes.
        
        Returns:
            Dictionary of usage statistics by node ID
        """
        return self.usage_stats

    def _optimize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize context to fit within token limit"""
        result = {}
        remaining_tokens = self.max_context_tokens
        
        # First pass: include short values and high-priority keys
        high_priority = {"prompt", "system", "output", "error"}
        for key in list(high_priority & context.keys()):
            value = context[key]
            tokens = self._count_tokens(value)
            if tokens <= remaining_tokens:
                result[key] = value
                remaining_tokens -= tokens
                
        # Second pass: include remaining values
        for key, value in context.items():
            if key in result:
                continue
                
            tokens = self._count_tokens(value)
            if tokens <= remaining_tokens:
                result[key] = value
                remaining_tokens -= tokens
            elif isinstance(value, str):
                # Truncate long strings
                trunc_tokens = remaining_tokens - 1
                if trunc_tokens > 0:
                    encoded = self.encoder.encode(value)[:trunc_tokens]
                    result[key] = self.encoder.decode(encoded) + "..."
                remaining_tokens = 0
            
            if remaining_tokens <= 0:
                break
                
        return result

    def _count_tokens(self, value: Any) -> int:
        """Count tokens for a value"""
        if isinstance(value, (str, int, float, bool)):
            return len(self.encoder.encode(str(value)))
        elif isinstance(value, (list, dict)):
            return len(self.encoder.encode(json.dumps(value)))
        return len(self.encoder.encode(str(value)))

    def _get_parent_nodes(self, node_id: str) -> List[str]:
        """Get parent nodes for a given node ID.
        This method should be overridden by ScriptChain."""
        return []

    def get_context_with_optimization(self, node_id: str, include_parents: bool = True) -> Dict[str, Any]:
        """Compatibility method for existing code"""
        return self.get_context(node_id)

    def track_usage(self, usage: 'UsageMetadata') -> None:
        """Track token usage for a node.
        
        Args:
            usage: The usage metadata to track
        """
        # Store the usage stats
        node_id = getattr(usage, 'node_id', 'unknown')
        self.usage_stats[node_id] = {
            "total_tokens": usage.total_tokens or 0,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Update token counts if needed
        if node_id in self.usage_stats:
            # Add the total tokens to the existing count
            self.usage_stats[node_id]["total_tokens"] += usage.total_tokens or 0
        else:
            # Create a new entry if it doesn't exist
            self.usage_stats[node_id] = {
                "total_tokens": usage.total_tokens or 0,
                "timestamp": datetime.utcnow().isoformat()
            }