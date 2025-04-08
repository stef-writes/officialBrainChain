"""
Simple context management with token awareness and inheritance
"""

from typing import Dict, Any, Union, List, Optional
import tiktoken
import json

class ContextManager:
    """Manages workflow context with token limit awareness"""
    
    def __init__(self, max_context_tokens: int = 4000):
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = max_context_tokens
        self._context: Dict[str, Dict[str, Any]] = {}
        self._token_counts: Dict[str, int] = {}

    def set_context(self, node_id: str, context: Dict[str, Any]) -> None:
        """Set context for a node with token tracking"""
        if not isinstance(context, dict):
            raise ValueError("Context must be a dictionary")
        
        # Filter out None values
        context = {k: v for k, v in context.items() if v is not None}
        
        # Calculate token counts
        token_count = 0
        for value in context.values():
            if isinstance(value, (str, int, float, bool)):
                token_count += len(self.encoder.encode(str(value)))
            elif isinstance(value, (list, dict)):
                token_count += len(self.encoder.encode(json.dumps(value)))
            else:
                token_count += len(self.encoder.encode(str(value)))
        
        self._token_counts[node_id] = token_count
        self._context[node_id] = context

    def get_context(self, node_id: str, include_parents: bool = False) -> Dict[str, Any]:
        """Get context for a node, optionally including parent context"""
        if node_id not in self._context:
            raise ValueError(f"No context found for node {node_id}")
            
        if not include_parents:
            return self._optimize_context(self._context[node_id])
            
        # Include parent context
        context = {}
        for parent_id in self._get_parent_nodes(node_id):
            if parent_id in self._context:
                context.update(self._context[parent_id])
        context.update(self._context[node_id])
        
        return self._optimize_context(context)

    def _optimize_context(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize context to fit within token limit"""
        result = {}
        remaining_tokens = self.max_tokens
        
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
        return self.get_context(node_id, include_parents)

    def clear_context(self, node_id: Optional[str] = None) -> None:
        """Clear context for a node or all nodes"""
        if node_id is None:
            self._context.clear()
            self._token_counts.clear()
        else:
            self._context.pop(node_id, None)
            self._token_counts.pop(node_id, None)