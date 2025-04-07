"""
Enhanced context management with token-aware optimization
"""

from typing import Dict, Any, Union, List, Optional
import tiktoken
import time
from datetime import datetime

class ContextManager:
    """Manages workflow context with token limit awareness"""
    
    def __init__(self, max_context_tokens: int = 4000):
        self.storage = {}
        self.error_log = []
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.max_tokens = max_context_tokens
        self._context: Dict[str, Dict[str, Any]] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    def get_relevant_context(self, node_id: str, max_tokens: int = 4000) -> Union[str, List]:
        """Retrieve and truncate context to fit model limits
        
        Args:
            node_id: Target node ID for context requirements
            max_tokens: Model-specific token limit (default: GPT-4)
            
        Returns:
            Truncated context that fits within token limits
        """
        full_context = self._collect_context(node_id)
        return self._truncate_context(full_context, max_tokens)

    def _collect_context(self, node_id: str) -> Dict:
        """Aggregate context from parent nodes"""
        # Implementation from previous version
        return {...}

    def _truncate_context(self, context: Union[str, List], max_tokens: int) -> Union[str, List]:
        """Token-aware context truncation
        
        Handles both string and message list formats
        """
        if isinstance(context, str):
            tokens = self.encoder.encode(context)
            return self.encoder.decode(tokens[:max_tokens])
            
        if isinstance(context, list):
            # Truncate message list while preserving structure
            token_count = 0
            truncated = []
            for msg in context:
                msg_tokens = self.encoder.encode(msg["content"])
                if token_count + len(msg_tokens) > max_tokens:
                    break
                truncated.append(msg)
                token_count += len(msg_tokens)
            return truncated
            
        raise ValueError("Unsupported context format")

    # Previous methods maintained with new internal calls
    def get(self, node_id: str) -> Union[str, List]:
        """Public getter with automatic truncation"""
        return self.get_relevant_context(node_id, self.max_tokens)

    def set_context(self, node_id: str, context: Dict[str, Any]) -> None:
        """Set context for a specific node.
        
        Args:
            node_id: The ID of the node to set context for
            context: The context data to set
        """
        self._context[node_id] = context
        self._metadata[node_id] = {
            "timestamp": datetime.utcnow().isoformat(),
            "context_size": len(str(context))
        }
    
    def get_context(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get context for a specific node.
        
        Args:
            node_id: The ID of the node to get context for
            
        Returns:
            The context data for the node, or None if not found
        """
        return self._context.get(node_id)
    
    def clear_context(self, node_id: Optional[str] = None) -> None:
        """Clear context for a specific node or all nodes.
        
        Args:
            node_id: The ID of the node to clear context for, or None to clear all
        """
        if node_id is None:
            self._context.clear()
            self._metadata.clear()
        else:
            self._context.pop(node_id, None)
            self._metadata.pop(node_id, None)
    
    def get_metadata(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a specific node's context.
        
        Args:
            node_id: The ID of the node to get metadata for
            
        Returns:
            The metadata for the node's context, or None if not found
        """
        return self._metadata.get(node_id)