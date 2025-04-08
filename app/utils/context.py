"""
Simple context management with token awareness and inheritance
"""

from typing import Dict, Any, Union, List, Optional, TYPE_CHECKING
import tiktoken
import json
from datetime import datetime
from uuid import uuid4
import logging
from app.context.vector import VectorStore
from pathlib import Path
from app.context.vector import HybridSearchConfig

if TYPE_CHECKING:
    from app.models.node_models import UsageMetadata

logger = logging.getLogger(__name__)

class ContextManager:
    """Manages workflow context with token limit awareness"""
    
    def __init__(self, storage_path: str = "context_store.json", max_context_tokens: int = 1000):
        """Initialize the context manager.
        
        Args:
            storage_path: Path to store context persistently
            max_context_tokens: Maximum number of tokens to keep in context
        """
        self.storage_path = Path(storage_path)
        self.encoder = tiktoken.get_encoding("cl100k_base")
        self.max_context_tokens = max_context_tokens
        self.contexts: Dict[str, Dict[str, Any]] = {}
        self.usage_stats: Dict[str, Dict[str, int]] = {}
        self.vector_store = VectorStore(config=HybridSearchConfig(alpha=0.65))  # Initialize vector store with hybrid config
        self._load()
        logger.info(f"Initialized context manager at {storage_path}")

    def _load(self) -> None:
        """Load contexts from storage"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    self.contexts = json.load(f)
                logger.debug(f"Loaded {len(self.contexts)} contexts from {self.storage_path}")
            except Exception as e:
                logger.error(f"Error loading contexts: {e}")
                self.contexts = {}

    def _save(self) -> None:
        """Save contexts to storage"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(self.contexts, f)
            logger.debug(f"Saved {len(self.contexts)} contexts to {self.storage_path}")
        except Exception as e:
            logger.error(f"Error saving contexts: {e}")

    def set_context(self, node_id: str, context: Dict[str, Any]) -> None:
        """Set context for a node with versioning.
        
        Args:
            node_id: Node identifier
            context: Context data
        """
        # Add version information
        version_id = str(uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        # Optimize context
        optimized_context = self._optimize_context(context)
        
        # Store with version info
        self.contexts[node_id] = {
            'data': optimized_context,
            'version': version_id,
            'timestamp': timestamp
        }
        
        # Save to storage
        self._save()
        logger.debug(f"Set context for node {node_id} with version {version_id}")

    def get_context(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get context for a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Context data if found, None otherwise
        """
        if node_id in self.contexts:
            return self.contexts[node_id]['data']
        return None

    def get_context_with_version(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get context with version information for a node.
        
        Args:
            node_id: Node identifier
            
        Returns:
            Context data with version info if found, None otherwise
        """
        return self.contexts.get(node_id)

    def clear_context(self, node_id: str) -> None:
        """Clear context for a node.
        
        Args:
            node_id: Node identifier
        """
        if node_id in self.contexts:
            del self.contexts[node_id]
            self._save()
            logger.debug(f"Cleared context for node {node_id}")

    def clear_all_contexts(self) -> None:
        """Clear all contexts."""
        self.contexts = {}
        self._save()
        logger.debug("Cleared all contexts")

    def get_usage_stats(self) -> Dict[str, Dict[str, int]]:
        """Get usage statistics for all nodes.
        
        Returns:
            Dictionary of usage statistics by node ID
        """
        return self.usage_stats

    def get_context_with_optimization(self, node_id: str, include_parents: bool = True) -> Dict[str, Any]:
        """Get context with optimization and vector-based retrieval.
        
        Args:
            node_id: ID of the node
            include_parents: Whether to include parent contexts
            
        Returns:
            Optimized context data
        """
        direct_context = self.get_context(node_id)
        logger.debug(f"Retrieved direct context for node {node_id}: {direct_context}")
        vector_context = self._get_vector_context(direct_context)
        logger.debug(f"Retrieved vector context for node {node_id}: {vector_context}")
        return self._optimize_context({**direct_context, **vector_context})
    
    def _get_vector_context(self, current_context: Dict) -> Dict:
        """Get semantically similar contexts using vector store.
        
        Args:
            current_context: Current context data
            
        Returns:
            Dictionary of vector-retrieved contexts
        """
        query_text = " ".join(str(v) for v in current_context.values() 
                            if isinstance(v, (str, int, float)))
        similar = self.vector_store.find_similar(query_text)
        return {f"vector_{ctx['node_id']}": ctx['text'] for ctx in similar}

    def _optimize_context(self, context: Dict) -> Dict:
        """Optimize context by scoring and pruning based on relevance.
        
        Args:
            context: Context to optimize
            
        Returns:
            Optimized context dictionary
        """
        scored = []
        for key, value in context.items():
            score = self._relevance_score(key, value)
            scored.append((key, value, score))
        
        scored.sort(key=lambda x: x[2], reverse=True)
        return self._allocate_tokens(scored)

    def _relevance_score(self, key: str, value: Any) -> float:
        """Calculate relevance score for a context item.
        
        Args:
            key: Context key
            value: Context value
            
        Returns:
            Relevance score between 0 and 1
        """
        base_scores = {
            'output': 1.0,
            'vector_': 0.8, 
            'system': 0.6
        }
        score = next((v for k,v in base_scores.items() if key.startswith(k)), 0.3)
        
        # Recency boost
        if isinstance(value, dict) and 'timestamp' in value:
            hours_old = (datetime.utcnow() - datetime.fromisoformat(value['timestamp'])).total_seconds() / 3600
            score *= max(0.5, 1 - (hours_old / 48))
            
        return score

    def _count_tokens(self, value: Any) -> int:
        """Estimate token count for a value.
        
        Args:
            value: Value to count tokens for
            
        Returns:
            Estimated token count
        """
        if isinstance(value, str):
            # Rough estimate: 1 token per 4 characters
            return len(value) // 4
        elif isinstance(value, dict):
            return sum(self._count_tokens(v) for v in value.values())
        elif isinstance(value, list):
            return sum(self._count_tokens(item) for item in value)
        else:
            # For other types, assume 1 token
            return 1

    def _truncate_value(self, value: Any, max_tokens: int) -> Any:
        """Truncate a value to fit within token budget.
        
        Args:
            value: Value to truncate
            max_tokens: Maximum tokens allowed
            
        Returns:
            Truncated value
        """
        if isinstance(value, str):
            # Truncate string to roughly max_tokens * 4 characters
            return value[:max_tokens * 4]
        elif isinstance(value, dict):
            # Keep only essential keys
            return {k: v for k, v in value.items() if k in ['timestamp', 'version']}
        elif isinstance(value, list):
            # Keep only first few items
            return value[:max_tokens]
        else:
            return value

    def _allocate_tokens(self, scored_items: List) -> Dict:
        """Allocate tokens to context items based on scores.
        
        Args:
            scored_items: List of (key, value, score) tuples
            
        Returns:
            Dictionary of allocated context items
        """
        budget = self.max_context_tokens
        allocated = {}
        
        for key, value, score in scored_items:
            tokens = self._count_tokens(value)
            if tokens <= budget:
                allocated[key] = value
                budget -= tokens
            else:
                allocated[key] = self._truncate_value(value, budget)
                break
                
        return allocated

    def _get_parent_nodes(self, node_id: str) -> List[str]:
        """Get parent nodes for a given node ID.
        This method should be overridden by ScriptChain."""
        return []

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