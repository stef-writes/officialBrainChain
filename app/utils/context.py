"""
Context management for LLM node execution
"""

from typing import Dict, List, Optional, Any
from app.models.node_models import NodeExecutionResult, NodeMetadata
from app.utils.logging import logger
from app.models.vector_store import VectorStoreConfig
from app.vector.pinecone_store import PineconeVectorStore

class GraphContextManager:
    """Manages context for graph-based LLM node execution"""
    
    def __init__(
        self,
        max_tokens: int = 4000,
        graph: Optional[Dict] = None,
        vector_store_config: Optional[Dict] = None
    ):
        """Initialize context manager.
        
        Args:
            max_tokens: Maximum number of tokens to include in context
            graph: Optional graph structure for dependency tracking
            vector_store_config: Optional vector store configuration
        """
        self.max_tokens = max_tokens
        self.graph = graph or {}
        
        # Initialize vector store if config provided
        if vector_store_config:
            vs_config = VectorStoreConfig(
                index_name=vector_store_config.get('index_name', 'default-index'),
                environment=vector_store_config.get('environment', 'us-west1'),
                dimension=vector_store_config.get('dimension', 384),
                pod_type=vector_store_config.get('pod_type', 'p1'),
                replicas=vector_store_config.get('replicas', 1)
            )
            self.vector_store = PineconeVectorStore(vs_config)
        else:
            self.vector_store = None
            
        self.context: Dict[str, Any] = {}
        
    def get_context(self, node_id: str) -> Dict[str, Any]:
        """Get context for a specific node.
        
        Args:
            node_id: ID of the node to get context for
            
        Returns:
            Dictionary containing context for the node
        """
        return self.context.get(node_id, {})
        
    async def get_context_with_optimization(
        self,
        node_id: str,
        query: str,
        k: int = 5,
        threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Get optimized context for a node using vector similarity.
        
        Args:
            node_id: ID of the node to get context for
            query: Query string for similarity search
            k: Number of similar contexts to retrieve
            threshold: Similarity threshold
            
        Returns:
            Dictionary containing optimized context
        """
        if not self.vector_store:
            return self.get_context(node_id)
            
        try:
            # Get base context
            base_context = self.get_context(node_id)
            
            # Get similar contexts from vector store
            similar_contexts = await self.vector_store.similarity_search(
                query,
                k=k,
                threshold=threshold
            )
            
            # Merge contexts while respecting token limit
            merged_context = self._merge_contexts(base_context, similar_contexts)
            return merged_context
            
        except Exception as e:
            logger.error(f"Error getting optimized context: {e}")
            return self.get_context(node_id)
            
    async def get_context_with_version(
        self,
        node_id: str,
        version: str = "latest"
    ) -> Dict[str, Any]:
        """Get versioned context for a node.
        
        Args:
            node_id: ID of the node to get context for
            version: Version of context to retrieve
            
        Returns:
            Dictionary containing versioned context
        """
        context = self.get_context(node_id)
        if version == "latest":
            return context
            
        # Get versioned context from vector store
        if self.vector_store:
            try:
                versioned_context = await self.vector_store.get_version(
                    node_id,
                    version
                )
                if versioned_context:
                    return versioned_context
            except Exception as e:
                logger.error(f"Error getting versioned context: {e}")
                
        return context
            
    def set_context(self, node_id: str, context: Dict[str, Any]) -> None:
        """Set context for a node.
        
        Args:
            node_id: ID of the node to set context for
            context: Context dictionary to set
        """
        self.context[node_id] = context
        
    def clear_context(self, node_id: str) -> None:
        """Clear context for a node.
        
        Args:
            node_id: ID of the node to clear context for
        """
        if node_id in self.context:
            del self.context[node_id]
        
    async def update(self, node_id: str, result: NodeExecutionResult) -> None:
        """Update context with node execution result.
        
        Args:
            node_id: ID of the node that was executed
            result: Execution result containing output and metadata
        """
        current_context = self.get_context(node_id)
        current_context.update({
            "output": result.output,
            "metadata": result.metadata.model_dump() if result.metadata else {}
        })
        self.set_context(node_id, current_context)
        
        # Update vector store if available
        if self.vector_store:
            try:
                await self.vector_store.store_context(
                    node_id,
                    current_context,
                    result.metadata.model_dump() if result.metadata else {}
                )
            except Exception as e:
                logger.error(f"Error updating vector store: {e}")
        
    def log_error(self, node_id: str, error: Exception) -> None:
        """Log error for a node.
        
        Args:
            node_id: ID of the node that encountered an error
            error: Exception that was raised
        """
        logger.error(f"Error in node {node_id}: {str(error)}")
        
    def _merge_contexts(
        self,
        base_context: Dict[str, Any],
        similar_contexts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Merge contexts while respecting token limit.
        
        Args:
            base_context: Base context dictionary
            similar_contexts: List of similar context dictionaries
            
        Returns:
            Merged context dictionary
        """
        # Implementation depends on specific token counting and merging logic
        # This is a placeholder that simply returns the base context
        return base_context