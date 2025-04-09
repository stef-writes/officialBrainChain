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
from langchain import LangChain
from langchain.vectorstores import PineconeVectorStore
import os
import networkx as nx
from app.models.node_models import NodeExecutionResult, NodeMetadata
import time

if TYPE_CHECKING:
    from app.models.node_models import UsageMetadata

logger = logging.getLogger(__name__)

class GraphContextManager:
    """Enhanced context manager that uses graph-based context inheritance and vector store optimization"""
    
    def __init__(
        self,
        max_tokens: int = 4000,
        graph: Optional[nx.DiGraph] = None,
        vector_store: Optional[PineconeVectorStore] = None
    ):
        """Initialize the context manager.
        
        Args:
            max_tokens: Maximum number of tokens allowed in the context
            graph: NetworkX graph representing the workflow
            vector_store: Vector store for context optimization
        """
        self.max_tokens = max_tokens
        self.graph = graph or nx.DiGraph()
        self.vector_store = vector_store
        self.context_store: Dict[str, Dict[str, Any]] = {}
        self.error_log: Dict[str, List[Dict[str, Any]]] = {}
        self.token_counts: Dict[str, int] = {}
        
        logger.info(f"Initialized GraphContextManager with max_tokens={max_tokens}")
    
    def get_context(self, node_id: str, dependencies: Optional[List[str]] = None) -> Dict[str, Any]:
        """Get context for a node, including outputs from its dependencies.
        
        Args:
            node_id: ID of the node to get context for
            dependencies: Optional list of dependency node IDs to include
            
        Returns:
            Dictionary containing the node's context
        """
        context = {}
        
        # If dependencies are provided, use them directly
        if dependencies:
            for dep_id in dependencies:
                if dep_id in self.context_store:
                    context.update(self.context_store[dep_id])
        else:
            # Otherwise, use the graph structure to determine dependencies
            for predecessor in self.graph.predecessors(node_id):
                if predecessor in self.context_store:
                    context.update(self.context_store[predecessor])
        
        # Add any node-specific context
        if node_id in self.context_store:
            context.update(self.context_store[node_id])
        
        return context
    
    def get_context_with_optimization(self, node_id: str) -> Dict[str, Any]:
        """Get optimized context for a node using vector similarity.
        
        This method uses the vector store to find the most relevant context
        for the node based on semantic similarity.
        
        Args:
            node_id: ID of the node to get context for
            
        Returns:
            Dictionary containing the optimized context
        """
        # Start with the basic context
        context = self.get_context(node_id)
        
        # If we have a vector store, use it for optimization
        if self.vector_store and self.context_store:
            try:
                # Get node metadata for query
                node_metadata = self._get_node_metadata(node_id)
                if not node_metadata:
                    return context
                
                # Create a query from the node metadata
                query = self._create_query_from_metadata(node_metadata)
                
                # Search for similar contexts
                similar_contexts = self.vector_store.similarity_search(
                    query,
                    k=3  # Limit to top 3 most relevant contexts
                )
                
                # Merge relevant contexts
                for similar_context in similar_contexts:
                    if isinstance(similar_context, dict) and 'metadata' in similar_context:
                        source_node = similar_context['metadata'].get('node_id')
                        if source_node and source_node != node_id:
                            if source_node in self.context_store:
                                # Only add if it doesn't exceed token limit
                                if self._estimate_tokens(context, self.context_store[source_node]) <= self.max_tokens:
                                    context.update(self.context_store[source_node])
                
            except Exception as e:
                logger.warning(f"Error during context optimization: {str(e)}")
                # Fall back to basic context
        
        return context
    
    def set_context(self, node_id: str, context: Dict[str, Any]) -> None:
        """Set context for a node.
        
        Args:
            node_id: ID of the node to set context for
            context: Dictionary containing the context
        """
        self.context_store[node_id] = context
        self.token_counts[node_id] = self._estimate_tokens(context)
        
        # Store in vector store if available
        if self.vector_store:
            try:
                metadata = self._get_node_metadata(node_id)
                if metadata:
                    self.vector_store.add_texts(
                        texts=[json.dumps(context)],
                        metadatas=[{'node_id': node_id, **metadata}]
                    )
            except Exception as e:
                logger.warning(f"Error storing context in vector store: {str(e)}")
    
    def update(self, node_id: str, result: NodeExecutionResult) -> None:
        """Update context with node execution result.
        
        Args:
            node_id: ID of the node that was executed
            result: Execution result containing output
        """
        if result.success and result.output:
            self.set_context(
                node_id,
                {"output": result.output}
            )
    
    def log_error(self, node_id: str, error_result: NodeExecutionResult) -> None:
        """Log an error for a node.
        
        Args:
            node_id: ID of the node that encountered an error
            error_result: Error result containing error details
        """
        if node_id not in self.error_log:
            self.error_log[node_id] = []
        
        self.error_log[node_id].append({
            'timestamp': time.time(),
            'error_type': error_result.metadata.error_type if error_result.metadata else 'Unknown',
            'message': error_result.error,
            'metadata': error_result.metadata.dict() if error_result.metadata else {}
        })
        
        logger.error(f"Error in node {node_id}: {error_result.error}")
    
    def _get_node_metadata(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a node from the graph.
        
        Args:
            node_id: ID of the node to get metadata for
            
        Returns:
            Dictionary containing node metadata or None if not found
        """
        if node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id]
            if 'node' in node_data and hasattr(node_data['node'], 'config'):
                return {
                    'node_type': node_data['node'].__class__.__name__,
                    'node_id': node_id,
                    'dependencies': node_data['node'].config.dependencies if hasattr(node_data['node'].config, 'dependencies') else []
                }
        return None
    
    def _create_query_from_metadata(self, metadata: Dict[str, Any]) -> str:
        """Create a query string from node metadata for vector search.
        
        Args:
            metadata: Node metadata dictionary
            
        Returns:
            Query string for vector search
        """
        return f"Node type: {metadata.get('node_type', 'Unknown')}, ID: {metadata.get('node_id', 'Unknown')}"
    
    def _estimate_tokens(self, context: Dict[str, Any], additional_context: Optional[Dict[str, Any]] = None) -> int:
        """Estimate the number of tokens in a context.
        
        This is a simple estimation based on character count.
        For more accurate token counting, consider using a tokenizer.
        
        Args:
            context: Context dictionary
            additional_context: Optional additional context to include in the estimation
            
        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        context_str = json.dumps(context)
        if additional_context:
            context_str += json.dumps(additional_context)
        
        return len(context_str) // 4