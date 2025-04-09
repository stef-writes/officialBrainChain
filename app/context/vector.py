"""
Vector store for context management and similarity search
"""

from typing import List, Dict, Any, Optional
import numpy as np
from app.utils.logging import logger

class VectorStore:
    """Vector store for context management and similarity search"""
    
    def __init__(
        self,
        index_name: str,
        dimension: int = 1536,
        metric: str = "cosine"
    ):
        """Initialize vector store.
        
        Args:
            index_name: Name of the vector index
            dimension: Dimension of vectors
            metric: Distance metric to use (cosine, euclidean, dot)
        """
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.vectors: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, Dict[str, Any]] = {}
        
        logger.debug(
            f"Initialized VectorStore with index={index_name}, "
            f"dimension={dimension}, metric={metric}"
        )
        
    def add_vector(
        self,
        vector_id: str,
        vector: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add vector to store.
        
        Args:
            vector_id: Unique identifier for vector
            vector: Vector to store
            metadata: Optional metadata to store with vector
        """
        if vector.shape != (self.dimension,):
            raise ValueError(
                f"Vector dimension mismatch. Expected {self.dimension}, "
                f"got {vector.shape[0]}"
            )
            
        self.vectors[vector_id] = vector
        if metadata:
            self.metadata[vector_id] = metadata
            
    def similarity_search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Search for similar vectors.
        
        Args:
            query_vector: Query vector
            k: Number of results to return
            threshold: Similarity threshold
            
        Returns:
            List of similar vectors with metadata
        """
        if not self.vectors:
            return []
            
        if query_vector.shape != (self.dimension,):
            raise ValueError(
                f"Query vector dimension mismatch. Expected {self.dimension}, "
                f"got {query_vector.shape[0]}"
            )
            
        scores = {}
        for vector_id, vector in self.vectors.items():
            if self.metric == "cosine":
                score = np.dot(query_vector, vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(vector)
                )
            elif self.metric == "euclidean":
                score = -np.linalg.norm(query_vector - vector)
            else:  # dot product
                score = np.dot(query_vector, vector)
                
            if score >= threshold:
                scores[vector_id] = score
                
        # Sort by score and get top k
        sorted_scores = sorted(
            scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        results = []
        for vector_id, score in sorted_scores:
            result = {
                "id": vector_id,
                "score": float(score),
                "vector": self.vectors[vector_id].tolist()
            }
            if vector_id in self.metadata:
                result["metadata"] = self.metadata[vector_id]
            results.append(result)
            
        return results
        
    def clear(self) -> None:
        """Clear all vectors and metadata"""
        self.vectors.clear()
        self.metadata.clear()
        logger.debug(f"Cleared vector store {self.index_name}") 