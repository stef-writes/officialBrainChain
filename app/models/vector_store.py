"""
Configuration models for vector store operations
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum

class SimilarityMetric(str, Enum):
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"

class VectorStoreConfig(BaseModel):
    """Configuration for vector store operations"""
    index_name: str = Field(..., description="Name of the vector index")
    dimension: int = Field(384, description="Must match embedding model dimension")
    environment: str = Field(..., description="Pinecone environment")
    metric: SimilarityMetric = Field(SimilarityMetric.COSINE, description="Similarity metric")
    batch_size: int = Field(100, description="Batch size for operations")
    pod_type: str = Field("p1", description="Pinecone pod type")
    replicas: int = Field(1, description="Number of replicas")
    metadata_config: Dict[str, str] = Field(
        default_factory=dict,
        description="Metadata field configuration"
    )

class EmbeddingConfig(BaseModel):
    """Configuration for embedding operations"""
    model_name: str = Field(
        "all-MiniLM-L6-v2",
        description="Name of the sentence transformer model"
    )
    batch_size: int = Field(32, description="Batch size for embedding operations")
    cache_size: int = Field(1000, description="Size of embedding cache")
    max_length: Optional[int] = Field(512, description="Maximum sequence length")
    normalize_embeddings: bool = Field(True, description="Whether to normalize embeddings")

class VectorSearchResult(BaseModel):
    """Result from vector search operation"""
    id: str
    score: float
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None

class BatchOperation(BaseModel):
    """Batch operation for vector store"""
    vectors: List[List[float]]
    ids: List[str]
    metadata: List[Dict[str, Any]] 