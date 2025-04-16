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
    """Configuration for vector store"""
    index_name: str = Field(..., description="Name of the vector store index")
    environment: str = Field(..., description="Environment name")
    dimension: int = Field(..., description="Dimension of vectors")
    metric: SimilarityMetric = Field(..., description="Distance metric for similarity search")
    pod_type: str = Field(..., description="Type of pod (specific to some providers like Pinecone)")
    replicas: int = Field(..., description="Number of replicas (specific to some providers)")
    api_key: str = Field(..., description="API key")
    host: Optional[str] = Field(None, description="Host URL for the vector store")
    metadata_config: Dict[str, str] = Field(
        default_factory=dict,
        description="Metadata field configuration (specific to some providers)"
    )
    batch_size: int = Field(100, description="Batch size for operations")

class VectorSearchResult(BaseModel):
    """Result from vector search operation"""
    id: str
    score: float
    metadata: Dict[str, Any]
    vector: Optional[List[float]] = None 