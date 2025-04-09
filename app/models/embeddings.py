"""
Configuration models for embedding and search operations
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class HybridSearchConfig(BaseModel):
    """Configuration for hybrid semantic search.
    
    Attributes:
        alpha: Weight for lexical matching (1-alpha for conceptual matching)
        model_name: Name of the sentence transformer model to use
        batch_size: Batch size for encoding operations
        min_similarity: Minimum similarity score to consider a match
        max_results: Maximum number of results to return
    """
    alpha: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for lexical matching (1-alpha for conceptual matching)"
    )
    model_name: str = Field(
        default="all-MiniLM-L6-v2",
        description="Name of the sentence transformer model to use"
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        description="Batch size for encoding operations"
    )
    min_similarity: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score to consider a match"
    )
    max_results: int = Field(
        default=5,
        ge=1,
        description="Maximum number of results to return"
    )
    
    class Config:
        """Pydantic model configuration"""
        json_schema_extra = {
            "example": {
                "alpha": 0.7,
                "model_name": "all-MiniLM-L6-v2",
                "batch_size": 32,
                "min_similarity": 0.7,
                "max_results": 5
            }
        } 