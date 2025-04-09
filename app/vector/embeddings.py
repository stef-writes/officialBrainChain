"""
Embedding manager for vector operations
"""

from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from cachetools import TTLCache, LRUCache
import logging
from app.models.vector_store import EmbeddingConfig
from app.vector.base import BatchProcessor, with_retry

logger = logging.getLogger(__name__)

class EmbeddingManager:
    """Manages text embeddings with caching and batching"""
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize embedding manager.
        
        Args:
            config: Embedding configuration
        """
        self.config = config
        self.model = SentenceTransformer(config.model_name)
        self.batch_processor = BatchProcessor(config.batch_size)
        
        # Cache for raw text to vector mappings
        self.cache = TTLCache(
            maxsize=config.cache_size,
            ttl=3600  # 1 hour TTL
        )
        
        # Cache for preprocessed text
        self.preprocess_cache = LRUCache(maxsize=config.cache_size)
        
        # Ensure model is in evaluation mode
        self.model.eval()
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model.to('cuda')
            
        logger.info(
            f"Initialized EmbeddingManager with model {config.model_name}"
            f" (dimension: {self.dimension})"
        )
    
    @property
    def dimension(self) -> int:
        """Get embedding dimension"""
        return self.model.get_sentence_embedding_dimension()
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for embedding.
        
        Args:
            text: Text to preprocess
            
        Returns:
            Preprocessed text
        """
        if text in self.preprocess_cache:
            return self.preprocess_cache[text]
            
        # Basic preprocessing
        processed = text.strip().lower()
        
        # Truncate if needed
        if self.config.max_length:
            tokens = processed.split()
            if len(tokens) > self.config.max_length:
                processed = " ".join(tokens[:self.config.max_length])
        
        self.preprocess_cache[text] = processed
        return processed
    
    async def get_embeddings(
        self,
        texts: List[str],
        normalize: Optional[bool] = None
    ) -> List[List[float]]:
        """Get embeddings for texts.
        
        Args:
            texts: List of texts to embed
            normalize: Whether to normalize vectors (overrides config)
            
        Returns:
            List of embedding vectors
            
        Raises:
            Exception: If embedding fails
        """
        if normalize is None:
            normalize = self.config.normalize_embeddings
            
        # Preprocess texts
        processed_texts = [self._preprocess_text(text) for text in texts]
        
        # Check cache first
        vectors = []
        texts_to_embed = []
        original_indices = []
        
        for i, text in enumerate(processed_texts):
            if text in self.cache:
                vectors.append(self.cache[text])
            else:
                texts_to_embed.append(text)
                original_indices.append(i)
        
        if texts_to_embed:
            # Define batch operation
            async def embed_batch(batch: List[str]) -> List[List[float]]:
                with torch.no_grad():
                    embeddings = self.model.encode(
                        batch,
                        convert_to_numpy=True,
                        normalize_embeddings=normalize
                    )
                return embeddings.tolist()
            
            # Process in batches
            new_vectors = await self.batch_processor.process(
                texts_to_embed,
                embed_batch
            )
            
            # Update cache
            for text, vector in zip(texts_to_embed, new_vectors):
                self.cache[text] = vector
            
            # Insert new vectors at original positions
            for idx, vector in zip(original_indices, new_vectors):
                vectors.insert(idx, vector)
        
        return vectors
    
    async def get_embedding(
        self,
        text: str,
        normalize: Optional[bool] = None
    ) -> List[float]:
        """Get embedding for single text.
        
        Args:
            text: Text to embed
            normalize: Whether to normalize vector
            
        Returns:
            Embedding vector
        """
        vectors = await self.get_embeddings([text], normalize=normalize)
        return vectors[0]
    
    def clear_cache(self) -> None:
        """Clear embedding caches"""
        self.cache.clear()
        self.preprocess_cache.clear()
        logger.info("Cleared embedding caches") 