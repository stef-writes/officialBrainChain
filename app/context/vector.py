"""
Vector store for semantic context retrieval
"""

from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import json
import logging
from collections import defaultdict
from nltk.stem import PorterStemmer
import math
import numpy as np
from sentence_transformers import SentenceTransformer
from app.models.embeddings import HybridSearchConfig

logger = logging.getLogger(__name__)

class VectorStore:
    """Manages vector embeddings and similarity operations"""
    
    EMBEDDING_VERSION = "v2-hybrid"  # Change when updating embeddings
    
    # Window configuration
    WINDOW_SIZE = 5  # Words per window
    WINDOW_STRIDE = 3  # Words between windows
    
    def __init__(self, storage_path: str = "vector_store.json", config: Optional[HybridSearchConfig] = None):
        """Initialize the vector store.
        
        Args:
            storage_path: Path to store vectors persistently
            config: Configuration for hybrid search
        """
        self.storage_path = Path(storage_path)
        self.vectors: Dict[str, List[float]] = {}
        self.contexts: Dict[str, Dict[str, Any]] = {}
        self.stemmer = PorterStemmer()
        self.doc_freq = defaultdict(int)
        self.total_docs = 0
        self.STOP_WORDS = {
            "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of",
            "with", "by", "from", "up", "about", "into", "over", "after", "is",
            "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "will", "would", "shall", "should", "may",
            "might", "must", "can", "could"
        }
        
        # Initialize configuration
        self.config = config or HybridSearchConfig()
        
        # Initialize sentence transformer for conceptual matching
        self.conceptual_model = SentenceTransformer(self.config.model_name)
        
        self._load()
        logger.info(f"Initialized vector store at {storage_path} with config: {self.config}")

    def _preprocess_text(self, text: str) -> List[str]:
        """Preprocess text by tokenizing, stemming, and removing stop words.
        
        Args:
            text: Text to preprocess
            
        Returns:
            List of preprocessed tokens
        """
        # Convert to lowercase and split into tokens
        tokens = text.lower().split()
        
        # Remove stop words and short tokens, then stem
        return [
            self.stemmer.stem(t) for t in tokens
            if t not in self.STOP_WORDS and len(t) > 2
        ]

    def _compute_tf_idf(self, terms: List[str]) -> Dict[str, float]:
        """Compute TF-IDF scores for terms.
        
        Args:
            terms: List of preprocessed terms
            
        Returns:
            Dictionary of term to TF-IDF score
        """
        # Count term frequencies
        term_freq = defaultdict(int)
        for term in terms:
            term_freq[term] += 1
            
        # Compute TF-IDF scores
        scores = {}
        for term, freq in term_freq.items():
            tf = freq / len(terms)
            # Ensure total_docs is non-zero and handle edge cases
            if self.total_docs > 0:
                idf = math.log((self.total_docs + 1) / (self.doc_freq[term] + 1))
                scores[term] = tf * idf
            else:
                scores[term] = 0  # Default to zero if total_docs is zero
            
        return scores

    def _load(self) -> None:
        """Load vectors and contexts from storage"""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                    self.vectors = data.get('vectors', {})
                    self.contexts = data.get('contexts', {})
                    self.doc_freq = defaultdict(int, data.get('doc_freq', {}))
                    self.total_docs = data.get('total_docs', 0)
                logger.debug(f"Loaded {len(self.vectors)} vectors from {self.storage_path}")
            except Exception as e:
                logger.error(f"Error loading vectors: {e}")
                self.vectors = {}
                self.contexts = {}

    def _save(self) -> None:
        """Save vectors and contexts to storage"""
        try:
            with open(self.storage_path, 'w') as f:
                json.dump({
                    'vectors': self.vectors,
                    'contexts': self.contexts,
                    'doc_freq': dict(self.doc_freq),
                    'total_docs': self.total_docs
                }, f)
            logger.debug(f"Saved {len(self.vectors)} vectors to {self.storage_path}")
        except Exception as e:
            logger.error(f"Error saving vectors: {e}")

    def _store_window(self, text: str, metadata: Dict[str, Any], window_position: int) -> str:
        """Store a single window of text with its embedding.
        
        Args:
            text: Window text to store
            metadata: Associated metadata
            window_position: Position of the window in the original text
            
        Returns:
            ID of the stored window
        """
        # Preprocess text and compute TF-IDF
        terms = self._preprocess_text(text)
        tf_idf_scores = self._compute_tf_idf(terms)
        
        # Update document frequencies
        for term in set(terms):
            self.doc_freq[term] += 1
        self.total_docs += 1
        
        # Create vector from TF-IDF scores
        window_id = f"{metadata.get('node_id', 'unknown')}_window_{window_position}"
        self.vectors[window_id] = list(tf_idf_scores.values())
        
        # Compute conceptual embedding
        conceptual_embedding = self.conceptual_model.encode(
            text, 
            batch_size=self.config.batch_size
        )
        
        # Store context with window information
        self.contexts[window_id] = {
            'text': text,
            'metadata': metadata,
            'conceptual_embedding': conceptual_embedding.tolist(),
            'is_window': True,
            'window_position': window_position,
            'parent_node_id': metadata.get('node_id')
        }
        
        return window_id

    def add_context(self, text: str, metadata: Dict[str, Any]) -> None:
        """Add a new context with its embedding, using window-based approach.
        
        Args:
            text: Context text to add
            metadata: Associated metadata
        """
        # Store the full context first
        vector_id = metadata.get('node_id', str(len(self.vectors)))
        
        # Preprocess text and compute TF-IDF for full context
        terms = self._preprocess_text(text)
        tf_idf_scores = self._compute_tf_idf(terms)
        
        # Update document frequencies
        for term in set(terms):
            self.doc_freq[term] += 1
        self.total_docs += 1
        
        # Create vector from TF-IDF scores
        self.vectors[vector_id] = list(tf_idf_scores.values())
        
        # Compute conceptual embedding for full context
        conceptual_embedding = self.conceptual_model.encode(
            text, 
            batch_size=self.config.batch_size
        )
        
        # Store full context
        self.contexts[vector_id] = {
            'text': text,
            'metadata': metadata,
            'conceptual_embedding': conceptual_embedding.tolist(),
            'is_window': False
        }
        
        # Split text into windows and store each window
        tokens = text.split()
        window_ids = []
        
        for i in range(0, len(tokens), self.WINDOW_STRIDE):
            window_text = ' '.join(tokens[i:i+self.WINDOW_SIZE])
            if window_text.strip():  # Only store non-empty windows
                window_id = self._store_window(window_text, metadata, len(window_ids) + 1)
                window_ids.append(window_id)
        
        # Add window references to the full context
        self.contexts[vector_id]['window_ids'] = window_ids
        
        # Save to storage
        self._save()
        logger.debug(f"Added context for node {metadata.get('node_id')} with {len(window_ids)} windows")

    def _find_lexical_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar contexts using lexical (TF-IDF) matching.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of similar contexts with metadata
        """
        if not self.vectors:
            return []
            
        # Preprocess query and compute TF-IDF
        query_terms = self._preprocess_text(query)
        query_scores = self._compute_tf_idf(query_terms)
        
        # Compute similarities
        similarities = []
        query_vector = np.array(list(query_scores.values()))
        
        for vector_id, vector in self.vectors.items():
            vector_array = np.array(vector)
            # Compute cosine similarity
            similarity = np.dot(query_vector, vector_array) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector_array)
            )
            
            # Apply minimum similarity threshold
            if similarity >= self.config.min_similarity:
                similarities.append({
                    'node_id': vector_id,
                    'text': self.contexts[vector_id]['text'],
                    'metadata': self.contexts[vector_id]['metadata'],
                    'similarity': float(similarity),
                    'is_window': self.contexts[vector_id].get('is_window', False),
                    'window_position': self.contexts[vector_id].get('window_position')
                })
            
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

    def _find_conceptual_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar contexts using conceptual (sentence embedding) matching.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of similar contexts with metadata
        """
        if not self.contexts:
            return []
            
        # Encode query
        query_embedding = self.conceptual_model.encode(
            query, 
            batch_size=self.config.batch_size
        )
        
        # Compute similarities
        similarities = []
        for vector_id, context in self.contexts.items():
            context_embedding = np.array(context['conceptual_embedding'])
            similarity = np.dot(query_embedding, context_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(context_embedding)
            )
            
            # Apply minimum similarity threshold
            if similarity >= self.config.min_similarity:
                similarities.append({
                    'node_id': vector_id,
                    'text': context['text'],
                    'metadata': context['metadata'],
                    'similarity': float(similarity),
                    'is_window': context.get('is_window', False),
                    'window_position': context.get('window_position')
                })
            
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:top_k]

    def find_similar(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        """Find similar contexts using hybrid lexical and conceptual matching.
        
        Args:
            query: Query text
            top_k: Number of results to return (overrides config.max_results)
            
        Returns:
            List of similar contexts with metadata
        """
        # Use config value if top_k not specified
        top_k = top_k or self.config.max_results
        
        # Get lexical and conceptual matches
        lexical_results = self._find_lexical_similar(query, top_k * 2)
        conceptual_results = self._find_conceptual_similar(query, top_k * 2)
        
        # Create dictionaries for easy lookup
        lexical_scores = {r['node_id']: r['similarity'] for r in lexical_results}
        conceptual_scores = {r['node_id']: r['similarity'] for r in conceptual_results}
        
        # Combine scores
        combined_scores = {}
        for node_id in set(lexical_scores.keys()) | set(conceptual_scores.keys()):
            lex_score = lexical_scores.get(node_id, 0)
            con_score = conceptual_scores.get(node_id, 0)
            combined_scores[node_id] = (
                self.config.alpha * lex_score + 
                (1 - self.config.alpha) * con_score
            )
        
        # Ensure vector alignment
        for node_id in combined_scores.keys():
            if len(self.vectors[node_id]) != len(conceptual_results[0]['conceptual_embedding']):
                logger.warning(f"Vector dimension mismatch for node {node_id}")
                continue
        
        # Get top k results
        top_node_ids = sorted(combined_scores.keys(), 
                            key=lambda x: combined_scores[x], 
                            reverse=True)[:top_k]
        
        # Format results
        results = []
        for node_id in top_node_ids:
            context = self.contexts[node_id]
            results.append({
                'node_id': node_id,
                'text': context['text'],
                'metadata': context['metadata'],
                'similarity': combined_scores[node_id],
                'lexical_similarity': lexical_scores.get(node_id, 0),
                'conceptual_similarity': conceptual_scores.get(node_id, 0),
                'is_window': context.get('is_window', False),
                'window_position': context.get('window_position'),
                'parent_node_id': context.get('parent_node_id')
            })
            
        return results 