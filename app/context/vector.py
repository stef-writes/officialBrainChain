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
from dotenv import load_dotenv
import os
import pinecone

logger = logging.getLogger(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pinecone.init(api_key=PINECONE_API_KEY, environment='us-west1-gcp')  # Adjust environment as needed

class VectorStore:
    """Manages vector embeddings and similarity operations"""
    
    EMBEDDING_VERSION = "v2-hybrid"  # Change when updating embeddings
    
    # Window configuration
    WINDOW_SIZE = 5  # Words per window
    WINDOW_STRIDE = 3  # Words between windows
    
    def __init__(self, index_name: str = 'your-index-name'):
        """Initialize the vector store.
        
        Args:
            index_name: Name of the Pinecone index
        """
        self.index_name = index_name
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(self.index_name, dimension=128)  # Adjust dimension as needed
        self.index = pinecone.Index(self.index_name)
        self.model = SentenceTransformer('all-MiniLM-L6-v2')  # Example model
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
        
        self._load()
        logger.info(f"Initialized vector store with index: {self.index_name}")

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
        """Load vectors and contexts from Pinecone"""
        try:
            vectors = self.index.fetch(ids=self.index.list_ids())
            self.vectors = {f"vector_{i}": vector for i, vector in enumerate(vectors['vectors'])}
            self.contexts = {f"vector_{i}": {'text': f"vector_{i}", 'metadata': {'node_id': f"vector_{i}"}} for i in range(len(self.vectors))}
            self.doc_freq = defaultdict(int, {term: freq for vector in self.vectors.values() for term, freq in self._compute_tf_idf(self._preprocess_text(f"vector_{i}"))})
            self.total_docs = len(self.vectors)
            logger.debug(f"Loaded {len(self.vectors)} vectors from Pinecone")
        except Exception as e:
            logger.error(f"Error loading vectors: {e}")
            self.vectors = {}
            self.contexts = {}

    def _save(self) -> None:
        """Save vectors and contexts to Pinecone"""
        try:
            vectors = {f"vector_{i}": vector for i, vector in enumerate(self.vectors.values())}
            self.index.upsert(vectors)
            logger.debug(f"Saved {len(self.vectors)} vectors to Pinecone")
        except Exception as e:
            logger.error(f"Error saving vectors: {e}")

    def _text_to_vector(self, text: str) -> List[float]:
        return self.model.encode(text).tolist()

    def add_context(self, text: str, metadata: Dict[str, Any]) -> None:
        """Add a new context with its embedding, using window-based approach.
        
        Args:
            text: Context text to add
            metadata: Associated metadata
        """
        vector = self._text_to_vector(text)
        vector_id = metadata.get('node_id', str(len(self.vectors)))
        self.index.upsert([(vector_id, vector)])
        
        # Split text into windows and store each window
        tokens = text.split()
        window_ids = []
        
        for i in range(0, len(tokens), self.WINDOW_STRIDE):
            window_text = ' '.join(tokens[i:i+self.WINDOW_SIZE])
            if window_text.strip():  # Only store non-empty windows
                window_id = f"vector_{len(self.vectors)}"
                self.vectors[window_id] = vector
                self.contexts[window_id] = {
                    'text': window_text,
                    'metadata': metadata,
                    'is_window': True,
                    'window_position': len(window_ids) + 1
                }
                window_ids.append(window_id)
        
        # Add window references to the full context
        self.contexts[vector_id]['window_ids'] = window_ids
        
        # Save to Pinecone
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
            if similarity >= 0.7:  # Assuming a default similarity threshold
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
        query_embedding = self._text_to_vector(query)
        
        # Compute similarities
        similarities = []
        for vector_id, context in self.contexts.items():
            context_embedding = np.array(context['conceptual_embedding'])
            similarity = np.dot(query_embedding, context_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(context_embedding)
            )
            
            # Apply minimum similarity threshold
            if similarity >= 0.7:  # Assuming a default similarity threshold
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

    def find_similar(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find similar contexts using hybrid lexical and conceptual matching.
        
        Args:
            query: Query text
            top_k: Number of results to return
            
        Returns:
            List of similar contexts with metadata
        """
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
                lex_score + 
                con_score
            )
        
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