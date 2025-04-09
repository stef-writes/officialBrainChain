"""
Callback handlers for ScriptChain execution events.

This module provides a set of callback handlers for monitoring and tracking ScriptChain execution:

1. ScriptChainCallback: Abstract base class defining the callback interface
2. LoggingCallback: Basic logging for production use
3. MetricsCallback: Performance and usage metrics collection

For debugging purposes, see debug_callback.py which provides the DebugCallback class
with more detailed event tracking and analysis capabilities.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import logging
import json
import time
from datetime import datetime
from app.models.node_models import NodeConfig, NodeExecutionResult, UsageMetadata

logger = logging.getLogger(__name__)

class ScriptChainCallback(ABC):
    """Abstract base class for ScriptChain callback handlers.
    
    This class defines the interface for callback handlers that can be registered
    with a ScriptChain to receive notifications about execution events.
    
    For basic logging, use LoggingCallback.
    For metrics collection, use MetricsCallback.
    For detailed debugging, use DebugCallback from debug_callback.py.
    """
    
    @abstractmethod
    async def on_chain_start(self, chain_id: str, data: Dict[str, Any]) -> None:
        """Called when a chain execution starts."""
        pass
    
    @abstractmethod
    async def on_chain_end(self, chain_id: str, data: Dict[str, Any]) -> None:
        """Called when a chain execution ends."""
        pass
    
    @abstractmethod
    async def on_node_start(self, chain_id: str, data: Dict[str, Any]) -> None:
        """Called when a node execution starts."""
        pass
    
    @abstractmethod
    async def on_node_end(self, chain_id: str, data: Dict[str, Any]) -> None:
        """Called when a node execution ends successfully."""
        pass
    
    @abstractmethod
    async def on_node_error(self, chain_id: str, data: Dict[str, Any]) -> None:
        """Called when a node execution fails."""
        pass
    
    @abstractmethod
    async def on_context_update(self, node_id: str, context: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        """Called when a node's context is updated."""
        pass
    
    @abstractmethod
    async def on_vector_store_op(self,
        operation: str,  # "store" or "retrieve"
        node_id: str,
        context_snippet: str,
        similarity_score: Optional[float] = None
    ) -> None:
        """Called when a vector store operation occurs."""
        pass

class LoggingCallback(ScriptChainCallback):
    """Callback handler that logs chain and node events.
    
    This is the primary callback for production use, providing essential
    logging of workflow execution events at configurable log levels.
    """
    
    def __init__(self, log_level: int = logging.INFO):
        self.logger = logging.getLogger(__name__)
        self.log_level = log_level
    
    async def on_chain_start(self, chain_id: str, data: Dict[str, Any]) -> None:
        self.logger.log(
            self.log_level,
            f"Chain {chain_id} started with {data.get('total_nodes', 0)} nodes"
        )
    
    async def on_chain_end(self, chain_id: str, data: Dict[str, Any]) -> None:
        success = data.get('success', False)
        duration = data.get('duration', 0)
        
        if success:
            self.logger.log(
                self.log_level,
                f"Chain {chain_id} completed successfully in {duration:.2f} seconds"
            )
        else:
            error = data.get('error', 'Unknown error')
            self.logger.log(
                logging.ERROR,
                f"Chain {chain_id} failed after {duration:.2f} seconds: {error}"
            )
    
    async def on_node_start(self, chain_id: str, data: Dict[str, Any]) -> None:
        node_id = data.get('node_id', 'unknown')
        self.logger.log(
            self.log_level,
            f"Node {node_id} in chain {chain_id} started execution"
        )
    
    async def on_node_end(self, chain_id: str, data: Dict[str, Any]) -> None:
        node_id = data.get('node_id', 'unknown')
        level = data.get('level', 'unknown')
        self.logger.log(
            self.log_level,
            f"Node {node_id} in chain {chain_id} (level {level}) completed successfully"
        )
    
    async def on_node_error(self, chain_id: str, data: Dict[str, Any]) -> None:
        node_id = data.get('node_id', 'unknown')
        error = data.get('error', {}).get('error', 'Unknown error')
        level = data.get('level', 'unknown')
        self.logger.log(
            logging.ERROR,
            f"Node {node_id} in chain {chain_id} (level {level}) failed: {error}"
        )

    async def on_context_update(self, node_id: str, context: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        self.logger.log(
            self.log_level,
            f"Context updated for node {node_id} with metadata: {metadata}"
        )

    async def on_vector_store_op(self,
        operation: str,
        node_id: str,
        context_snippet: str,
        similarity_score: Optional[float] = None
    ) -> None:
        msg = f"Vector store {operation} for node {node_id}"
        if similarity_score is not None:
            msg += f" with similarity score {similarity_score:.3f}"
        self.logger.log(self.log_level, msg)

class MetricsCallback(ScriptChainCallback):
    """Callback handler that collects execution metrics.
    
    This callback focuses on collecting performance metrics, timing data,
    and usage statistics for analysis and optimization.
    """
    
    def __init__(self):
        self.metrics = {
            'chains': {},
            'nodes': {},
            'vector_ops': [],
            'context_updates': []
        }
    
    async def on_chain_start(self, chain_id: str, data: Dict[str, Any]) -> None:
        self.metrics['chains'][chain_id] = {
            'start_time': time.time(),
            'node_count': data.get('total_nodes', 0),
            'execution_levels': len(data.get('execution_levels', [])),
            'nodes': {}
        }
    
    async def on_chain_end(self, chain_id: str, data: Dict[str, Any]) -> None:
        if chain_id in self.metrics['chains']:
            chain_metrics = self.metrics['chains'][chain_id]
            chain_metrics['end_time'] = time.time()
            chain_metrics['duration'] = chain_metrics['end_time'] - chain_metrics['start_time']
            chain_metrics['success'] = data.get('success', False)
            
            if not chain_metrics['success']:
                chain_metrics['error'] = data.get('error', 'Unknown error')
    
    async def on_node_start(self, chain_id: str, data: Dict[str, Any]) -> None:
        node_id = data.get('node_id', 'unknown')
        if chain_id in self.metrics['chains']:
            self.metrics['chains'][chain_id]['nodes'][node_id] = {
                'start_time': time.time(),
                'level': data.get('level', 'unknown')
            }
    
    async def on_node_end(self, chain_id: str, data: Dict[str, Any]) -> None:
        node_id = data.get('node_id', 'unknown')
        if chain_id in self.metrics['chains'] and node_id in self.metrics['chains'][chain_id]['nodes']:
            node_metrics = self.metrics['chains'][chain_id]['nodes'][node_id]
            node_metrics['end_time'] = time.time()
            node_metrics['duration'] = node_metrics['end_time'] - node_metrics['start_time']
            node_metrics['success'] = True
            
            if 'result' in data and 'usage' in data['result']:
                node_metrics['usage'] = data['result']['usage']
    
    async def on_node_error(self, chain_id: str, data: Dict[str, Any]) -> None:
        node_id = data.get('node_id', 'unknown')
        if chain_id in self.metrics['chains'] and node_id in self.metrics['chains'][chain_id]['nodes']:
            node_metrics = self.metrics['chains'][chain_id]['nodes'][node_id]
            node_metrics['end_time'] = time.time()
            node_metrics['duration'] = node_metrics['end_time'] - node_metrics['start_time']
            node_metrics['success'] = False
            node_metrics['error'] = data.get('error', {}).get('error', 'Unknown error')

    async def on_context_update(self, node_id: str, context: Dict[str, Any], metadata: Dict[str, Any]) -> None:
        self.metrics['context_updates'].append({
            'node_id': node_id,
            'timestamp': datetime.utcnow().isoformat(),
            'context_size': len(str(context)),
            'metadata': metadata
        })

    async def on_vector_store_op(self,
        operation: str,
        node_id: str,
        context_snippet: str,
        similarity_score: Optional[float] = None
    ) -> None:
        self.metrics['vector_ops'].append({
            'operation': operation,
            'node_id': node_id,
            'timestamp': datetime.utcnow().isoformat(),
            'similarity_score': similarity_score
        })
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get the collected metrics."""
        return self.metrics
    
    def export_metrics(self, filepath: str) -> None:
        """Export metrics to a JSON file."""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.metrics, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}") 