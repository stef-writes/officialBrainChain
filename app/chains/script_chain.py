"""
Main workflow orchestration chain
"""

from datetime import datetime
import networkx as nx
from typing import Dict, List, Optional, Any
from app.utils.context import ContextManager
from app.utils.retry import AsyncRetry
from app.nodes.base import BaseNode
from app.models.nodes import NodeExecutionResult, NodeMetadata
import logging

logger = logging.getLogger(__name__)

class ScriptChain:
    """Orchestrates execution of node workflows
    
    Attributes:
        graph: NetworkX graph representing workflow
        context: State management across nodes
        retry: Retry configuration for API calls
    """
    
    def __init__(self, retry_config: Optional[Dict[str, Any]] = None):
        """Initialize the script chain.
        
        Args:
            retry_config: Configuration for retry behavior. Defaults to None.
        """
        self.graph = nx.DiGraph()
        self.context = ContextManager()
        self.nodes: Dict[str, BaseNode] = {}
        self.retry = AsyncRetry(**(retry_config or {})) if retry_config else None
        
    def add_node(self, node: BaseNode) -> None:
        """Add a node to the chain.
        
        Args:
            node: The node to add
        """
        if node.node_id not in self.nodes:
            self.nodes[node.node_id] = node
            self.graph.add_node(node.node_id)
        
    def add_edge(self, from_node_id: str, to_node_id: str) -> None:
        """Add a dependency edge between nodes.
        
        Args:
            from_node_id: ID of the source node
            to_node_id: ID of the target node
            
        Raises:
            ValueError: If adding the edge would create a cycle
        """
        # Verify both nodes exist
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            raise ValueError("Both nodes must exist in the chain")
        
        # Check for self-loops
        if from_node_id == to_node_id:
            raise ValueError("Cannot add edge from node to itself")
        
        # Add edge and check for cycles
        self.graph.add_edge(from_node_id, to_node_id)
        try:
            cycles = list(nx.simple_cycles(self.graph))
            if cycles:
                self.graph.remove_edge(from_node_id, to_node_id)
                raise ValueError(f"Adding edge would create cycles: {cycles}")
        except nx.NetworkXNoCycle:
            pass  # No cycles found, edge is valid
        
    async def execute(self) -> NodeExecutionResult:
        """Execute the chain of nodes in dependency order.
        
        Returns:
            The combined execution result
        """
        start_time = datetime.utcnow()
        results: Dict[str, Any] = {}
        
        try:
            # Get execution order from graph
            execution_order = list(nx.topological_sort(self.graph))
            
            # Execute nodes in order
            for node_id in execution_order:
                node = self.nodes[node_id]
                
                # Get inputs from dependencies and node's context
                context = self.context.get_context(node_id) or {}
                for pred in self.graph.predecessors(node_id):
                    if pred in results and results[pred].success:
                        context.update(results[pred].output)
                
                # Execute node
                if self.retry:
                    result = await self.retry.run(node.execute, context)
                else:
                    result = await node.execute(context)
                
                results[node_id] = result
                
                # Stop execution if node failed
                if not result.success:
                    break
            
            # Create combined result
            success = all(r.success for r in results.values())
            return NodeExecutionResult(
                success=success,
                output=str(results) if results else None,
                error=None if success else next(r.error for r in results.values() if not r.success),
                metadata=NodeMetadata(
                    node_id="script_chain",
                    node_type="chain",
                    version="1.0.0",
                    description="Script chain execution result",
                    error_type=None if success else "ExecutionError"
                ),
                duration=(datetime.utcnow() - start_time).total_seconds(),
                timestamp=datetime.utcnow()
            )
            
        except Exception as e:
            logger.error(f"Error executing script chain: {str(e)}")
            return NodeExecutionResult(
                success=False,
                output=None,
                error=str(e),
                metadata=NodeMetadata(
                    node_id="script_chain",
                    node_type="chain",
                    version="1.0.0",
                    description="Script chain execution result",
                    error_type=e.__class__.__name__
                ),
                duration=(datetime.utcnow() - start_time).total_seconds(),
                timestamp=datetime.utcnow()
            )
    
    def _collect_inputs(self, node_id: str) -> Dict:
        """Gather inputs from connected nodes"""
        inputs = {}
        for predecessor in self.graph.predecessors(node_id):
            inputs.update(self.context.get_context(predecessor))
        return inputs
    
    def _update_context(self, node_id: str, result: Dict):
        """Updated to handle new metadata format"""
        self.context.set(
            node_id,
            result["output"]["content"],  # Changed from result["content"]
            metadata={
                "usage": result["metadata"].get("usage", {}),
                "templates": result["metadata"].get("templates", []),
                "model": result["metadata"].get("model"),
                "timestamp": datetime.now().isoformat()
            }
        )
    
    def _handle_error(self, node_id: str, error: Exception):
        """Enhanced error logging"""
        self.context.log_error(
            node_id=node_id,
            error_type=error.__class__.__name__,
            message=str(error),
            metadata={
                "node_version": self.graph.nodes[node_id]["node"].config.metadata.version,
                "retry_count": self.retry.current_retry
            }
        )