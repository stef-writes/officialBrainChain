"""
Main workflow orchestration chain for LLM nodes
"""

from datetime import datetime
import networkx as nx
from typing import Dict, List, Optional, Any
from app.models.node_models import NodeConfig, NodeExecutionResult, NodeMetadata, UsageMetadata
from app.models.config import LLMConfig, MessageTemplate
from app.nodes.text_generation import TextGenerationNode
from app.utils.context import ContextManager
from app.utils.retry import AsyncRetry
from app.nodes.base import BaseNode
import logging

logger = logging.getLogger(__name__)

class ScriptChain:
    """Orchestrates execution of LLM node workflows
    
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
            
            # Add edges for dependencies
            for dep_id in node.config.dependencies:
                if dep_id in self.nodes:
                    self.graph.add_edge(dep_id, node.node_id)
                else:
                    logger.warning(f"Dependency {dep_id} not found for node {node.node_id}")
                    
    def add_edge(self, from_node_id: str, to_node_id: str) -> None:
        """Add a dependency edge between nodes.
        
        Args:
            from_node_id: ID of the source node
            to_node_id: ID of the target node
        """
        if from_node_id not in self.nodes:
            raise ValueError(f"Source node {from_node_id} not found in chain")
        if to_node_id not in self.nodes:
            raise ValueError(f"Target node {to_node_id} not found in chain")
            
        self.graph.add_edge(from_node_id, to_node_id)
        # Update dependencies in target node's config
        if from_node_id not in self.nodes[to_node_id].config.dependencies:
            self.nodes[to_node_id].config.dependencies.append(from_node_id)
    
    async def execute(self) -> NodeExecutionResult:
        """Execute the chain of nodes in dependency order.
        
        Returns:
            The combined execution result
        """
        start_time = datetime.utcnow()
        results: Dict[str, NodeExecutionResult] = {}
        total_usage = UsageMetadata()
        
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
                        pred_output = results[pred].output
                        if pred_output is not None:
                            context.update({"output": pred_output})
                
                # Execute node
                if self.retry:
                    result = await self.retry.execute(node.execute, context)
                else:
                    result = await node.execute(context)
                
                results[node_id] = result
                
                # Update context with node's output
                if result.success and result.output:
                    self._update_context(node_id, result)
                
                # Aggregate usage metadata
                if result.usage:
                    total_usage.prompt_tokens = (total_usage.prompt_tokens or 0) + (result.usage.prompt_tokens or 0)
                    total_usage.completion_tokens = (total_usage.completion_tokens or 0) + (result.usage.completion_tokens or 0)
                    total_usage.total_tokens = (total_usage.total_tokens or 0) + (result.usage.total_tokens or 0)
                    total_usage.api_calls = (total_usage.api_calls or 0) + (result.usage.api_calls or 0)
                
                # Stop execution if node failed
                if not result.success:
                    break
            
            # Create combined result
            success = all(r.success for r in results.values())
            failed_result = next((r for r in results.values() if not r.success), None) if not success else None
            
            return NodeExecutionResult(
                success=success,
                output=str(results) if results else None,
                error=None if success else failed_result.error,
                metadata=NodeMetadata(
                    node_id="script_chain",
                    node_type="chain",
                    version="1.0.0",
                    description="Script chain execution result",
                    error_type=None if success else failed_result.metadata.error_type
                ),
                duration=(datetime.utcnow() - start_time).total_seconds(),
                timestamp=datetime.utcnow(),
                usage=total_usage if total_usage.total_tokens > 0 else None
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
    
    def _update_context(self, node_id: str, result: NodeExecutionResult):
        """Update context with node execution result"""
        if result.success and result.output:
            self.context.set_context(
                node_id,
                {"output": result.output}
            )
    
    def _handle_error(self, node_id: str, error: Exception):
        """Enhanced error logging"""
        self.context.log_error(
            node_id=node_id,
            error_type=error.__class__.__name__,
            message=str(error),
            metadata={
                "node_version": self.nodes[node_id].config.metadata.version,
                "retry_count": self.retry.current_retry if self.retry else 0
            }
        )