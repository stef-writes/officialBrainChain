"""
Main workflow orchestration chain for LLM nodes
"""

from datetime import datetime
import networkx as nx
from typing import Dict, List, Optional, Any, Union
from app.models.node_models import NodeConfig, NodeExecutionResult, NodeMetadata, UsageMetadata
from app.models.config import LLMConfig, MessageTemplate
from app.nodes.text_generation import TextGenerationNode
from app.utils.context import ContextManager
from app.utils.retry import AsyncRetry
from app.nodes.base import BaseNode
from app.utils.callbacks import ScriptChainCallback
import logging

logger = logging.getLogger(__name__)

class ScriptChain:
    """Orchestrates execution of LLM node workflows
    
    Attributes:
        graph: NetworkX graph representing workflow
        context: State management across nodes
        retry: Retry configuration for API calls
        callbacks: List of callback handlers
    """
    
    def __init__(
        self,
        max_context_tokens: int = 4000,
        callbacks: Optional[List[ScriptChainCallback]] = None
    ):
        """Initialize the script chain.
        
        Args:
            max_context_tokens: Maximum number of tokens allowed in the context
            callbacks: Optional list of callback handlers
        """
        self.graph = nx.DiGraph()
        self.context = ContextManager(max_context_tokens)
        self.logger = logging.getLogger(__name__)
        self.nodes: Dict[str, BaseNode] = {}
        self.retry = None
        self.callbacks = callbacks or []
        
        # Override the _get_parent_nodes method in the context manager
        self.context._get_parent_nodes = self._get_parent_nodes
        
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
        """Execute the workflow chain.
        
        Returns:
            NodeExecutionResult with aggregated results
        """
        # Initialize results and usage tracking
        results = {}
        total_usage = UsageMetadata(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            api_calls=0
        )
        
        try:
            # Get execution order from graph
            execution_order = list(nx.topological_sort(self.graph))
            
            # Notify chain start
            chain_id = f"chain_{datetime.utcnow().timestamp()}"
            chain_config = {
                "node_count": len(self.nodes),
                "execution_order": execution_order
            }
            for callback in self.callbacks:
                await callback.on_chain_start(chain_id, chain_config)
            
            # Execute nodes in order
            for node_id in execution_order:
                node = self.nodes[node_id]
                
                # Notify node start
                for callback in self.callbacks:
                    await callback.on_node_start(node_id, node.config)
                
                # Get inputs from dependencies and node's context
                context = self.context.get_context_with_optimization(node_id)
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
                    self.context.set_context(node_id, {"output": result.output})
                    # Notify context update
                    for callback in self.callbacks:
                        await callback.on_context_update(
                            node_id,
                            {"output": result.output},
                            {"timestamp": datetime.utcnow().isoformat()}
                        )
                
                # Notify node completion
                for callback in self.callbacks:
                    await callback.on_node_complete(
                        node_id,
                        result,
                        result.usage
                    )
                
                # Stop execution if node failed
                if not result.success:
                    # Notify node error
                    for callback in self.callbacks:
                        await callback.on_node_error(
                            node_id,
                            Exception(result.error) if result.error else Exception("Unknown error"),
                            context
                        )
                    break
            
            # Create combined result
            success = all(r.success for r in results.values())
            
            # Get the final node's output
            final_node_id = execution_order[-1]
            final_result = results[final_node_id]
            
            # Get usage stats from context manager
            usage_stats = self.context.get_usage_stats()
            
            # Aggregate usage from all nodes
            for node_id, usage in usage_stats.items():
                if usage:
                    total_usage.prompt_tokens = (total_usage.prompt_tokens or 0) + (usage.prompt_tokens or 0)
                    total_usage.completion_tokens = (total_usage.completion_tokens or 0) + (usage.completion_tokens or 0)
                    total_usage.total_tokens = (total_usage.total_tokens or 0) + (usage.total_tokens or 0)
                    total_usage.api_calls = (total_usage.api_calls or 0) + (usage.api_calls or 0)
            
            # Create the final result
            result = NodeExecutionResult(
                success=success,
                output=final_result.output if success else None,
                error=final_result.error if not success else None,
                metadata=NodeMetadata(
                    node_id="chain",
                    node_type="chain",
                    version="1.0.0",
                    description="Workflow chain execution",
                    error_type=final_result.metadata.error_type if not success else None,
                    timestamp=datetime.utcnow()
                ),
                duration=sum(r.duration for r in results.values()),
                timestamp=datetime.utcnow(),
                usage=total_usage
            )
            
            # Notify chain end
            for callback in self.callbacks:
                await callback.on_chain_end(chain_id, {
                    "success": success,
                    "output": result.output,
                    "error": result.error,
                    "usage": total_usage.dict() if total_usage else None
                })
            
            return result
            
        except Exception as e:
            # Handle unexpected errors
            logger.error(f"Chain execution failed: {str(e)}", exc_info=True)
            
            # Notify chain end with error
            for callback in self.callbacks:
                await callback.on_chain_end(chain_id, {
                    "success": False,
                    "error": str(e),
                    "error_type": "ChainError"
                })
            
            return NodeExecutionResult(
                success=False,
                output=None,
                error=f"Chain execution failed: {str(e)}",
                metadata=NodeMetadata(
                    node_id="chain",
                    node_type="chain",
                    version="1.0.0",
                    description="Workflow chain execution",
                    error_type="ChainError",
                    timestamp=datetime.utcnow()
                ),
                duration=sum(r.duration for r in results.values()) if results else 0,
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

    def _get_parent_nodes(self, node_id: str) -> List[str]:
        """Get parent nodes for a given node ID based on the graph structure.
        
        Args:
            node_id: The ID of the node to get parents for
            
        Returns:
            List of parent node IDs
        """
        return list(self.graph.predecessors(node_id))

    async def execute_node(self, node: BaseNode, node_id: str) -> Dict[str, Any]:
        """Execute a single node with optimized context
        
        Args:
            node: The node to execute
            node_id: The ID of the node
            
        Returns:
            The node's output
        """
        try:
            # Get optimized context for the node
            context = self.context.get_context_with_optimization(node_id)
            
            # Execute the node with the optimized context
            result = await node.execute(context)
            
            # Store the output in context
            if result.success and result.output:
                self.context.set_context(node_id, {"output": result.output})
            
            return result
        except Exception as e:
            self.logger.error(f"Error executing node {node_id}: {str(e)}")
            raise
    
    async def execute_workflow(self, nodes: List[BaseNode], node_ids: List[str]) -> Dict[str, NodeExecutionResult]:
        """Execute a workflow of nodes with optimized context management
        
        Args:
            nodes: List of nodes to execute
            node_ids: List of node IDs corresponding to the nodes
            
        Returns:
            Dictionary mapping node IDs to their execution results
        """
        # Build the graph
        for i, node_id in enumerate(node_ids):
            self.graph.add_node(node_id)
            if i > 0:
                self.graph.add_edge(node_ids[i-1], node_id)
        
        # Execute nodes in topological order
        results = {}
        for node_id in nx.topological_sort(self.graph):
            node_index = node_ids.index(node_id)
            node = nodes[node_index]
            
            # Execute the node with optimized context
            result = await self.execute_node(node, node_id)
            results[node_id] = result
        
        return results