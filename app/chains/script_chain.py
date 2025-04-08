"""
Main workflow orchestration chain for LLM nodes
"""

from datetime import datetime
import networkx as nx
from typing import Dict, List, Optional, Any, Union, Set
from app.models.node_models import NodeConfig, NodeExecutionResult, NodeMetadata, UsageMetadata
from app.models.config import LLMConfig, MessageTemplate
from app.nodes.text_generation import TextGenerationNode
from app.utils.context import ContextManager
from app.utils.retry import AsyncRetry
from app.nodes.base import BaseNode
from app.utils.callbacks import ScriptChainCallback
import logging
from uuid import uuid4
import asyncio

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
        callbacks: Optional[List[ScriptChainCallback]] = None,
        batch_size: int = 5
    ):
        """Initialize the script chain.
        
        Args:
            max_context_tokens: Maximum number of tokens allowed in the context
            callbacks: Optional list of callback handlers
            batch_size: Size of batches for processing
        """
        self.chain_id = f"chain_{uuid4().hex[:8]}"
        self.graph = nx.DiGraph()
        self.context = ContextManager(storage_path="context_store.json", max_context_tokens=max_context_tokens)
        self.logger = logging.getLogger(__name__)
        self.nodes: Dict[str, BaseNode] = {}
        self.retry = None
        self.callbacks = callbacks or []
        self.batch_size = batch_size
        logger.info(f"Created new script chain with ID: {self.chain_id}")
        
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
    
    async def execute(self) -> Dict[str, Any]:
        """Execute the entire chain in dependency order with batch processing."""
        execution_order = [(datetime.utcnow(), node_id) for node_id in self._get_execution_order()]
        
        # Notify chain start
        chain_config = {
            "node_count": len(self.nodes),
            "execution_order": execution_order,
            "batch_size": self.batch_size,
            "created_at": datetime.utcnow().isoformat()
        }
        for callback in self.callbacks:
            await callback.on_chain_start(self.chain_id, chain_config)

        results = {}
        total_usage = UsageMetadata()
        success = True

        # Process in batches
        for batch in self._chunked(execution_order, self.batch_size):
            if not success:
                break

            # Process batch concurrently
            batch_results = await asyncio.gather(
                *[self._process_node(node_id_tuple) for node_id_tuple in batch]
            )

            # Store results and check for failures
            for node_id_tuple, result in zip(batch, batch_results):
                _, node_id = node_id_tuple  # Extract node_id from the tuple
                results[node_id] = result
                total_usage = self._aggregate_usage(total_usage, result)
                
                if not result.success:
                    success = False
                    break

        if success:
            total_duration = (datetime.utcnow() - execution_order[0][0]).total_seconds()
            
            # Notify chain end
            chain_result = {
                "chain_id": self.chain_id,
                "success": success,
                "duration": total_duration,
                "node_results": {k: v.dict() for k, v in results.items()}
            }
            for callback in self.callbacks:
                await callback.on_chain_end(self.chain_id, chain_result)
            
            return {
                "success": success,
                "output": results[execution_order[-1][1]].output if success else None,
                "usage": total_usage.dict() if total_usage else None,
                "duration": total_duration
            }
        else:
            total_duration = (datetime.utcnow() - execution_order[0][0]).total_seconds()
            
            # Notify chain end with error
            chain_result = {
                "chain_id": self.chain_id,
                "success": False,
                "duration": total_duration,
                "node_results": {k: v.dict() for k, v in results.items()}
            }
            for callback in self.callbacks:
                await callback.on_chain_end(self.chain_id, chain_result)
            
            return {
                "success": False,
                "error": results[execution_order[-1][1]].error,
                "duration": total_duration
            }
    
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
            
            logger.debug(f"Retrieved context for node {node_id}: {context}")
            logger.debug(f"Execution result for node {node_id}: {result}")
            
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

    def _chunked(self, iterable, size):
        """Yield successive size chunks from iterable"""
        for i in range(0, len(iterable), size):
            yield iterable[i:i + size]

    async def _process_node(self, node_id_tuple: tuple) -> NodeExecutionResult:
        """Wrapper for node execution with error handling"""
        try:
            _, node_id = node_id_tuple  # Extract node_id from the tuple
            node = self.nodes[node_id]
            context = self.context.get_context_with_optimization(node_id)
            
            # Log the start of node execution
            logger.info(f"Starting execution of node {node_id}")
            
            # Execute node (with retry if configured)
            if self.retry:
                result = await self.retry.execute(node.execute, context)
            else:
                result = await node.execute(context)
            
            # Log the result of node execution
            logger.info(f"Execution result for node {node_id}: {result}")
            
            # Update context
            if result.success and result.output:
                self.context.set_context(node_id, {"output": result.output})
                logger.info(f"Context updated for node {node_id}")

            logger.debug(f"Retrieved context for node {node_id}: {context}")
            logger.debug(f"Execution result for node {node_id}: {result}")

            return result

        except Exception as e:
            logger.error(f"Error processing node {node_id}: {str(e)}")
            return NodeExecutionResult(
                success=False,
                error=str(e),
                metadata=NodeMetadata(
                    node_id=node_id,
                    node_type="error",
                    error_type=e.__class__.__name__
                )
            )

    def _aggregate_usage(self, total_usage: UsageMetadata, result: NodeExecutionResult) -> UsageMetadata:
        """Aggregate usage from a node execution result"""
        if result.usage:
            total_usage.prompt_tokens += result.usage.prompt_tokens
            total_usage.completion_tokens += result.usage.completion_tokens
            total_usage.total_tokens += result.usage.total_tokens
        return total_usage

    def _get_execution_order(self) -> List[str]:
        """Determine the execution order of nodes based on dependencies."""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible as e:
            self.logger.error("Cyclic dependency detected in the node graph.")
            raise ValueError("Cyclic dependency detected in the node graph.") from e