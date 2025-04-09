"""
Enhanced workflow orchestration system with level-based parallel execution and robust context management
"""

from datetime import datetime
import networkx as nx
from typing import Dict, List, Optional, Any, Union, Set
from pydantic import BaseModel, ValidationError
from app.models.node_models import NodeConfig, NodeExecutionResult, NodeMetadata, UsageMetadata
from app.models.config import LLMConfig, MessageTemplate
from app.utils.context import GraphContextManager
from app.utils.retry import AsyncRetry
from app.nodes.base import BaseNode
from app.utils.callbacks import ScriptChainCallback
import logging
from uuid import uuid4
import asyncio
from langchain import LangChain
from langchain.vectorstores import PineconeVectorStore
import os
import traceback
from collections import deque

logger = logging.getLogger(__name__)

class ExecutionLevel(BaseModel):
    """Represents a group of nodes that can be executed in parallel"""
    level: int
    node_ids: List[str]
    dependencies: List[str]

class ScriptChain:
    """Advanced workflow orchestrator with level-based parallel execution"""
    
    def __init__(
        self,
        max_context_tokens: int = 4000,
        callbacks: Optional[List[ScriptChainCallback]] = None,
        concurrency_level: int = 10,
        retry_policy: Optional[AsyncRetry] = None,
        vector_store_config: Optional[Dict] = None
    ):
        """Initialize the script chain with enhanced configuration"""
        self.chain_id = f"chain_{uuid4().hex[:8]}"
        self.graph = nx.DiGraph()
        self.node_registry: Dict[str, BaseNode] = {}
        
        # Initialize vector store with configurable settings
        vs_config = vector_store_config or {
            'index_name': os.getenv('PINECONE_INDEX', 'default-index'),
            'api_key': os.getenv('PINECONE_API_KEY'),
            'environment': os.getenv('PINECONE_ENV', 'us-west1')
        }
        self.vector_store = PineconeVectorStore(**vs_config)
        
        # Custom context manager using graph structure
        self.context = GraphContextManager(
            max_tokens=max_context_tokens,
            graph=self.graph,
            vector_store=self.vector_store
        )
        
        # Execution configuration
        self.concurrency_level = concurrency_level
        self.retry_policy = retry_policy or AsyncRetry(
            max_retries=3,
            delay=0.5,
            backoff=2,
            exceptions=(Exception,)
        )
        
        # Observability
        self.callbacks = callbacks or []
        self.metrics = {
            'start_time': None,
            'total_tokens': 0,
            'node_execution_times': {}
        }
        
        logger.info(f"Initialized new ScriptChain: {self.chain_id}")

    def add_node(self, node: BaseNode) -> None:
        """Register a node with dependency validation"""
        try:
            NodeConfig.validate(node.config)
            if node.node_id in self.node_registry:
                raise ValueError(f"Node {node.node_id} already exists")
                
            self.node_registry[node.node_id] = node
            self.graph.add_node(node.node_id, node=node)
            
            # Add edges for declared dependencies
            for dep_id in node.config.dependencies:
                if dep_id not in self.node_registry:
                    logger.warning(f"Unresolved dependency {dep_id} for node {node.node_id}")
                self.graph.add_edge(dep_id, node.node_id)
                
            logger.debug(f"Added node {node.node_id} with {len(node.config.dependencies)} dependencies")
            
        except ValidationError as ve:
            logger.error(f"Invalid node configuration: {ve}")
            raise

    def add_edge(self, from_node_id: str, to_node_id: str) -> None:
        """Add a dependency edge between nodes with validation.
        
        Args:
            from_node_id: ID of the source node
            to_node_id: ID of the target node
        """
        if from_node_id not in self.node_registry:
            raise ValueError(f"Source node {from_node_id} not found in chain")
        if to_node_id not in self.node_registry:
            raise ValueError(f"Target node {to_node_id} not found in chain")
            
        self.graph.add_edge(from_node_id, to_node_id)
        # Update dependencies in target node's config
        if from_node_id not in self.node_registry[to_node_id].config.dependencies:
            self.node_registry[to_node_id].config.dependencies.append(from_node_id)
            logger.debug(f"Added dependency {from_node_id} to node {to_node_id}")

    def validate_workflow(self) -> None:
        """Validate workflow structure before execution"""
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Workflow contains cyclic dependencies")
            
        orphan_nodes = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        if len(orphan_nodes) > 1:
            logger.warning(f"Multiple orphan nodes detected: {orphan_nodes}")
            
        # Check for disconnected components
        if not nx.is_weakly_connected(self.graph):
            components = list(nx.weakly_connected_components(self.graph))
            logger.warning(f"Workflow contains {len(components)} disconnected components")
            for i, component in enumerate(components):
                logger.warning(f"Component {i+1}: {component}")

    def _get_level_dependencies(self, level_nodes: List[str]) -> List[str]:
        """Get all dependencies for nodes in a level"""
        dependencies = set()
        for node_id in level_nodes:
            for predecessor in self.graph.predecessors(node_id):
                dependencies.add(predecessor)
        return list(dependencies)

    def calculate_execution_levels(self) -> List[ExecutionLevel]:
        """Group nodes into parallel execution levels using Kahn's algorithm"""
        in_degree = {n: self.graph.in_degree(n) for n in self.graph.nodes}
        queue = deque([n for n, d in in_degree.items() if d == 0])
        levels = []
        current_level = 0
        
        while queue:
            level_size = len(queue)
            level_nodes = []
            
            for _ in range(level_size):
                node = queue.popleft()
                level_nodes.append(node)
                
                for successor in self.graph.successors(node):
                    in_degree[successor] -= 1
                    if in_degree[successor] == 0:
                        queue.append(successor)
            
            levels.append(ExecutionLevel(
                level=current_level,
                node_ids=level_nodes,
                dependencies=self._get_level_dependencies(level_nodes)
            ))
            current_level += 1
            
        return levels

    async def execute(self) -> Dict[str, Any]:
        """Execute workflow with level-based parallel processing"""
        self.validate_workflow()
        execution_levels = self.calculate_execution_levels()
        self.metrics['start_time'] = datetime.utcnow()
        
        # Notify execution start
        await self._trigger_callbacks('on_chain_start', {
            'chain_id': self.chain_id,
            'total_nodes': len(self.node_registry),
            'execution_levels': [l.dict() for l in execution_levels]
        })
        
        results = {}
        try:
            for level in execution_levels:
                level_results = await self._process_level(level)
                results.update(level_results)
                
                if any(not r.success for r in level_results.values()):
                    return await self._handle_execution_failure(level, results)
                
            return await self._handle_execution_success(results)
            
        except Exception as e:
            logger.error(f"Critical chain failure: {traceback.format_exc()}")
            return await self._handle_critical_failure(e)

    async def _process_level(self, level: ExecutionLevel) -> Dict[str, NodeExecutionResult]:
        """Process all nodes in a level with controlled concurrency"""
        semaphore = asyncio.Semaphore(self.concurrency_level)
        
        async def process_node(node_id: str):
            async with semaphore:
                node = self.node_registry[node_id]
                context = self.context.get_context(node_id, level.dependencies)
                
                try:
                    # Execute with retry policy and context optimization
                    result = await self.retry_policy.execute(
                        node.execute,
                        context,
                        metadata={'node_id': node_id}
                    )
                    
                    # Update context and metrics
                    self.context.update(node_id, result)
                    self._update_metrics(node_id, result)
                    
                    await self._trigger_callbacks('on_node_end', {
                        'node_id': node_id,
                        'result': result.dict(),
                        'level': level.level
                    })
                    
                    return node_id, result
                    
                except Exception as e:
                    error_result = NodeExecutionResult(
                        success=False,
                        error=str(e),
                        metadata=NodeMetadata(
                            node_id=node_id,
                            error_type=e.__class__.__name__,
                            traceback=traceback.format_exc()
                        )
                    )
                    self.context.log_error(node_id, error_result)
                    await self._trigger_callbacks('on_node_error', {
                        'node_id': node_id,
                        'error': error_result.dict(),
                        'level': level.level
                    })
                    return node_id, error_result
        
        # Execute all nodes in level concurrently
        tasks = [process_node(nid) for nid in level.node_ids]
        results = dict(await asyncio.gather(*tasks))
        
        # Validate level completion
        if all(r.success for r in results.values()):
            logger.info(f"Completed level {level.level} with {len(results)} nodes")
            return results
            
        logger.error(f"Level {level.level} failed with {sum(not r.success for r in results.values())} errors")
        return results

    async def _handle_execution_success(self, results: Dict[str, NodeExecutionResult]) -> Dict[str, Any]:
        """Handle successful execution completion"""
        total_duration = (datetime.utcnow() - self.metrics['start_time']).total_seconds()
        
        # Aggregate usage metrics
        total_usage = UsageMetadata()
        for result in results.values():
            if result.usage:
                total_usage.prompt_tokens += result.usage.prompt_tokens
                total_usage.completion_tokens += result.usage.completion_tokens
                total_usage.total_tokens += result.usage.total_tokens
        
        # Find the final output (from nodes with no successors)
        final_nodes = [n for n in self.graph.nodes if self.graph.out_degree(n) == 0]
        final_output = None
        if final_nodes:
            # Use the last node in the execution order as the final output
            final_node_id = final_nodes[-1]
            if final_node_id in results and results[final_node_id].success:
                final_output = results[final_node_id].output
        
        # Notify chain end
        chain_result = {
            "chain_id": self.chain_id,
            "success": True,
            "duration": total_duration,
            "node_results": {k: v.dict() for k, v in results.items()},
            "metrics": self.metrics
        }
        await self._trigger_callbacks('on_chain_end', chain_result)
        
        return {
            "success": True,
            "output": final_output,
            "usage": total_usage.dict() if total_usage else None,
            "duration": total_duration,
            "metrics": self.metrics
        }

    async def _handle_execution_failure(self, failed_level: ExecutionLevel, results: Dict[str, NodeExecutionResult]) -> Dict[str, Any]:
        """Handle execution failure at a specific level"""
        total_duration = (datetime.utcnow() - self.metrics['start_time']).total_seconds()
        
        # Find the first failed node
        failed_nodes = [nid for nid, result in results.items() if not result.success]
        error_message = f"Execution failed at level {failed_level.level} with {len(failed_nodes)} failed nodes"
        
        # Notify chain end with error
        chain_result = {
            "chain_id": self.chain_id,
            "success": False,
            "duration": total_duration,
            "node_results": {k: v.dict() for k, v in results.items()},
            "failed_level": failed_level.level,
            "failed_nodes": failed_nodes,
            "metrics": self.metrics
        }
        await self._trigger_callbacks('on_chain_end', chain_result)
        
        return {
            "success": False,
            "error": error_message,
            "failed_nodes": failed_nodes,
            "duration": total_duration,
            "metrics": self.metrics
        }

    async def _handle_critical_failure(self, error: Exception) -> Dict[str, Any]:
        """Handle critical failure that prevents normal execution"""
        total_duration = (datetime.utcnow() - self.metrics['start_time']).total_seconds()
        
        # Notify chain end with critical error
        chain_result = {
            "chain_id": self.chain_id,
            "success": False,
            "duration": total_duration,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "metrics": self.metrics
        }
        await self._trigger_callbacks('on_chain_end', chain_result)
        
        return {
            "success": False,
            "error": str(error),
            "traceback": traceback.format_exc(),
            "duration": total_duration,
            "metrics": self.metrics
        }

    def _update_metrics(self, node_id: str, result: NodeExecutionResult) -> None:
        """Update execution metrics"""
        exec_time = (datetime.utcnow() - self.metrics['start_time']).total_seconds()
        self.metrics['node_execution_times'][node_id] = exec_time
        
        if result.usage:
            self.metrics['total_tokens'] += result.usage.total_tokens

    async def _trigger_callbacks(self, event: str, data: Dict) -> None:
        """Trigger registered callbacks"""
        for callback in self.callbacks:
            method = getattr(callback, event, None)
            if method:
                try:
                    await method(self.chain_id, data)
                except Exception as e:
                    logger.error(f"Error in callback {event}: {str(e)}")
                    logger.debug(traceback.format_exc())

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