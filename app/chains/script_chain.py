"""
Enhanced workflow orchestration system with level-based parallel execution and robust context management
"""

from datetime import datetime
import networkx as nx
from typing import Dict, List, Optional, Any, Union, Set, Tuple
from pydantic import BaseModel, ValidationError
from app.models.node_models import NodeConfig, NodeExecutionResult, NodeMetadata, UsageMetadata
from app.models.config import LLMConfig, MessageTemplate
from app.utils.context import GraphContextManager
from app.utils.retry import AsyncRetry
from app.nodes.base import BaseNode
from app.nodes.text_generation import TextGenerationNode
from app.utils.callbacks import ScriptChainCallback
import logging
from uuid import uuid4
import asyncio
import os
import traceback
from collections import deque
from app.vector.pinecone_store import PineconeVectorStore
from app.models.vector_store import VectorStoreConfig

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
        self.nodes = {}  # For backward compatibility
        self.node_registry: Dict[str, BaseNode] = {}
        self.callbacks = callbacks or []
        
        # Initialize vector store with configurable settings
        vs_config = VectorStoreConfig(
            index_name=vector_store_config.get('index_name', os.getenv('PINECONE_INDEX', 'default-index')),
            environment=vector_store_config.get('environment', os.getenv('PINECONE_ENV', 'us-west1')),
            dimension=vector_store_config.get('dimension', 384),
            pod_type=vector_store_config.get('pod_type', 'p1'),
            replicas=vector_store_config.get('replicas', 1)
        ) if vector_store_config else None
        
        self.vector_store = PineconeVectorStore(vs_config) if vs_config else None
        
        # Custom context manager using graph structure
        self.context = GraphContextManager(
            max_tokens=max_context_tokens,
            graph=self.graph,
            vector_store_config=vector_store_config
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
        self.metrics = {
            'start_time': None,
            'total_tokens': 0,
            'node_execution_times': {}
        }
        
        logger.info(f"Initialized new ScriptChain: {self.chain_id}")

    def add_callback(self, callback: ScriptChainCallback) -> None:
        """Add a callback handler to the chain.
        
        Args:
            callback: Callback handler to add
        """
        self.callbacks.append(callback)
        logger.debug(f"Added callback handler: {callback.__class__.__name__}")

    def add_node(self, node: Union[BaseNode, NodeConfig]) -> None:
        """Register a node with dependency validation"""
        try:
            if isinstance(node, NodeConfig):
                # Create a TextGenerationNode instance from the config
                node_instance = TextGenerationNode(node, self.context)
                node_config = node.model_dump()
            else:
                # Node is already a BaseNode instance
                node_instance = node
                node_config = node.config.model_dump()
                
            if node_instance.node_id in self.node_registry:
                raise ValueError(f"Node {node_instance.node_id} already exists")
                
            self.node_registry[node_instance.node_id] = node_instance
            self.nodes[node_instance.node_id] = node_instance  # For backward compatibility
            self.graph.add_node(node_instance.node_id, node=node_instance)
            
            # Add edges for declared dependencies
            for dep_id in node_config['dependencies']:
                if dep_id not in self.node_registry:
                    logger.warning(f"Unresolved dependency {dep_id} for node {node_instance.node_id}")
                self.graph.add_edge(dep_id, node_instance.node_id)
                
            logger.debug(f"Added node {node_instance.node_id} with {len(node_config['dependencies'])} dependencies")
            
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

    def validate_workflow(self) -> bool:
        """Validate workflow structure before execution"""
        # Check for cyclic dependencies
        cycles = list(nx.simple_cycles(self.graph))
        if cycles:
            cycle_str = ", ".join([" -> ".join(cycle) for cycle in cycles])
            logger.error(f"Workflow contains cyclic dependencies: {cycle_str}")
            raise ValueError("Workflow contains cyclic dependencies")
            
        orphan_nodes = [n for n in self.graph.nodes if self.graph.in_degree(n) == 0]
        if len(orphan_nodes) > 1:
            logger.warning(f"Multiple orphan nodes detected: {orphan_nodes}")
            
        # Check for disconnected components
        if not nx.is_weakly_connected(self.graph) and len(self.graph.nodes) > 1:
            components = list(nx.weakly_connected_components(self.graph))
            logger.warning(f"Workflow contains {len(components)} disconnected components")
            for i, component in enumerate(components):
                logger.warning(f"Component {i+1}: {component}")
        
        return True

    def _get_level_dependencies(self, level_nodes: List[str]) -> List[str]:
        """Get all dependencies for nodes in a level"""
        dependencies = set()
        for node_id in level_nodes:
            for predecessor in self.graph.predecessors(node_id):
                dependencies.add(predecessor)
        return list(dependencies)

    def _calculate_execution_levels(self) -> List[ExecutionLevel]:
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

    async def execute(self) -> Dict[str, NodeExecutionResult]:
        """Execute workflow with level-based parallel processing"""
        self.validate_workflow()
        execution_levels = self._calculate_execution_levels()
        self.metrics['start_time'] = datetime.utcnow()
        
        # Notify execution start
        await self._trigger_callbacks('chain_start', NodeExecutionResult(
            success=True,
            output={
                'chain_id': self.chain_id,
                'total_nodes': len(self.node_registry),
                'execution_levels': [l.model_dump() for l in execution_levels]
            },
            metadata=NodeMetadata(
                node_id=self.chain_id,
                node_type="chain",
                start_time=self.metrics['start_time']
            )
        ))
        
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
        
        async def process_node(node_id: str) -> Tuple[str, NodeExecutionResult]:
            async with semaphore:
                node = self.node_registry[node_id]
                if not hasattr(node, 'execute'):
                    logger.error(f"Node {node_id} does not have execute method")
                    return node_id, NodeExecutionResult(
                        success=False,
                        error=f"Node {node_id} is not properly initialized",
                        output=None,
                        metadata=NodeMetadata(
                            node_id=node_id,
                            node_type="unknown",
                            error_type="InvalidNodeError"
                        )
                    )
                
                context = self.context.get_context(node_id)
                start_time = datetime.utcnow()
                
                try:
                    # Execute with retry policy and context optimization
                    if hasattr(self, 'execute_node'):  # For test mocking
                        result = await self.execute_node(node_id, context)
                    else:
                        result = await self.retry_policy.execute(
                            node.execute,
                            context,
                            metadata={'node_id': node_id}
                        )
                    
                    # Update context and metrics
                    await self.context.update(node_id, result)
                    self._update_metrics(node_id, result)
                    
                    await self._trigger_callbacks('node_end', NodeExecutionResult(
                        success=True,
                        output={
                            'node_id': node_id,
                            'result': result.model_dump(),
                            'level': level.level
                        },
                        metadata=NodeMetadata(
                            node_id=node_id,
                            node_type=node.config.type if hasattr(node, 'config') else "unknown",
                            start_time=start_time,
                            end_time=datetime.utcnow()
                        )
                    ))
                    
                    return node_id, result
                    
                except Exception as e:
                    error_result = NodeExecutionResult(
                        success=False,
                        error=str(e),
                        output=None,
                        metadata=NodeMetadata(
                            node_id=node_id,
                            node_type=node.config.type if hasattr(node, 'config') else "unknown",
                            start_time=start_time,
                            end_time=datetime.utcnow(),
                            error_type=e.__class__.__name__,
                            error_traceback=traceback.format_exc()
                        )
                    )
                    await self._trigger_callbacks('node_error', error_result)
                    return node_id, error_result

        tasks = [process_node(node_id) for node_id in level.node_ids]
        results = await asyncio.gather(*tasks)
        return dict(results)

    async def _handle_execution_success(self, results: Dict[str, NodeExecutionResult]) -> Dict[str, NodeExecutionResult]:
        """Handle successful chain execution"""
        end_time = datetime.utcnow()
        success_result = NodeExecutionResult(
            success=True,
            output=results,
            metadata=NodeMetadata(
                node_id=self.chain_id,
                node_type="chain",
                start_time=self.metrics['start_time'],
                end_time=end_time
            )
        )
        await self._trigger_callbacks('chain_end', success_result)
        return success_result

    async def _handle_execution_failure(self, level: ExecutionLevel, results: Dict[str, NodeExecutionResult]) -> Dict[str, NodeExecutionResult]:
        """Handle level execution failure"""
        end_time = datetime.utcnow()
        failed_nodes = [(nid, r) for nid, r in results.items() if not r.success]
        error_message = failed_nodes[0][1].error if failed_nodes else f"Execution failed at level {level.level}"
        
        failure_result = NodeExecutionResult(
            success=False,
            error=error_message,
            output=results,
            metadata=NodeMetadata(
                node_id=self.chain_id,
                node_type="chain",
                start_time=self.metrics['start_time'],
                end_time=end_time,
                error_type="LevelExecutionError",
                error_traceback=f"Failed nodes at level {level.level}: {[nid for nid, r in failed_nodes]}"
            )
        )
        await self._trigger_callbacks('chain_end', failure_result)
        return failure_result

    async def _handle_critical_failure(self, error: Exception) -> Dict[str, NodeExecutionResult]:
        """Handle critical chain failure"""
        end_time = datetime.utcnow()
        failure_result = NodeExecutionResult(
            success=False,
            error=str(error),
            output=None,
            metadata=NodeMetadata(
                node_id=self.chain_id,
                node_type="chain",
                start_time=self.metrics['start_time'],
                end_time=end_time,
                error_type=error.__class__.__name__,
                error_traceback=traceback.format_exc()
            )
        )
        await self._trigger_callbacks('chain_end', failure_result)
        return failure_result

    def _update_metrics(self, node_id: str, result: NodeExecutionResult) -> None:
        """Update chain metrics with node execution results"""
        if result.usage:
            self.metrics['total_tokens'] += result.usage.total_tokens
        self.metrics['node_execution_times'][node_id] = datetime.utcnow()

    async def _trigger_callbacks(self, event: str, data: NodeExecutionResult) -> None:
        """Trigger registered callbacks for an event"""
        event_map = {
            'chain_start': 'on_chain_start',
            'chain_end': 'on_chain_end',
            'node_start': 'on_node_start',
            'node_end': 'on_node_end',
            'node_error': 'on_node_error'
        }
        
        method_name = event_map.get(event)
        if not method_name:
            logger.warning(f"Unknown callback event: {event}")
            return
            
        for callback in self.callbacks:
            try:
                method = getattr(callback, method_name)
                if event == 'chain_start':
                    await method(
                        chain_id=self.chain_id,
                        inputs=data.output or {}
                    )
                elif event == 'chain_end':
                    await method(
                        chain_id=self.chain_id,
                        outputs=data.output or {},
                        error=data.error if not data.success else None
                    )
                elif event == 'node_start':
                    await method(
                        chain_id=self.chain_id,
                        node_id=data.metadata.node_id,
                        inputs=data.output or {}
                    )
                elif event == 'node_end':
                    await method(
                        chain_id=self.chain_id,
                        node_id=data.metadata.node_id,
                        outputs=data.output or {}
                    )
                else:  # node_error
                    await method(
                        chain_id=self.chain_id,
                        node_id=data.metadata.node_id,
                        error=data.error
                    )
            except Exception as e:
                logger.error(f"Callback error in {callback.__class__.__name__}: {e}")