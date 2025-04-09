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

    def add_node(self, node: NodeConfig) -> None:
        """Add a node to the workflow.
        
        Args:
            node (NodeConfig): The node to add
            
        Raises:
            ValueError: If adding the node would create a cyclic dependency
        """
        # Create a temporary graph for cycle detection
        temp_graph = self.graph.copy()
        temp_graph.add_node(node.id)
        for dep in node.dependencies:
            temp_graph.add_edge(dep, node.id)
        
        def has_cycle(node_id: str, visited: Set[str], path: Set[str]) -> bool:
            if node_id in path:
                return True
            if node_id in visited:
                return False
                
            visited.add(node_id)
            path.add(node_id)
            
            for successor in temp_graph.successors(node_id):
                if has_cycle(successor, visited, path):
                    return True
                    
            path.remove(node_id)
            return False
            
        # Check for cycles starting from the new node
        if has_cycle(node.id, set(), set()):
            raise ValueError("Workflow contains cyclic dependencies")
            
        # If no cycles, add the node and its edges
        self.nodes[node.id] = node
        
        # Create proper node instance based on type
        if node.type == "llm":
            node_instance = TextGenerationNode(
                config=node,
                context_manager=self.context
            )
        else:
            node_instance = BaseNode(
                config=node
            )
            
        self.node_registry[node.id] = node_instance
        self.graph.add_node(node.id)
        for dep in node.dependencies:
            self.graph.add_edge(dep, node.id)

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
        """Validate the workflow structure.
        
        Checks for:
        - Cyclic dependencies
        - Orphan nodes (no dependencies)
        - Disconnected components
        
        Raises:
            ValueError: If cyclic dependencies are detected
        """
        # Check for cyclic dependencies using DFS
        visited = set()
        path = set()
        
        def has_cycle(node_id: str) -> bool:
            if node_id in path:
                cycle_path = " -> ".join(list(path) + [node_id])
                raise ValueError(f"Workflow contains cyclic dependencies: {cycle_path}")
            if node_id in visited:
                return False
            
            visited.add(node_id)
            path.add(node_id)
            
            for successor in self.graph.successors(node_id):
                if has_cycle(successor):
                    return True
                    
            path.remove(node_id)
            return False
        
        # Check each node for cycles
        for node_id in self.nodes:
            if has_cycle(node_id):
                return False
        
        # Check for orphan nodes
        orphans = []
        for node_id, node in self.nodes.items():
            if not node.dependencies and not any(
                node_id in n.dependencies for n in self.nodes.values()
            ):
                orphans.append(node_id)
            
        if orphans:
            logger.warning(f"Orphan nodes detected: {', '.join(orphans)}")
        
        # Check for disconnected components
        components = self._find_components()
        if len(components) > 1:
            logger.warning(
                f"Multiple disconnected components detected: {len(components)}"
            )
        
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

    async def execute(self) -> NodeExecutionResult:
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
                    failed_nodes = [(nid, r) for nid, r in level_results.items() if not r.success]
                    error_message = failed_nodes[0][1].error if failed_nodes else f"Execution failed at level {level.level}"
                    return NodeExecutionResult(
                        success=False,
                        error=error_message,
                        output=results,
                        metadata=NodeMetadata(
                            node_id=self.chain_id,
                            node_type="chain",
                            start_time=self.metrics['start_time'],
                            end_time=datetime.utcnow(),
                            error_type="LevelExecutionError"
                        )
                    )
                
            return NodeExecutionResult(
                success=True,
                output=results,
                metadata=NodeMetadata(
                    node_id=self.chain_id,
                    node_type="chain",
                    start_time=self.metrics['start_time'],
                    end_time=datetime.utcnow()
                )
            )
            
        except Exception as e:
            logger.error(f"Critical chain failure: {traceback.format_exc()}")
            return NodeExecutionResult(
                success=False,
                error=str(e),
                output=results,
                metadata=NodeMetadata(
                    node_id=self.chain_id,
                    node_type="chain",
                    start_time=self.metrics['start_time'],
                    end_time=datetime.utcnow(),
                    error_type=e.__class__.__name__,
                    error_traceback=traceback.format_exc()
                )
            )

    async def _process_level(self, level: ExecutionLevel) -> Dict[str, NodeExecutionResult]:
        """Process all nodes in a level with controlled concurrency"""
        semaphore = asyncio.Semaphore(self.concurrency_level)
        
        async def process_node(node_id: str) -> Tuple[str, NodeExecutionResult]:
            async with semaphore:
                try:
                    node = self.node_registry[node_id]
                except KeyError:
                    return node_id, NodeExecutionResult(
                        success=False,
                        error=f"Node {node_id} not found in registry",
                        metadata=NodeMetadata(
                            node_id=node_id,
                            node_type="unknown",
                            error_type="NodeNotFoundError"
                        )
                    )
                
                if not hasattr(node, 'execute'):
                    logger.error(f"Node {node_id} does not have execute method")
                    return node_id, NodeExecutionResult(
                        success=False,
                        error=f"Node {node_id} is not properly initialized",
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
                        result = await self.execute_node(node, node_id)
                    else:
                        result = await self.retry_policy.execute(
                            lambda: node.execute(context)
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
                            node_type=node.type if hasattr(node, 'type') else "unknown",
                            start_time=start_time,
                            end_time=datetime.utcnow()
                        )
                    ))
                    
                    return node_id, result
                    
                except Exception as e:
                    error_result = NodeExecutionResult(
                        success=False,
                        error=str(e),
                        metadata=NodeMetadata(
                            node_id=node_id,
                            node_type=node.type if hasattr(node, 'type') else "unknown",
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

    async def execute_node(self, node: BaseNode, node_id: str) -> NodeExecutionResult:
        """Execute a single node with retry and context management.
        
        Args:
            node: Node instance to execute
            node_id: ID of the node
            
        Returns:
            NodeExecutionResult containing execution output and metadata
        """
        try:
            # Get optimized context using node's prompt as query
            context = await self.context.get_context_with_optimization(
                node_id=node_id,
                query=node.config.prompt or "",  # Use prompt as query or empty string
                k=5,
                threshold=0.7
            )
            
            # Execute node with retry policy
            result = await self.retry_policy.execute(
                lambda: node.execute(context)
            )
            
            # Update context with result
            await self.context.update(node_id, result)
            
            # Update metrics
            if result.metadata and result.metadata.usage:
                self.metrics['total_tokens'] += result.metadata.usage.total_tokens
            
            return result
            
        except Exception as e:
            error_msg = f"Error executing node {node_id}: {str(e)}"
            logger.error(error_msg)
            self.context.log_error(node_id, e)
            
            # Create failure result
            return NodeExecutionResult(
                success=False,
                error=error_msg,
                metadata=NodeMetadata(
                    start_time=datetime.utcnow(),
                    end_time=datetime.utcnow(),
                    usage=UsageMetadata(total_tokens=0)
                )
            )

    def _find_components(self) -> List[Set[str]]:
        """Find disconnected components in the workflow graph using DFS.
        
        Returns:
            List[Set[str]]: List of sets containing node IDs in each component
        """
        components = []
        visited = set()
        
        def dfs(node_id: str, component: Set[str]):
            if node_id in visited:
                return
            visited.add(node_id)
            component.add(node_id)
            
            # Check dependencies
            node = self.nodes.get(node_id)
            if node and node.dependencies:
                for dep in node.dependencies:
                    dfs(dep, component)
                    
            # Check nodes that depend on this one
            for other_id, other_node in self.nodes.items():
                if node_id in other_node.dependencies:
                    dfs(other_id, component)
        
        # Find all components
        for node_id in self.nodes:
            if node_id not in visited:
                component = set()
                dfs(node_id, component)
                components.append(component)
                
        return components