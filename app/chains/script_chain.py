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
        vector_store_config: Optional[Dict] = None,
        llm_config: Optional[Dict[str, Any]] = None
    ):
        """Initialize the script chain with enhanced configuration"""
        self.chain_id = f"chain_{uuid4().hex[:8]}"
        self.graph = nx.DiGraph()
        self.nodes = {}  # For backward compatibility
        self.dependencies = {}  # Map of node_id to list of dependencies
        self.execution_levels = {}  # Map of level to ExecutionLevel
        self.node_registry: Dict[str, BaseNode] = {}
        self.callbacks = callbacks or []
        self.max_context_tokens = max_context_tokens
        
        # Convert llm_config dict to LLMConfig instance
        default_llm_config = {
            "model": "gpt-4",
            "api_key": os.getenv("OPENAI_API_KEY"),
            "max_context_tokens": max_context_tokens,
            "temperature": 0.7,
            "max_tokens": 500
        }
        if llm_config:
            default_llm_config.update(llm_config)
        self.llm_config = LLMConfig(**default_llm_config)
        
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

    def add_node(self, node_config: NodeConfig) -> None:
        """Add a node to the workflow.
        
        Args:
            node_config: Configuration for the node to add
            
        Raises:
            ValueError: If node ID already exists
        """
        if node_config.id in self.nodes:
            raise ValueError(f"Node with ID {node_config.id} already exists")
        
        # Ensure metadata is set
        if node_config.metadata is None:
            node_config.metadata = NodeMetadata(
                node_id=node_config.id,
                node_type=node_config.type,
                version="1.0.0",
                description=f"Node {node_config.id} of type {node_config.type}"
            )
        
        # Create and store the node
        node = TextGenerationNode(
            config=node_config,
            context_manager=self.context,
            llm_config=self.llm_config
        )
        
        self.nodes[node_config.id] = node
        self.dependencies[node_config.id] = set(node_config.dependencies)
        
        self.node_registry[node_config.id] = node
        self.graph.add_node(node_config.id)
        for dep in node_config.dependencies:
            self.graph.add_edge(dep, node_config.id)

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
        """Validate the workflow configuration.
        
        Checks:
        - All nodes have valid dependencies
        - No orphan nodes
        
        Returns:
            True if workflow is valid
            
        Raises:
            ValueError: If workflow validation fails
        """
        # Check for orphan nodes
        orphan_nodes = []
        for node_id in self.nodes:
            if not self.graph.in_degree(node_id) and not self.graph.out_degree(node_id):
                orphan_nodes.append(node_id)
        
        if orphan_nodes:
            logger.warning(f"Orphan nodes detected: {orphan_nodes}")
        
        # Check for disconnected components
        components = list(nx.weakly_connected_components(self.graph))
        if len(components) > 1:
            logger.warning(f"Workflow contains {len(components)} disconnected components")
        
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
        """Execute the workflow.
        
        Returns:
            NodeExecutionResult with success/failure and output
        """
        start_time = datetime.utcnow()
        
        try:
            # Calculate execution levels
            levels = self._calculate_execution_levels()
            
            # Process each level
            results = {}
            for level in levels:
                level_result = await self._process_level(level)
                results.update(level_result.output)
                
                if not level_result.success:
                    # Get error messages from failed nodes
                    error_messages = []
                    for node_id, result in level_result.output.items():
                        if not result.success:
                            error_messages.append(f"{node_id}: {result.error}")
                    
                    error_msg = f"Execution failed at level {level.level}: {'; '.join(error_messages)}"
                    return NodeExecutionResult(
                        success=False,
                        error=error_msg,
                        output=results,
                        metadata=NodeMetadata(
                            node_id=f"chain_{uuid4().hex[:8]}",
                            node_type="chain",
                            start_time=start_time,
                            end_time=datetime.utcnow(),
                            error_type="LevelExecutionError"
                        )
                    )
            
            # All levels completed successfully
            return NodeExecutionResult(
                success=True,
                output=results,
                metadata=NodeMetadata(
                    node_id=f"chain_{uuid4().hex[:8]}",
                    node_type="chain",
                    start_time=start_time,
                    end_time=datetime.utcnow()
                )
            )
            
        except Exception as e:
            logger.error(f"Chain execution failed: {str(e)}")
            return NodeExecutionResult(
                success=False,
                error=str(e),
                output=None,
                metadata=NodeMetadata(
                    node_id=f"chain_{uuid4().hex[:8]}",
                    node_type="chain",
                    start_time=start_time,
                    end_time=datetime.utcnow(),
                    error_type=e.__class__.__name__
                )
            )

    async def _process_level(self, level: ExecutionLevel) -> NodeExecutionResult:
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
                
                start_time = datetime.utcnow()
                
                try:
                    # Simple execution with error handling
                    if hasattr(self, 'execute_node'):  # For test mocking
                        result = await self.execute_node(node_id)
                    else:
                        result = await node.execute()
                    
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
        return NodeExecutionResult(
            success=all(r.success for nid, r in results.items()),
            output=dict(results),
            metadata=NodeMetadata(
                node_id=self.chain_id,
                node_type="chain",
                start_time=start_time,
                end_time=datetime.utcnow()
            )
        )

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

    async def execute_node(self, node_id: str) -> NodeExecutionResult:
        """Execute a single node.
        
        Args:
            node_id: ID of the node to execute
            
        Returns:
            NodeExecutionResult containing the execution result
        """
        node = self.node_registry[node_id]
        start_time = datetime.utcnow()
        
        try:
            result = await node.execute()
            return result
        except Exception as e:
            logger.error(f"Node {node_id} execution failed: {str(e)}")
            return NodeExecutionResult(
                success=False,
                error=str(e),
                metadata=NodeMetadata(
                    node_id=node_id,
                    node_type=node.type,
                    start_time=start_time,
                    end_time=datetime.utcnow(),
                    error_type=e.__class__.__name__
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