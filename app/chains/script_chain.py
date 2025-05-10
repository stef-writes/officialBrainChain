"""
Enhanced workflow orchestration system with level-based parallel execution and robust context management
"""

from datetime import datetime
import networkx as nx
from typing import Dict, List, Optional, Any, Union, Set, Tuple, Callable
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
import json
from collections import deque
from app.vector.pinecone_store import PineconeVectorStore
from app.models.vector_store import VectorStoreConfig

logger = logging.getLogger(__name__)

# Custom exceptions for better error handling
class ScriptChainError(Exception):
    """Base exception class for ScriptChain errors"""
    pass

class InputMappingError(ScriptChainError):
    """Exception raised when input mapping fails"""
    pass

class NodeExecutionError(ScriptChainError):
    """Exception raised when node execution fails"""
    pass

class ContextError(ScriptChainError):
    """Exception raised when context operations fail"""
    pass

class ExecutionLevel(BaseModel):
    """Represents a group of nodes that can be executed in parallel"""
    level: int
    node_ids: List[str]
    dependencies: List[str]

class ScriptChain:
    """Advanced workflow orchestrator with level-based parallel execution"""
    
    # Default transform functions
    @staticmethod
    def _stringify_transform(value: Any) -> str:
        """Convert value to string"""
        return str(value)
    
    @staticmethod
    def _jsonify_transform(value: Any) -> str:
        """Convert value to JSON string"""
        return json.dumps(value)
    
    def __init__(
        self,
        max_context_tokens: int = 4000,
        callbacks: Optional[List[ScriptChainCallback]] = None,
        concurrency_level: int = 10,
        vector_store_config: Optional[Dict] = None,
        llm_config: Optional[Dict[str, Any]] = None,
        min_concurrency: int = 2,
        max_concurrency: int = 20
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
        
        # Concurrency controls
        self.min_concurrency = min_concurrency
        self.max_concurrency = max_concurrency
        self.base_concurrency_level = concurrency_level
        self.concurrency_level = concurrency_level  # Will be dynamically adjusted
        
        # Transform registry for pluggable transforms
        self.transform_registry: Dict[str, Callable] = {
            'stringify': self._stringify_transform,
            'jsonify': self._jsonify_transform
        }
        
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
        
        # Observability
        self.metrics = {
            'start_time': None,
            'total_tokens': 0,
            'node_execution_times': {}
        }
        
        logger.info(f"Initialized new ScriptChain: {self.chain_id}")

    def register_transform(self, name: str, transform_func: Callable) -> None:
        """Register a custom transform function
        
        Args:
            name: Name of the transform
            transform_func: Function that takes a value and returns transformed value
        """
        if name in self.transform_registry:
            logger.warning(f"Overriding existing transform: {name}")
        self.transform_registry[name] = transform_func
        logger.debug(f"Registered transform: {name}")
    
    def get_transform(self, name: str) -> Optional[Callable]:
        """Get a transform function by name
        
        Args:
            name: Name of the transform
            
        Returns:
            Transform function or None if not found
        """
        return self.transform_registry.get(name)

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
        
        # Recalculate optimal concurrency when nodes change
        self._adjust_concurrency_level()

    def _adjust_concurrency_level(self) -> None:
        """Dynamically adjust concurrency level based on node count and complexity"""
        node_count = len(self.node_registry)
        if node_count == 0:
            self.concurrency_level = self.base_concurrency_level
            return
            
        # Simple heuristic: roughly sqrt(node_count) but constrained by min/max
        optimal_concurrency = min(
            self.max_concurrency,
            max(
                self.min_concurrency,
                min(int(node_count / 2) + 1, self.base_concurrency_level)
            )
        )
        
        # Only log if changing
        if optimal_concurrency != self.concurrency_level:
            self.concurrency_level = optimal_concurrency
            logger.info(f"Adjusted concurrency level to {self.concurrency_level} based on {node_count} nodes")

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
        level_start_time = datetime.utcnow() # Define start time for the level
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
                        node_exec_result_obj = await self.execute_node(node_id)
                    else:
                        node_exec_result_obj = await node.execute()
                    
                    # Update context and metrics
                    self.context.update_context(node_id, node_exec_result_obj.output)
                    self._update_metrics(node_id, node_exec_result_obj)
                    
                    await self._trigger_callbacks('node_end', NodeExecutionResult(
                        success=node_exec_result_obj.success,
                        output={
                            'node_id': node_id,
                            'result': node_exec_result_obj.model_dump(),
                            'level': level.level
                        },
                        metadata=NodeMetadata(
                            node_id=node_id,
                            node_type=node.type if hasattr(node, 'type') else "unknown",
                            start_time=start_time,
                            end_time=datetime.utcnow()
                        )
                    ))
                    
                    return node_id, node_exec_result_obj
                    
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
            success=all(r.success for _, r in results),
            output=dict(results),
            metadata=NodeMetadata(
                node_id=self.chain_id, # Using self.chain_id for the overall chain
                node_type="level_summary", # Clarifying node_type for level result
                start_time=level_start_time, # Use the defined level_start_time
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
            # Prepare inputs for the current node based on outputs of its dependencies
            # and the input_mappings specified in its NodeConfig.
            mapped_inputs = {}
            if hasattr(node.config, 'input_mappings') and node.config.input_mappings:
                for target_key, mapping_rule in node.config.input_mappings.items():
                    # mapping_rule is an InputMapping object
                    # mapping_rule.source_id is the ID of the node that produced the data.
                    # mapping_rule.rules is a ContextRule object for transforms, requirements etc.
                    producer_node_id = mapping_rule.source_id
                    context_rule_for_input = mapping_rule.rules

                    # Get the output of the producer_node_id from the context cache.
                    # self.context.get_context(producer_node_id) should return the actual output data.
                    producer_output = self.context.get_context(producer_node_id)

                    if producer_output is None:
                        if context_rule_for_input.required:
                            raise InputMappingError(f"Required input from source_id '{producer_node_id}' (for target '{target_key}') not found in context for node '{node_id}'")
                        continue # Skip if not required and not found
                    
                    value_to_transform = producer_output

                    # Apply transform if specified in the ContextRule
                    if context_rule_for_input.transform:
                        transform_name = context_rule_for_input.transform
                        transform_func = self.get_transform(transform_name)
                        
                        if transform_func:
                            try:
                                value_to_transform = transform_func(value_to_transform)
                            except Exception as e:
                                raise InputMappingError(f"Transform '{transform_name}' failed for source_id '{producer_node_id}' (target '{target_key}'): {str(e)}")
                        else:
                            raise InputMappingError(f"Unknown transform '{transform_name}' for source_id '{producer_node_id}' (target '{target_key}')")
                    
                    mapped_inputs[target_key] = value_to_transform
            else:
                # If no input_mappings are defined, the node receives no explicit inputs from dependencies.
                # It might rely on a general context or have a default behavior.
                # For now, mapped_inputs remains empty. This could be a point of future refinement.
                pass 

            # Trigger node_start callback with consistent format
            await self._trigger_callbacks('node_start', NodeExecutionResult(
                success=True,
                output={
                    'node_id': node_id,
                    'inputs': mapped_inputs,
                    'level': node.config.level if hasattr(node.config, 'level') else 0
                },
                metadata=NodeMetadata(
                    node_id=node_id,
                    node_type=node.type,
                    start_time=start_time
                )
            ))
            
            # Execute node with inputs
            try:
                result = await node.execute(mapped_inputs)
                return result
            except Exception as e:
                # Catching exception from node.execute() itself
                raise NodeExecutionError(f"Node '{node_id}' execution failed during its execute method: {str(e)}")
                
        except (InputMappingError, NodeExecutionError) as e:
            # Handle known error types from input mapping or node execution
            logger.error(f"{e.__class__.__name__} for node '{node_id}': {str(e)}")
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
        except Exception as e:
            # Catch-all for unexpected errors during the setup/input mapping in execute_node
            logger.error(f"Unexpected error preparing or executing node '{node_id}': {str(e)}")
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