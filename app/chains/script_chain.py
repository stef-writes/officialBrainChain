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
        vector_store_config: Optional[VectorStoreConfig] = None,
        llm_config: Optional[Union[Dict[str, Any], LLMConfig]] = None
    ):
        """Initialize the script chain with enhanced configuration"""
        self.chain_id = f"chain_{uuid4().hex[:8]}"
        self.graph = nx.DiGraph()
        self.dependencies = {}  # Map of node_id to list of dependencies
        self.execution_levels = {}  # Map of level to ExecutionLevel
        self.node_registry: Dict[str, BaseNode] = {}
        self.callbacks = callbacks or []
        self.max_context_tokens = max_context_tokens
        
        # Convert llm_config dict to LLMConfig instance if needed
        if isinstance(llm_config, dict):
            default_llm_config = {
                "model": "gpt-4",
                "api_key": os.getenv("OPENAI_API_KEY"),
                "max_context_tokens": max_context_tokens,
                "temperature": 0.7,
                "max_tokens": 500
            }
            default_llm_config.update(llm_config)
            self.llm_config = LLMConfig(**default_llm_config)
        else:
            self.llm_config = llm_config or LLMConfig(
                model="gpt-4",
                api_key=os.getenv("OPENAI_API_KEY"),
                max_context_tokens=max_context_tokens,
                temperature=0.7,
                max_tokens=500
            )
        
        # Initialize vector store if config is provided
        self.vector_store_config = vector_store_config
        self.vector_store = PineconeVectorStore(vector_store_config) if vector_store_config else None
        
        # Custom context manager using graph structure
        self.context = GraphContextManager(
            max_tokens=max_context_tokens,
            graph=self.graph,
            vector_store=self.vector_store
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
        if node_config.id in self.node_registry:
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
        for node_id in self.node_registry:
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
            # Trigger chain_start callback
            await self._trigger_callbacks('chain_start', NodeExecutionResult(
                success=True,
                output={},
                metadata=NodeMetadata(
                    node_id=self.chain_id,
                    node_type="chain",
                    start_time=start_time
                )
            ))
            
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
                    error_result = NodeExecutionResult(
                        success=False,
                        error=error_msg,
                        output=results,
                        metadata=NodeMetadata(
                            node_id=self.chain_id,
                            node_type="chain",
                            start_time=start_time,
                            end_time=datetime.utcnow(),
                            error_type="LevelExecutionError"
                        )
                    )
                    
                    # Trigger chain_end callback with error
                    await self._trigger_callbacks('chain_end', error_result)
                    return error_result
            
            # All levels completed successfully
            success_result = NodeExecutionResult(
                success=True,
                output=results,
                metadata=NodeMetadata(
                    node_id=self.chain_id,
                    node_type="chain",
                    start_time=start_time,
                    end_time=datetime.utcnow()
                )
            )
            
            # Trigger chain_end callback with success
            await self._trigger_callbacks('chain_end', success_result)
            return success_result
            
        except Exception as e:
            logger.error(f"Chain execution failed: {str(e)}")
            error_result = NodeExecutionResult(
                success=False,
                error=str(e),
                output=None,
                metadata=NodeMetadata(
                    node_id=self.chain_id,
                    node_type="chain",
                    start_time=start_time,
                    end_time=datetime.utcnow(),
                    error_type=e.__class__.__name__
                )
            )
            
            # Trigger chain_end callback with error
            await self._trigger_callbacks('chain_end', error_result)
            return error_result

    async def _process_level(self, level: ExecutionLevel) -> NodeExecutionResult:
        """Process all nodes in a level with controlled concurrency"""
        semaphore = asyncio.Semaphore(self.concurrency_level)
        start_time = datetime.utcnow()
        
        async def process_node(node_id: str) -> Tuple[str, NodeExecutionResult]:
            async with semaphore:
                node_start_time = datetime.utcnow()
                max_retries = 3

                try:
                    node = self.node_registry[node_id]
                    if not hasattr(node, 'execute'):
                        logger.error(f"Node {node_id} does not have execute method")
                        return node_id, NodeExecutionResult(
                            success=False,
                            error=f"Node {node_id} is not properly initialized or lacks execute method",
                            metadata=NodeMetadata(
                                node_id=node_id,
                                node_type=getattr(node, 'type', 'unknown'),
                                start_time=node_start_time,
                                end_time=datetime.utcnow(),
                                error_type="InvalidNodeError"
                            )
                        )

                    # Get the node's own previous context if it exists 
                    node_context_data = await self.context.get_context(node_id)
                    
                    # Prepare the execution context
                    # This will be enriched with dependency outputs in execute_node
                    node_context = node_context_data or {}
                    
                    # Now execute the node with all the necessary context
                    # The execute_node method will handle dependency resolution
                    result = await self.execute_node(node_id, node_context=node_context, max_retries=max_retries)

                    if result.success:
                        # Update the context with this node's result
                        self.context.update_context(node_id, result)
                        self._update_metrics(node_id, result)

                        await self._trigger_callbacks('node_end', NodeExecutionResult(
                            success=True,
                            output={
                                'node_id': node_id,
                                'result': result.model_dump(),
                                'level': level.level
                            },
                            metadata=result.metadata
                        ))
                        return node_id, result
                    else:
                        await self._trigger_callbacks('node_error', result)
                        return node_id, result

                except KeyError:
                    return node_id, NodeExecutionResult(
                        success=False,
                        error=f"Node {node_id} not found in registry",
                        metadata=NodeMetadata(
                            node_id=node_id,
                            node_type="unknown",
                            start_time=node_start_time,
                            end_time=datetime.utcnow(),
                            error_type="NodeNotFoundError"
                        )
                    )
                except Exception as e:
                    logger.error(f"Unexpected error processing node {node_id} wrapper: {str(e)}", exc_info=True)
                    node_type = "unknown"
                    try:
                        if node_id in self.node_registry:
                            node_type = getattr(self.node_registry[node_id], 'type', 'unknown')
                    except Exception:
                        pass

                    error_result = NodeExecutionResult(
                        success=False,
                        error=f"Unexpected error in chain processing for node {node_id}: {str(e)}",
                        metadata=NodeMetadata(
                            node_id=node_id,
                            node_type=node_type,
                            start_time=node_start_time,
                            end_time=datetime.utcnow(),
                            error_type=e.__class__.__name__,
                            error_traceback=traceback.format_exc()
                        )
                    )
                    await self._trigger_callbacks('node_error', error_result)
                    return node_id, error_result

        # Execute all nodes in this level in parallel (respecting concurrency limit)
        tasks = [process_node(node_id) for node_id in level.node_ids]
        results = await asyncio.gather(*tasks)
        results_dict = dict(results)
        
        return NodeExecutionResult(
            success=all(r.success for r in results_dict.values()),
            output=results_dict,
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

    async def execute_node(self, node_id: str, node_context: Optional[Dict[str, Any]] = None, max_retries: int = 3) -> NodeExecutionResult:
        """Execute a single node with retry logic.
        
        Args:
            node_id: ID of the node to execute
            node_context: The context dictionary for the node execution.
            max_retries: Maximum number of retry attempts
            
        Returns:
            NodeExecutionResult containing the execution result
        """
        if node_id not in self.node_registry:
            return NodeExecutionResult(
                success=False,
                error=f"Node {node_id} not found in registry",
                metadata=NodeMetadata(
                    node_id=node_id,
                    node_type="unknown",
                    error_type="NodeNotFoundError"
                )
            )

        node = self.node_registry[node_id]
        node_config = node.config
        start_time = datetime.utcnow()
        attempt = 0

        # Initialize node execution context
        execution_context = node_context or {}
        
        # Resolve dependencies and apply input mappings if defined
        if hasattr(node_config, 'input_mappings') and node_config.input_mappings:
            for target_key, mapping in node_config.input_mappings.items():
                source_id = mapping.source_id
                # Get the source node's execution result from context
                source_context = await self.context.get_context(source_id)
                if source_context and source_context.get('output'):
                    source_output = source_context['output']
                    
                    # Apply any transformation rules if defined
                    if mapping.rules and hasattr(mapping.rules, 'transform') and mapping.rules.transform:
                        # For now, simple key extraction from the output
                        # This could be extended with more complex transformations
                        if mapping.rules.transform in source_output:
                            execution_context[target_key] = source_output[mapping.rules.transform]
                    else:
                        # If no specific transform is defined, use the entire output
                        execution_context[target_key] = source_output
                        
                    logger.debug(f"Applied input mapping from {source_id} to {target_key} for node {node_id}")
                elif mapping.rules and mapping.rules.required:
                    # If the required dependency isn't available, log warning
                    logger.warning(f"Required dependency {source_id} not found for node {node_id}")
        
        # For nodes that define dependencies but not explicit mappings,
        # make dependency outputs available in the context under their node IDs
        if hasattr(node_config, 'dependencies') and node_config.dependencies:
            for dep_id in node_config.dependencies:
                if dep_id not in execution_context:  # Don't override explicit mappings
                    dep_context = await self.context.get_context(dep_id)
                    if dep_context and dep_context.get('output'):
                        # Make the dependency's output available under its node ID
                        execution_context[dep_id] = dep_context['output']
                        logger.debug(f"Added dependency {dep_id} output to context for node {node_id}")

        # Now execute the node with the enhanced context
        while attempt < max_retries:
            try:
                result = await node.execute(execution_context)
                if result.success:
                    if result.metadata and result.metadata.start_time is None:
                        result.metadata.start_time = start_time
                    if result.metadata and result.metadata.end_time is None:
                        result.metadata.end_time = datetime.utcnow()
                    
                    # Store the context used for this execution in the result
                    result.context_used = execution_context
                    return result

                error_reason = result.error or f"Node {node_id} returned success=False"
                logger.warning(f"Node {node_id} execution attempt {attempt + 1}/{max_retries} returned success=False: {error_reason}")
                attempt += 1
                if attempt < max_retries:
                    await asyncio.sleep(1 * attempt)
                else:
                    return result

            except Exception as e:
                logger.error(f"Node {node_id} execution failed (attempt {attempt + 1}/{max_retries}): {str(e)}")
                attempt += 1
                if attempt < max_retries:
                    await asyncio.sleep(1 * attempt)
                else:
                    error_traceback = traceback.format_exc()
                    return NodeExecutionResult(
                        success=False,
                        error=str(e),
                        metadata=NodeMetadata(
                            node_id=node_id,
                            node_type=getattr(node, 'type', 'unknown'),
                            start_time=start_time,
                            end_time=datetime.utcnow(),
                            error_type=e.__class__.__name__,
                            error_traceback=error_traceback
                        ),
                        context_used=execution_context
                    )

        return NodeExecutionResult(
            success=False,
            error=f"Node {node_id} failed after {max_retries} attempts (reached end of retry loop)",
            metadata=NodeMetadata(
                node_id=node_id,
                node_type=getattr(node, 'type', 'unknown'),
                start_time=start_time,
                end_time=datetime.utcnow(),
                error_type="MaxRetriesExceeded"
            ),
            context_used=execution_context
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
            
            # Check dependencies using node_registry
            node = self.node_registry.get(node_id)
            if node and hasattr(node, 'config') and node.config.dependencies:
                for dep in node.config.dependencies:
                    # Ensure dependency exists in registry before recursion
                    if dep in self.node_registry:
                        dfs(dep, component)

            # Check nodes that depend on this one using node_registry
            for other_id, other_node in self.node_registry.items():
                # Check config/dependencies exist before accessing
                if hasattr(other_node, 'config') and node_id in other_node.config.dependencies:
                    dfs(other_id, component)
        
        # Find all components by iterating node_registry
        for node_id in self.node_registry:
            if node_id not in visited:
                component = set()
                dfs(node_id, component)
                components.append(component)
                
        return components