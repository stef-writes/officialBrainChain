from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Depends, Body
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import yaml
import json

from app.chains.script_chain import ScriptChain
from app.models.node_models import NodeConfig, NodeExecutionResult, NodeMetadata
from app.models.config import LLMConfig
from app.models.vector_store import VectorStoreConfig

# Configure logger
logger = logging.getLogger(__name__)

router = APIRouter()

# Simple in-memory storage for chain instances.
# NOTE: This is not persistent and will be lost on server restart.
# Consider a more robust storage solution for production.
chains: Dict[str, ScriptChain] = {}
# Store execution results separately
chain_results: Dict[str, NodeExecutionResult] = {}
# Track chain execution status
chain_status: Dict[str, str] = {}

class ChainCreationRequest(BaseModel):
    """Request model for creating a new chain."""
    llm_config: Optional[LLMConfig] = None
    vector_store_config: Optional[VectorStoreConfig] = None

class ChainInfoResponse(BaseModel):
    """Response model for basic chain info."""
    chain_id: str
    nodes: List[str]
    edges: List[Tuple[str, str]] # Use Tuple instead of tuple

class ChainStatusResponse(BaseModel):
    """Response model for chain execution status."""
    chain_id: str
    status: str  # "pending", "running", "completed", "failed"
    message: Optional[str] = None

class NodeDefinition(BaseModel):
    """Node definition for workflow import."""
    id: str
    type: str
    config: Dict[str, Any] = Field(default_factory=dict)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class EdgeDefinition(BaseModel):
    """Edge definition for workflow import."""
    source: str
    target: str
    metadata: Optional[Dict[str, Any]] = None

class WorkflowDefinition(BaseModel):
    """Workflow definition model for importing complete workflows."""
    name: Optional[str] = None
    description: Optional[str] = None
    nodes: List[NodeDefinition]
    edges: List[EdgeDefinition]
    llm_config: Optional[LLMConfig] = None
    vector_store_config: Optional[VectorStoreConfig] = None
    
    @validator('nodes')
    def validate_nodes(cls, nodes):
        # Ensure node IDs are unique
        node_ids = [node.id for node in nodes]
        if len(node_ids) != len(set(node_ids)):
            raise ValueError("Node IDs must be unique")
        return nodes
    
    @validator('edges')
    def validate_edges(cls, edges, values):
        if 'nodes' not in values:
            return edges
            
        # Get all node IDs
        node_ids = [node.id for node in values['nodes']]
        
        # Check that edge source and target exist in nodes
        for edge in edges:
            if edge.source not in node_ids:
                raise ValueError(f"Edge source '{edge.source}' not found in nodes")
            if edge.target not in node_ids:
                raise ValueError(f"Edge target '{edge.target}' not found in nodes")
                
        return edges

@router.post(
    "/chains",
    summary="Create a new ScriptChain instance",
    status_code=status.HTTP_201_CREATED,
    response_model=Dict[str, str]
)
async def create_chain(request: ChainCreationRequest):
    """
    Initializes a new ScriptChain with optional LLM and Vector Store configurations.
    Returns the unique ID of the created chain.
    """
    try:
        # TODO: Handle max_context_tokens more dynamically if needed
        chain = ScriptChain(
            llm_config=request.llm_config,
            vector_store_config=request.vector_store_config
        )
        chains[chain.chain_id] = chain
        chain_status[chain.chain_id] = "pending"
        logger.info(f"Created new chain with ID: {chain.chain_id}")
        return {"chain_id": chain.chain_id}
    except Exception as e:
        logger.error(f"Failed to create chain: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create chain: {str(e)}"
        )

@router.post(
    "/chains/{chain_id}/nodes",
    summary="Add a node to a specific chain",
    status_code=status.HTTP_201_CREATED,
    response_model=Dict[str, str]
)
async def add_node_to_chain(chain_id: str, node_config: NodeConfig):
    """
    Adds a new node (defined by NodeConfig) to the specified chain.
    """
    if chain_id not in chains:
        logger.warning(f"Attempt to add node to non-existent chain: {chain_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chain not found")
    
    chain = chains[chain_id]
    try:
        chain.add_node(node_config)
        logger.info(f"Added node {node_config.id} to chain {chain_id}")
        return {"message": f"Node {node_config.id} added to chain {chain_id}"}
    except ValueError as e:
        logger.warning(f"Invalid node configuration for {node_config.id}: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to add node {node_config.id} to chain {chain_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add node: {str(e)}"
        )

async def run_chain_execution(chain_id: str, chain: ScriptChain):
    """Helper function to run chain execution and store results."""
    try:
        logger.info(f"Starting execution of chain {chain_id}")
        chain_status[chain_id] = "running"
        result = await chain.execute()
        chain_results[chain_id] = result
        chain_status[chain_id] = "completed"
        logger.info(f"Chain {chain_id} execution completed successfully")
    except Exception as e:
        logger.error(f"Chain {chain_id} execution failed in background: {e}", exc_info=True)
        # Store an error result
        chain_results[chain_id] = NodeExecutionResult(
            success=False,
            error=f"Chain execution failed: {str(e)}",
            metadata=NodeMetadata(node_id=chain_id, node_type="chain", error_type=e.__class__.__name__)
        )
        chain_status[chain_id] = "failed"

@router.post(
    "/chains/{chain_id}/execute",
    summary="Execute the chain",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=Dict[str, str]
)
async def execute_chain(chain_id: str, background_tasks: BackgroundTasks):
    """
    Triggers the execution of the specified chain in the background.
    Returns immediately with a message indicating execution has started.
    Check the results endpoint for completion status and output.
    """
    if chain_id not in chains:
        logger.warning(f"Attempt to execute non-existent chain: {chain_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chain not found")
    
    chain = chains[chain_id]
    
    # Add execution to background tasks
    background_tasks.add_task(run_chain_execution, chain_id, chain)
    
    # Clear previous results for this chain if any
    if chain_id in chain_results:
        del chain_results[chain_id]
    
    # Update status
    chain_status[chain_id] = "pending"
    logger.info(f"Chain {chain_id} execution queued in background")
        
    return {"message": f"Chain {chain_id} execution started in background."}


@router.get(
    "/chains/{chain_id}",
    summary="Get chain definition",
    response_model=ChainInfoResponse
)
async def get_chain_info(chain_id: str):
    """
    Retrieves the basic structure (nodes and edges) of the specified chain.
    """
    if chain_id not in chains:
        logger.warning(f"Attempt to get info for non-existent chain: {chain_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chain not found")
    
    chain = chains[chain_id]
    try:
        nodes = list(chain.graph.nodes)
        # Ensure edges are represented as lists of tuples for JSON serialization if needed
        edges = [tuple(edge) for edge in chain.graph.edges] 
        logger.info(f"Retrieved info for chain {chain_id}")
        return ChainInfoResponse(chain_id=chain_id, nodes=nodes, edges=edges)
    except Exception as e:
        logger.error(f"Failed to get info for chain {chain_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve chain info: {str(e)}"
        )

@router.get(
    "/chains/{chain_id}/results",
    summary="Get chain execution results",
    response_model=NodeExecutionResult
)
async def get_chain_results(chain_id: str):
    """
    Retrieves the results of a completed chain execution.
    If the execution is still in progress or failed to start,
    it will return an appropriate status or error.
    """
    if chain_id not in chains:
        logger.warning(f"Attempt to get results for non-existent chain: {chain_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chain not found")
        
    if chain_id not in chain_results:
        logger.info(f"Results for chain {chain_id} not yet available")
        # Check if the chain exists but results are not ready
        # This could mean it's still running or hasn't been executed yet.
        # For simplicity, we return a 404, but a 202 Accepted or specific status might be better.
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Results not available for this chain (may be running or not executed).")

    logger.info(f"Retrieved results for chain {chain_id}")
    return chain_results[chain_id]

@router.get(
    "/chains/{chain_id}/status",
    summary="Get chain execution status",
    response_model=ChainStatusResponse
)
async def get_chain_status(chain_id: str):
    """
    Returns the current execution status of the chain.
    """
    if chain_id not in chains:
        logger.warning(f"Attempt to get status for non-existent chain: {chain_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chain not found")
    
    status_value = chain_status.get(chain_id, "pending")
    message = None
    
    # Add informative message based on status
    if status_value == "failed" and chain_id in chain_results:
        message = chain_results[chain_id].error
    
    logger.info(f"Retrieved status for chain {chain_id}: {status_value}")
    return ChainStatusResponse(chain_id=chain_id, status=status_value, message=message)

@router.get(
    "/chains/{chain_id}/results/{node_id}",
    summary="Get results for a specific node in the chain",
    response_model=Optional[Any]
)
async def get_node_results(chain_id: str, node_id: str):
    """
    Retrieves the results for a specific node in the chain, if available.
    """
    if chain_id not in chains:
        logger.warning(f"Attempt to get node results for non-existent chain: {chain_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chain not found")
    
    if chain_id not in chain_results:
        logger.info(f"Results for chain {chain_id} not yet available")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chain results not available")
    
    # Get the chain result
    result = chain_results[chain_id]
    
    # Check if node results are available
    if not hasattr(result, 'node_results') or not result.node_results or node_id not in result.node_results:
        logger.warning(f"Node {node_id} results not found in chain {chain_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Results for node {node_id} not found")
    
    logger.info(f"Retrieved results for node {node_id} in chain {chain_id}")
    return result.node_results[node_id]

@router.delete(
    "/chains/{chain_id}",
    summary="Delete a chain",
    status_code=status.HTTP_204_NO_CONTENT
)
async def delete_chain(chain_id: str):
    """
    Deletes a chain and its associated resources.
    """
    if chain_id not in chains:
        logger.warning(f"Attempt to delete non-existent chain: {chain_id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Chain not found")
    
    # Check if chain is currently running
    if chain_id in chain_status and chain_status[chain_id] == "running":
        logger.warning(f"Attempt to delete currently running chain: {chain_id}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete a chain that is currently running"
        )
    
    # Remove chain and associated data
    del chains[chain_id]
    if chain_id in chain_results:
        del chain_results[chain_id]
    if chain_id in chain_status:
        del chain_status[chain_id]
    
    logger.info(f"Chain {chain_id} successfully deleted")
    return None  # No content for successful DELETE

@router.post(
    "/workflow",
    summary="Import a complete workflow definition",
    status_code=status.HTTP_201_CREATED,
    response_model=Dict[str, str]
)
async def import_workflow(
    workflow: WorkflowDefinition = Body(...),
):
    """
    Imports a complete workflow definition in JSON format.
    Creates a new chain with all specified nodes and edges.
    Returns the ID of the created chain.
    """
    try:
        # Create a new chain
        chain = ScriptChain(
            llm_config=workflow.llm_config,
            vector_store_config=workflow.vector_store_config
        )
        chain_id = chain.chain_id
        
        # Add all nodes first
        logger.info(f"Adding {len(workflow.nodes)} nodes to chain {chain_id}")
        for node_def in workflow.nodes:
            # Convert node definition to NodeConfig
            node_config = NodeConfig(
                id=node_def.id,
                type=node_def.type,
                config=node_def.config,
                inputs=node_def.inputs,
                metadata=node_def.metadata
            )
            try:
                chain.add_node(node_config)
                logger.debug(f"Added node {node_def.id} to chain {chain_id}")
            except Exception as e:
                # Remove the chain if there's an error
                logger.error(f"Failed to add node {node_def.id} to chain {chain_id}: {e}")
                raise ValueError(f"Failed to add node {node_def.id}: {str(e)}")
        
        # Add all edges
        logger.info(f"Adding {len(workflow.edges)} edges to chain {chain_id}")
        for edge in workflow.edges:
            try:
                chain.add_edge(edge.source, edge.target, edge.metadata)
                logger.debug(f"Added edge from {edge.source} to {edge.target}")
            except Exception as e:
                logger.error(f"Failed to add edge from {edge.source} to {edge.target}: {e}")
                raise ValueError(f"Failed to add edge from {edge.source} to {edge.target}: {str(e)}")
        
        # Store the chain
        chains[chain_id] = chain
        chain_status[chain_id] = "pending"
        
        logger.info(f"Successfully imported workflow to chain {chain_id}")
        return {"chain_id": chain_id}
        
    except ValueError as e:
        logger.warning(f"Invalid workflow definition: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to import workflow: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import workflow: {str(e)}"
        )

@router.post(
    "/workflow/yaml",
    summary="Import a complete workflow definition from YAML",
    status_code=status.HTTP_201_CREATED,
    response_model=Dict[str, str]
)
async def import_workflow_yaml(
    yaml_content: str = Body(..., media_type="text/yaml"),
):
    """
    Imports a complete workflow definition in YAML format.
    Creates a new chain with all specified nodes and edges.
    Returns the ID of the created chain.
    """
    try:
        # Parse YAML content to dict
        workflow_dict = yaml.safe_load(yaml_content)
        
        # Convert to WorkflowDefinition
        workflow = WorkflowDefinition.parse_obj(workflow_dict)
        
        # Use the existing JSON workflow importer
        return await import_workflow(workflow)
        
    except yaml.YAMLError as e:
        logger.warning(f"Invalid YAML format: {e}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Invalid YAML format: {str(e)}")
    except Exception as e:
        logger.error(f"Failed to import YAML workflow: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to import YAML workflow: {str(e)}"
        )

# TODO: Implement proper authentication/authorization.
# TODO: Need to add this router to the main FastAPI app instance (e.g., in main.py)
#       app.include_router(chain_router.router, prefix="/api/v1", tags=["Chains"])
