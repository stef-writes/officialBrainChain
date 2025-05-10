"""
API routes for the workflow engine
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from app.models.node_models import NodeConfig, NodeExecutionResult
from app.models.config import LLMConfig, MessageTemplate
from app.nodes.text_generation import TextGenerationNode
from app.chains.script_chain import ScriptChain
from app.utils.context import GraphContextManager

router = APIRouter()

class NodeRequest(BaseModel):
    """Request model for node operations"""
    config: NodeConfig
    context: Optional[Dict[str, Any]] = None

class ChainRequest(BaseModel):
    """Request model for chain operations"""
    nodes: List[NodeConfig]
    context: Optional[Dict[str, Any]] = None

@router.post("/nodes/text-generation", response_model=NodeExecutionResult)
async def create_text_generation_node(request: NodeRequest):
    """Create and execute a text generation node"""
    try:
        context_manager = GraphContextManager()
        node = TextGenerationNode(request.config, context_manager)
        result = await node.execute(request.context or {})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chains/execute", response_model=NodeExecutionResult)
async def execute_chain(request: ChainRequest):
    """Execute a chain of nodes"""
    try:
        # The GraphContextManager instance created here is not directly used by ScriptChain anymore.
        # ScriptChain now internally manages its own context manager.
        # However, if nodes were to be created *outside* the chain and then added, they might need one.
        # For this route, ScriptChain.add_node(NodeConfig) is used, which handles node creation internally.
        
        chain = ScriptChain() # Initialize without the unexpected context_manager argument
        
        # Add nodes to chain
        for node_config in request.nodes:
            if node_config.metadata.node_type == "ai":
                # ScriptChain.add_node now accepts NodeConfig and creates the node internally
                # using its own context manager and LLM config.
                chain.add_node(node_config)
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported node type: {node_config.metadata.node_type}"
                )
        
        # Execute chain
        result = await chain.execute()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/nodes/{node_id}/context")
async def get_node_context(node_id: str):
    """Get context for a specific node"""
    try:
        context_manager = GraphContextManager()
        context = context_manager.get_context(node_id)
        if context is None:
            raise HTTPException(status_code=404, detail=f"Context not found for node {node_id}")
        return context
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/nodes/{node_id}/context")
async def clear_node_context(node_id: str):
    """Clear context for a specific node"""
    try:
        context_manager = GraphContextManager()
        context_manager.set_context(node_id, {})  # Clear context by setting empty dict
        return {"message": f"Context cleared for node {node_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))