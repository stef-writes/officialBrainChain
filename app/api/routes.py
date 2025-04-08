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
from app.utils.context import ContextManager

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
        node = TextGenerationNode(request.config)
        result = await node.execute(request.context or {})
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/chains/execute", response_model=NodeExecutionResult)
async def execute_chain(request: ChainRequest):
    """Execute a chain of nodes"""
    try:
        chain = ScriptChain()
        
        # Add nodes to chain
        for node_config in request.nodes:
            if node_config.metadata.node_type == "ai":
                node = TextGenerationNode(node_config)
                chain.add_node(node)
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
        context_manager = ContextManager()
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
        context_manager = ContextManager()
        context_manager.clear_context(node_id)
        return {"message": f"Context cleared for node {node_id}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))