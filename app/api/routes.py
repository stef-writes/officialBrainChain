"""
API routes for the application
"""

from itertools import chain
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import Dict, Any

from app.chains.script_chain import ScriptChain
from app.nodes.ai_nodes import TextGenerationNode
from app.nodes.logic_nodes import DecisionNode
from app.utils.logging import logger

router = APIRouter()

class NodeConfig(BaseModel):
    id: str
    type: str
    config: dict

@router.post("/nodes")
async def create_node(config: NodeConfig):
    """Add node to workflow"""
    node_map = {
        "text_generation": TextGenerationNode,
        "decision": DecisionNode
    }
    
    try:
        node_class = node_map[config.type]
        node = node_class(
            node_id=config.id,
            node_type=config.type,
            **config.config
        )
        chain.add_node(node)
        return {"status": "success"}
    except KeyError:
        raise HTTPException(400, "Invalid node type")

@router.post("/execute")
async def execute_workflow():
    """Execute full workflow"""
    try:
        results = await chain.execute()
        return {
            "results": results,
            "stats": chain.context.get_stats()
        }
    except Exception as e:
        raise HTTPException(500, str(e))

@router.post("/generate")
async def generate_text(request: Request, data: Dict[str, Any]):
    """Generate text using the AI model"""
    try:
        node = TextGenerationNode()
        result = await node.execute(data)
        return result
    except Exception as e:
        logger.error(f"Error generating text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/decide")
async def make_decision(request: Request, data: Dict[str, Any]):
    """Make a decision using the decision node"""
    try:
        node = DecisionNode()
        result = await node.execute(data)
        return result
    except Exception as e:
        logger.error(f"Error making decision: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))