"""
Logic node implementations
"""

from typing import Dict, Any
from datetime import datetime

from app.nodes.base import BaseNode
from app.models.nodes import NodeConfig, NodeExecutionResult
from app.utils.logging import logger

class DecisionNode(BaseNode):
    """Node for making logical decisions"""
    
    def __init__(self):
        config = NodeConfig(
            metadata={
                "node_id": "decision",
                "node_type": "logic",
                "version": "1.0.0",
                "description": "Logical decision making"
            }
        )
        super().__init__(config)
    
    async def execute(self, context: Dict[str, Any]) -> NodeExecutionResult:
        """Execute decision logic"""
        try:
            start_time = datetime.now()
            
            # Extract decision parameters
            conditions = context.get("conditions", {})
            if not conditions:
                raise ValueError("No conditions provided for decision making")
            
            # Apply decision logic
            decision = self._evaluate_conditions(conditions)
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            return NodeExecutionResult(
                success=True,
                output={"decision": decision},
                metadata={
                    "duration": duration,
                    "conditions_evaluated": len(conditions)
                }
            )
            
        except Exception as e:
            logger.error(f"Error in decision making: {str(e)}")
            return NodeExecutionResult(
                success=False,
                error=str(e),
                metadata={
                    "error_type": type(e).__name__,
                    "timestamp": datetime.now().isoformat()
                }
            )
    
    def _evaluate_conditions(self, conditions: Dict[str, Any]) -> bool:
        """Evaluate decision conditions"""
        # Simple example - can be expanded based on needs
        return all(conditions.values()) 