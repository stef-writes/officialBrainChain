"""
Base node implementation defining the interface for all nodes
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from app.models.node_models import NodeConfig, NodeExecutionResult

class BaseNode(ABC):
    """Abstract base class for all nodes"""
    
    def __init__(self, config: NodeConfig):
        self.config = config
        
    @property
    def node_id(self) -> str:
        return self.config.metadata.node_id
    
    @property
    def node_type(self) -> str:
        return self.config.metadata.node_type
    
    @abstractmethod
    async def execute(self, context: Dict[str, Any]) -> NodeExecutionResult:
        """Execute node logic (abstract method)"""
        pass