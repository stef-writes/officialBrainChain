"""
Node implementations for the workflow engine
"""

from app.nodes.base import BaseNode
from app.nodes.text_generation import TextGenerationNode

__all__ = [
    "BaseNode",
    "TextGenerationNode"
]
