"""
Gaffer - AI Workflow Orchestration System
"""

from app.main import app
from app.chains.script_chain import ScriptChain

__version__ = "0.1.0"

__all__ = [
    "app",
    "ScriptChain"
]
