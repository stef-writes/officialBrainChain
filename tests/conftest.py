"""
Test configuration and fixtures for the Gaffer test suite.
"""

import pytest
import asyncio
import networkx as nx
from typing import Dict, Any, Generator
from app.chains.script_chain import ScriptChain
from app.utils.context import GraphContextManager
from app.utils.callbacks import LoggingCallback, MetricsCallback, DebugCallback
from app.models.node_models import NodeConfig
from app.context.vector import VectorStore

@pytest.fixture
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_vector_store() -> VectorStore:
    """Create a mock vector store for testing."""
    return VectorStore(
        index_name="test-index",
        dimension=1536,
        metric="cosine"
    )

@pytest.fixture
def test_graph() -> nx.DiGraph:
    """Create a test workflow graph."""
    graph = nx.DiGraph()
    graph.add_edge("node1", "node2")
    graph.add_edge("node2", "node3")
    return graph

@pytest.fixture
def script_chain() -> ScriptChain:
    """Create a test ScriptChain instance."""
    return ScriptChain(
        concurrency_level=2,
        retry_policy={
            'max_retries': 2,
            'delay': 0.1,
            'backoff': 1.5
        }
    )

@pytest.fixture
def context_manager(test_graph: nx.DiGraph, mock_vector_store: VectorStore) -> GraphContextManager:
    """Create a test GraphContextManager instance."""
    return GraphContextManager(
        max_tokens=1000,
        graph=test_graph,
        vector_store=mock_vector_store
    )

@pytest.fixture
def test_nodes() -> Dict[str, NodeConfig]:
    """Create test node configurations."""
    return {
        "node1": NodeConfig(
            id="node1",
            type="llm",
            model="gpt-4",
            prompt="Test prompt 1",
            level=0
        ),
        "node2": NodeConfig(
            id="node2",
            type="llm",
            model="gpt-4",
            prompt="Test prompt 2",
            level=1,
            dependencies=["node1"]
        ),
        "node3": NodeConfig(
            id="node3",
            type="llm",
            model="gpt-4",
            prompt="Test prompt 3",
            level=2,
            dependencies=["node2"]
        )
    }

@pytest.fixture
def callbacks() -> Dict[str, Any]:
    """Create test callback instances."""
    return {
        "logging": LoggingCallback(),
        "metrics": MetricsCallback(),
        "debug": DebugCallback()
    } 