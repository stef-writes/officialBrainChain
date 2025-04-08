import pytest
from pathlib import Path
from app.context.vector import VectorStore
from app.utils.context import ContextManager

@pytest.mark.asyncio
async def test_vector_storage_and_retrieval():
    # Initialize fresh vector store
    vector = VectorStore(storage_path=Path("/tmp/test_vector.json"))
    
    # Store test context
    vector.add_context("llm workflow orchestration", {"node_id": "test1"})
    vector.add_context("distributed task processing", {"node_id": "test2"})
    
    # Query related concept
    results = vector.find_similar("AI pipeline management", top_k=2)
    
    assert len(results) == 2
    assert any("llm workflow" in ctx["text"] for ctx in results)
    assert any("distributed task" in ctx["text"] for ctx in results)

@pytest.mark.asyncio
async def test_vector_context_injection():
    context_manager = ContextManager()
    
    # Store initial context
    context_manager.set_context("node1", {"output": "AI pipeline config"})
    
    # Get optimized context for new node
    ctx = context_manager.get_context_with_optimization("node2")
    
    assert "vector_node1" in ctx
    assert "AI pipeline" in ctx["vector_node1"] 