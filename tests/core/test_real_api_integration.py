"""
Integration tests using real OpenAI API calls
"""

import pytest
import os
import asyncio
from typing import Dict, Any
from app.chains.script_chain import ScriptChain
from app.models.config import LLMConfig
from app.nodes.text_generation import TextGenerationNode
from app.models.node_models import NodeConfig, NodeMetadata, NodeExecutionResult

def get_api_key() -> str:
    """Get OpenAI API key from environment"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set")
    return api_key

@pytest.mark.asyncio
async def test_real_chain_execution():
    """Test chain execution with real API calls"""
    api_key = get_api_key()
    
    # Create nodes with real API key
    node1 = TextGenerationNode.create(LLMConfig(
        api_key=api_key,
        model="gpt-4",  # Using GPT-4 as 3.5 is deprecated
        temperature=0.7,
        max_tokens=100
    ))
    
    node2 = TextGenerationNode.create(LLMConfig(
        api_key=api_key,
        model="gpt-4",
        temperature=0.5,
        max_tokens=150
    ))
    
    # Create a script chain
    chain = ScriptChain()
    chain.add_node(node1)
    chain.add_node(node2)
    
    # Add edge to create dependency
    chain.add_edge(node1.node_id, node2.node_id)
    
    # Set context for nodes
    chain.context.set_context(node1.node_id, {
        "prompt": "Write a one-sentence story about a cat."
    })
    chain.context.set_context(node2.node_id, {
        "prompt": "Continue the story with one more sentence about what happens next."
    })
    
    # Execute chain
    result = await chain.execute()
    
    # Verify successful execution
    assert result.success, f"Chain execution failed: {result.error}"
    assert isinstance(result.output, str)
    assert len(result.output) > 0
    assert result.metadata is not None
    assert result.duration > 0
    assert result.usage is not None
    assert result.usage.total_tokens > 0
    
    # Verify context was updated
    node1_output = chain.context.get_context(node1.node_id).get("output")
    assert node1_output and isinstance(node1_output, str)
    assert "cat" in node1_output.lower()

@pytest.mark.asyncio
async def test_real_chain_with_system_prompt():
    """Test chain execution with system prompts"""
    api_key = get_api_key()
    
    # Create node with system prompt
    config = NodeConfig(
        metadata=NodeMetadata(
            node_id="story_generator",
            node_type="ai",
            version="1.0.0",
            description="Story generation with style"
        ),
        llm_config=LLMConfig(
            api_key=api_key,
            model="gpt-4",
            temperature=0.7,
            max_tokens=150
        ),
        templates=[{
            "role": "system",
            "content": "You are a creative writer who specializes in children's stories."
        }]
    )
    
    node = TextGenerationNode(config)
    
    # Create chain
    chain = ScriptChain()
    chain.add_node(node)
    
    # Set context
    chain.context.set_context(node.node_id, {
        "prompt": "Write a short story about friendship between a cat and a dog."
    })
    
    # Execute chain
    result = await chain.execute()
    
    # Verify successful execution
    assert result.success, f"Chain execution failed: {result.error}"
    assert isinstance(result.output, str)
    assert len(result.output) > 0
    assert "cat" in result.output.lower() and "dog" in result.output.lower()
    assert result.metadata is not None
    assert result.duration > 0
    assert result.usage is not None
    assert result.usage.total_tokens > 0

@pytest.mark.asyncio
async def test_real_chain_concurrent_execution():
    """Test concurrent execution of multiple chains with real API calls"""
    api_key = get_api_key()
    
    async def create_and_execute_chain(prompt: str) -> NodeExecutionResult:
        node = TextGenerationNode.create(LLMConfig(
            api_key=api_key,
            model="gpt-4",
            temperature=0.7,
            max_tokens=100
        ))
        
        chain = ScriptChain()
        chain.add_node(node)
        chain.context.set_context(node.node_id, {"prompt": prompt})
        return await chain.execute()
    
    # Execute multiple chains concurrently
    prompts = [
        "Write a haiku about spring.",
        "Write a haiku about summer.",
        "Write a haiku about autumn.",
        "Write a haiku about winter."
    ]
    
    tasks = [create_and_execute_chain(prompt) for prompt in prompts]
    results = await asyncio.gather(*tasks)
    
    # Verify all executions were successful
    for result in results:
        assert result.success, f"Chain execution failed: {result.error}"
        assert isinstance(result.output, str)
        assert len(result.output) > 0
        assert result.metadata is not None
        assert result.duration > 0
        assert result.usage is not None
        assert result.usage.total_tokens > 0 