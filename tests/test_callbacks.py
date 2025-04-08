import pytest
from app.chains.script_chain import ScriptChain
from app.utils.callbacks import ScriptChainCallback
from uuid import uuid4
from app.models.node_models import NodeConfig, NodeExecutionResult
import logging

logger = logging.getLogger(__name__)

class TestCallback(ScriptChainCallback):
    def __init__(self):
        self.events = []
    
    async def on_chain_start(self, chain_id, config):
        self.events.append(("chain_start", chain_id, config))
    
    async def on_node_start(self, node_id, config):
        self.events.append(("node_start", node_id))
    
    async def on_node_complete(self, node_id, result):
        self.events.append(("node_complete", node_id, result.success))
    
    async def on_chain_end(self, chain_id, result):
        self.events.append(("chain_end", result["success"]))

    async def on_context_update(self, node_id, context):
        pass

    async def on_node_error(self, node_id, error):
        pass

    async def on_vector_store_op(self, operation, details):
        pass

class TestNode:
    def __init__(self, success=True):
        self.success = success
        self.node_id = f"node_{uuid4().hex[:8]}"
        self.config = NodeConfig(dependencies=[], metadata={'node_id': 'test_node', 'node_type': 'test_type'}, llm_config={'api_key': 'sk-test-1234567890abcdef1234567890abcdef1234567890'})

    async def execute(self, context):
        logger.debug(f"Executing TestNode {self.node_id} with context: {context}")
        return NodeExecutionResult(success=self.success)

@pytest.mark.asyncio
async def test_full_callback_lifecycle():
    callback = TestCallback()
    chain = ScriptChain(callbacks=[callback])
    
    # Add test nodes
    chain.add_node(TestNode())
    chain.add_node(TestNode())
    
    await chain.execute()
    
    assert len(callback.events) >= 4
    assert callback.events[0][0] == "chain_start"
    assert any(e[0] == "node_start" for e in callback.events)
    assert any(e[0] == "node_complete" for e in callback.events)
    assert callback.events[-1][0] == "chain_end" 