import pytest
from app.script_chain import ScriptChain
from app.utils.callbacks import ScriptChainCallback

class TestNode:
    def __init__(self, success=True):
        self.success = success

    async def execute(self, context):
        return NodeExecutionResult(success=self.success)

@pytest.mark.asyncio
async def test_batch_execution_order():
    # Create 10 independent nodes
    nodes = [TestNode() for _ in range(10)]
    
    # Configure chain with batch size 3
    chain = ScriptChain(batch_size=3)
    for node in nodes:
        chain.add_node(node)
    
    execution_order = chain._get_execution_order()
    
    # Verify batches preserve order
    batches = list(chain._chunked(execution_order, 3))
    assert len(batches) == 4  # 10/3 = 4 batches
    assert batches == [
        [0,1,2], [3,4,5], [6,7,8], [9]
    ]

@pytest.mark.asyncio
async def test_batch_error_handling():
    # Create chain with failing node
    chain = ScriptChain(batch_size=2)
    chain.add_node(TestNode(success=True))
    chain.add_node(TestNode(success=False))  # Failing node
    
    result = await chain.execute()
    
    assert not result["success"]
    assert len(result["node_results"]) == 2  # Only first batch processed 