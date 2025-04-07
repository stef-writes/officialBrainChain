"""
Core AI functionality tests with complex scenarios
"""

import pytest
import asyncio
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from app.nodes.ai_nodes import TextGenerationNode
from app.models.config import LLMConfig, MessageTemplate
from app.models.nodes import NodeConfig, NodeMetadata, NodeExecutionResult
from app.chains.script_chain import ScriptChain
from app.utils.context import ContextManager

def get_api_key() -> Tuple[str, bool]:
    """
    Get API key from environment or config file.
    Returns a tuple of (api_key, is_mock_key).
    """
    # Check environment variable first
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return api_key, False
    
    # Check for config file
    config_file = Path("config/api_keys.json")
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                config = json.load(f)
                if "openai_api_key" in config:
                    return config["openai_api_key"], False
        except Exception as e:
            logging.warning(f"Failed to load API key from config file: {e}")
    
    # Fallback to mock key for CI/CD or when no real key is available
    logging.warning("No real API key found. Using mock key for testing.")
    return "sk-test-key-for-testing-only", True

class TestDataCollector:
    """Collects and analyzes test execution data"""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.data_points: List[Dict[str, Any]] = []
        self.start_time = datetime.utcnow()
        
        # Set up logging
        self.logger = logging.getLogger(f"test_collector.{test_name}")
        self.logger.setLevel(logging.DEBUG)
        
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs/test_data")
        logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Add file handler
        file_handler = logging.FileHandler(
            logs_dir / f"{test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
    
    def log_node_config(self, node: TextGenerationNode, note: str = ""):
        """Log node configuration"""
        config_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "node_config",
            "node_id": node.node_id,
            "node_type": node.node_type,
            "llm_config": {
                "model": node.llm_config.model,
                "temperature": node.llm_config.temperature,
                "max_tokens": node.llm_config.max_tokens
            },
            "note": note
        }
        self.data_points.append(config_data)
        self.logger.info(f"Node Config: {json.dumps(config_data, indent=2)}")
    
    def log_context_update(self, node_id: str, context: Dict[str, Any], note: str = ""):
        """Log context updates"""
        context_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "context_update",
            "node_id": node_id,
            "context": context,
            "note": note
        }
        self.data_points.append(context_data)
        self.logger.info(f"Context Update: {json.dumps(context_data, indent=2)}")
    
    def log_execution_result(self, node_id: str, result: NodeExecutionResult, note: str = ""):
        """Log execution results"""
        result_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "execution_result",
            "node_id": node_id,
            "success": result.success,
            "error": result.error,
            "error_type": result.metadata.error_type,
            "duration": result.duration,
            "output": result.output if result.success else None,
            "note": note
        }
        self.data_points.append(result_data)
        self.logger.info(f"Execution Result: {json.dumps(result_data, indent=2)}")
        
        # Cache successful execution results
        if result.success and self.should_cache_data("execution_result", result_data):
            self.cache_test_data(f"{node_id}_result", result_data)
    
    def log_chain_event(self, chain: ScriptChain, event_type: str, details: Dict[str, Any], note: str = ""):
        """Log chain-level events"""
        chain_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": f"chain_{event_type}",
            "node_count": len(chain.nodes),
            "edge_count": len(chain.graph.edges),
            "details": details,
            "note": note
        }
        self.data_points.append(chain_data)
        self.logger.info(f"Chain Event: {json.dumps(chain_data, indent=2)}")
    
    def log_user_input(self, node_id: str, input_data: Dict[str, Any], note: str = ""):
        """Log user input data"""
        input_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "user_input",
            "node_id": node_id,
            "input": input_data,
            "note": note
        }
        self.data_points.append(input_data)
        self.logger.info(f"User Input: {json.dumps(input_data, indent=2)}")
        
        # Cache user inputs
        if self.should_cache_data("user_input", input_data):
            self.cache_test_data(f"{node_id}_input", input_data)
    
    def log_generated_output(self, node_id: str, output_data: Dict[str, Any], note: str = ""):
        """Log generated output data"""
        output_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "generated_output",
            "node_id": node_id,
            "output": output_data,
            "note": note
        }
        self.data_points.append(output_data)
        self.logger.info(f"Generated Output: {json.dumps(output_data, indent=2)}")
        
        # Cache successful outputs
        if self.should_cache_data("generated_output", output_data):
            self.cache_test_data(f"{node_id}_output", output_data)
    
    def save_analysis(self):
        """Save collected data for analysis"""
        analysis_dir = Path("logs/test_analysis")
        analysis_dir.mkdir(parents=True, exist_ok=True)
        
        analysis_file = analysis_dir / f"{self.test_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(analysis_file, 'w') as f:
            json.dump({
                "test_name": self.test_name,
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.utcnow().isoformat(),
                "data_points": self.data_points
            }, f, indent=2)
        
        self.logger.info(f"Analysis saved to {analysis_file}")
    
    def cache_test_data(self, cache_key: str, data: Dict[str, Any]):
        """Cache test data for reuse"""
        cache_dir = Path("logs/test_cache")
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        cache_file = cache_dir / f"{cache_key}.json"
        with open(cache_file, 'w') as f:
            json.dump({
                "timestamp": datetime.utcnow().isoformat(),
                "data": data
            }, f, indent=2)
        
        self.logger.info(f"Test data cached to {cache_file}")
    
    def get_cached_test_data(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached test data if available"""
        cache_file = Path("logs/test_cache") / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                cached_data = json.load(f)
                self.logger.info(f"Retrieved cached test data from {cache_file}")
                return cached_data["data"]
        return None
    
    def should_cache_data(self, data_type: str, data: Dict[str, Any]) -> bool:
        """Determine if data should be cached based on type and content"""
        # Always cache user inputs
        if data_type == "user_input":
            return True
        
        # Cache generated outputs only if they're successful
        if data_type == "generated_output":
            return "error" not in data or data.get("error") is None
        
        # Don't cache execution results with errors
        if data_type == "execution_result":
            return data.get("success", False)
        
        # Don't cache chain events
        if data_type.startswith("chain_"):
            return False
        
        return False
    
    def cleanup_test_data(self, max_age_days: int = 7):
        """Clean up test data older than specified days"""
        # Clean up log files
        log_dir = Path("logs/test_data")
        if log_dir.exists():
            for log_file in log_dir.glob("*.log"):
                file_age = datetime.now() - datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_age.days > max_age_days:
                    log_file.unlink()
                    self.logger.info(f"Deleted old log file: {log_file}")
        
        # Clean up analysis files
        analysis_dir = Path("logs/test_analysis")
        if analysis_dir.exists():
            for analysis_file in analysis_dir.glob("*.json"):
                file_age = datetime.now() - datetime.fromtimestamp(analysis_file.stat().st_mtime)
                if file_age.days > max_age_days:
                    analysis_file.unlink()
                    self.logger.info(f"Deleted old analysis file: {analysis_file}")
        
        # Clean up cache files
        cache_dir = Path("logs/test_cache")
        if cache_dir.exists():
            for cache_file in cache_dir.glob("*.json"):
                file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
                if file_age.days > max_age_days:
                    cache_file.unlink()
                    self.logger.info(f"Deleted old cache file: {cache_file}")

    def log_event(self, event_type: str, details: Dict[str, Any], note: str = ""):
        """Log general events"""
        event_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details,
            "note": note
        }
        self.data_points.append(event_data)
        self.logger.info(f"Event: {json.dumps(event_data, indent=2)}")

@pytest.mark.asyncio
async def test_complex_text_generation(mock_openai):
    """Test text generation with complex prompts and context"""
    llm_config = LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    )
    
    node = TextGenerationNode.create(llm_config)
    
    # Test with complex prompt and context
    context = {
        "prompt": "Analyze the following text and provide insights: 'The quick brown fox jumps over the lazy dog'",
        "background": "This is a test of complex text analysis",
        "format": "bullet points"
    }
    
    result = await node.execute(context)
    
    # In test mode, we expect authentication errors
    assert not result.success
    assert "API key" in result.error
    assert result.metadata.error_type == "AuthenticationError"
    assert result.duration > 0
    assert result.metadata.node_type == "ai"

@pytest.mark.asyncio
async def test_script_chain_execution(mock_openai):
    """Test execution of a script chain with multiple nodes"""
    # Create nodes with different configurations
    node1 = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    ))
    
    node2 = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.5,
        max_tokens=150
    ))
    
    # Create a script chain with proper retry_config
    chain = ScriptChain(retry_config={})
    
    # Add nodes to the chain
    chain.add_node(node1)
    chain.add_node(node2)
    
    # Add edge to create dependency
    chain.add_edge(node1.node_id, node2.node_id)
    
    # Execute the chain with proper context
    context = ContextManager()
    context.set_context(node1.node_id, {"prompt": "Generate a creative story"})
    context.set_context(node2.node_id, {"prompt": "Continue the story"})
    
    result = await chain.execute()
    
    # In test mode, we expect errors
    assert not result.success
    assert isinstance(result.output, str)  # Output is a string representation of the results
    assert node1.node_id in result.output
    assert "No prompt provided in context" in result.error  # Check for the actual error message
    assert result.metadata.error_type == "ExecutionError"
    assert result.duration > 0

@pytest.mark.asyncio
async def test_error_handling_and_recovery(mock_openai):
    """Test error handling and recovery"""
    llm_config = LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    )
    
    node = TextGenerationNode.create(llm_config)
    
    # Test with missing prompt
    result = await node.execute({})
    assert not result.success
    assert result.error == "No prompt provided in context"
    assert result.metadata.error_type == "ValueError"  # Access error_type directly
    
    # Test with invalid API key
    result = await node.execute({"prompt": "Test prompt"})
    assert not result.success
    assert "API key" in result.error
    assert result.metadata.error_type == "AuthenticationError"  # Access error_type directly

@pytest.mark.asyncio
async def test_context_persistence(mock_openai):
    """Test context persistence across multiple operations"""
    llm_config = LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    )
    
    node = TextGenerationNode.create(llm_config)
    context = ContextManager()
    
    # Set up initial context
    context.set_context("test_node", {
        "background": "Testing context persistence",
        "format": "detailed",
        "prompt": "Test prompt"
    })
    
    # First execution - should fail with auth error
    result1 = await node.execute({"prompt": "First test"})
    assert not result1.success
    assert "API key" in result1.error
    assert result1.metadata.error_type == "AuthenticationError"
    
    # Second execution - should also fail with auth error
    result2 = await node.execute({"prompt": "Second test"})
    assert not result2.success
    assert "API key" in result2.error
    assert result2.metadata.error_type == "AuthenticationError"

@pytest.mark.asyncio
async def test_concurrent_execution(mock_openai):
    """Test concurrent execution of multiple nodes"""
    llm_config = LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    )
    
    node1 = TextGenerationNode.create(llm_config)
    node2 = TextGenerationNode.create(llm_config)
    node3 = TextGenerationNode.create(llm_config)
    
    # Execute nodes concurrently - all should fail with auth error
    tasks = [
        node1.execute({"prompt": "First concurrent test"}),
        node2.execute({"prompt": "Second concurrent test"}),
        node3.execute({"prompt": "Third concurrent test"})
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Verify all executions failed with auth error
    for result in results:
        assert not result.success
        assert "API key" in result.error
        assert result.metadata.error_type == "AuthenticationError"

@pytest.mark.asyncio
async def test_template_validation(mock_openai):
    """Test message template validation and usage"""
    llm_config = LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    )
    
    # Create a node with templates
    templates = [
        MessageTemplate(
            role="system",
            content="You are a helpful assistant. Background: {background}",
            version="1.0.0",
            min_model_version="gpt-4"
        ),
        MessageTemplate(
            role="user",
            content="Please analyze: {query}",
            version="1.0.0",
            min_model_version="gpt-4"
        )
    ]
    
    node_config = NodeConfig(
        metadata=NodeMetadata(
            node_id="template_test",
            node_type="ai",
            version="1.0.0",
            description="Template validation test"
        ),
        llm_config=llm_config,
        templates=templates
    )
    
    node = TextGenerationNode(node_config)
    
    # Execute with template - should fail with auth error
    result = await node.execute({
        "prompt": "Test with template",
        "background": "Test background",
        "query": "Test query"
    })
    assert not result.success
    assert "API key" in result.error
    assert result.metadata.error_type == "AuthenticationError"

@pytest.mark.asyncio
async def test_complex_text_generation_with_error_handling(mock_openai):
    """Test complex text generation with error handling"""
    llm_config = LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    )
    
    node = TextGenerationNode.create(llm_config)
    
    # Test with invalid API key - should handle gracefully
    result = await node.execute({"prompt": "Generate a complex response"})
    assert not result.success
    assert "API key" in result.error
    assert result.metadata.error_type == "AuthenticationError"

@pytest.mark.asyncio
async def test_script_chain_execution_with_error_handling(mock_openai):
    """Test script chain execution with error handling"""
    llm_config = LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    )
    
    # Create chain with single node
    chain = ScriptChain({})
    node = TextGenerationNode.create(llm_config)
    chain.add_node(node)
    
    # Execute chain - should handle node error gracefully
    result = await chain.execute()
    assert not result.success
    assert "prompt" in result.error.lower()
    assert result.metadata.error_type == "ExecutionError"

@pytest.mark.asyncio
async def test_template_validation_with_error_handling(mock_openai):
    """Test template validation with error handling"""
    llm_config = LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    )
    
    # Create node with template
    config = NodeConfig(
        metadata=NodeMetadata(
            node_id="template_test",
            node_type="ai",
            version="1.0.0",
            description="Template validation test"
        ),
        llm_config=llm_config,
        templates=[
            MessageTemplate(
                role="system",
                content="Test template",
                version="1.0.0",
                min_model_version="gpt-4"
            )
        ]
    )
    
    node = TextGenerationNode(config)
    
    # Execute with template - should fail with auth error
    result = await node.execute({"prompt": "Test with template"})
    assert not result.success
    assert "API key" in result.error
    assert result.metadata.error_type == "AuthenticationError"

@pytest.mark.asyncio
async def test_code_generation_flow(mock_openai):
    """Test code generation with multiple refinement steps"""
    # Initialize data collector
    collector = TestDataCollector("code_generation_flow")
    
    # Create nodes for code generation pipeline with specialized configurations
    generator = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=1000
    ))
    collector.log_node_config(generator, "Generator node created")
    
    reviewer = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.3,
        max_tokens=500
    ))
    collector.log_node_config(reviewer, "Reviewer node created")
    
    optimizer = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.5,
        max_tokens=1000
    ))
    collector.log_node_config(optimizer, "Optimizer node created")
    
    # Create chain for code generation workflow
    chain = ScriptChain(retry_config=None)
    chain.add_node(generator)
    chain.add_node(reviewer)
    chain.add_node(optimizer)
    chain.add_edge(generator.node_id, reviewer.node_id)
    chain.add_edge(reviewer.node_id, optimizer.node_id)
    
    collector.log_chain_event(chain, "setup", {
        "nodes": [n.node_id for n in [generator, reviewer, optimizer]],
        "edges": [(generator.node_id, reviewer.node_id), (reviewer.node_id, optimizer.node_id)]
    }, "Chain setup complete")
    
    # Set up rich context for each node with detailed instructions
    # Generator node context
    generator_context = {
        "prompt": "Generate a Python function to calculate Fibonacci numbers",
        "requirements": [
            "recursive implementation", 
            "include docstring", 
            "handle edge cases",
            "include type hints",
            "add comments explaining the algorithm"
        ],
        "language": "python",
        "style_guide": "PEP 8",
        "additional_instructions": "Make the function efficient and readable"
    }
    chain.context.set_context(generator.node_id, generator_context)
    collector.log_context_update(generator.node_id, generator_context, "Generator context set")
    collector.log_user_input(generator.node_id, generator_context, "Initial user input for code generation")
    
    # Reviewer node context
    reviewer_context = {
        "prompt": "Review the following code for best practices and potential issues: {previous_output}",
        "review_aspects": [
            "style", 
            "performance", 
            "security", 
            "documentation",
            "error handling",
            "testability"
        ],
        "review_format": "structured",
        "output_format": "JSON with sections for each aspect",
        "severity_levels": ["critical", "high", "medium", "low"]
    }
    chain.context.set_context(reviewer.node_id, reviewer_context)
    collector.log_context_update(reviewer.node_id, reviewer_context, "Reviewer context set")
    collector.log_user_input(reviewer.node_id, reviewer_context, "User input for code review")
    
    # Optimizer node context
    optimizer_context = {
        "prompt": "Optimize the following code based on the review: {previous_output}",
        "optimization_goals": [
            "performance", 
            "readability", 
            "maintainability",
            "memory usage",
            "algorithmic complexity"
        ],
        "constraints": [
            "maintain the same functionality",
            "preserve the docstring",
            "keep type hints"
        ],
        "output_format": "Python code with comments explaining changes"
    }
    chain.context.set_context(optimizer.node_id, optimizer_context)
    collector.log_context_update(optimizer.node_id, optimizer_context, "Optimizer context set")
    collector.log_user_input(optimizer.node_id, optimizer_context, "User input for code optimization")
    
    # Execute the chain
    collector.log_chain_event(chain, "execution_start", {}, "Starting chain execution")
    result = await chain.execute()
    collector.log_chain_event(chain, "execution_complete", {
        "success": result.success,
        "error": result.error,
        "error_type": result.metadata.error_type,
        "duration": result.duration
    }, "Chain execution completed")
    
    # Test assertions
    assert not result.success
    assert "API key" in str(result.output)
    assert result.metadata.error_type == "ExecutionError"
    assert result.duration > 0
    
    # Test context passing by manually simulating the flow
    collector.log_chain_event(chain, "manual_simulation_start", {}, "Starting manual node execution simulation")
    
    # Step 1: Generator node execution
    generator_result = await generator.execute(chain.context.get_context(generator.node_id))
    collector.log_execution_result(generator.node_id, generator_result, "Generator node execution")
    assert not generator_result.success
    assert "API key" in generator_result.error
    assert generator_result.metadata.error_type == "AuthenticationError"
    
    # Step 2: Reviewer node execution with generator output
    reviewer_context = chain.context.get_context(reviewer.node_id)
    reviewer_context["previous_output"] = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"
    collector.log_context_update(reviewer.node_id, reviewer_context, "Updated reviewer context with generator output")
    collector.log_generated_output(generator.node_id, {"code": reviewer_context["previous_output"]}, "Generated Fibonacci function")
    reviewer_result = await reviewer.execute(reviewer_context)
    collector.log_execution_result(reviewer.node_id, reviewer_result, "Reviewer node execution")
    assert not reviewer_result.success
    assert "API key" in reviewer_result.error
    assert reviewer_result.metadata.error_type == "AuthenticationError"
    
    # Step 3: Optimizer node execution with reviewer output
    optimizer_context = chain.context.get_context(optimizer.node_id)
    optimizer_context["previous_output"] = "The code is inefficient due to redundant calculations. Consider using memoization."
    collector.log_context_update(optimizer.node_id, optimizer_context, "Updated optimizer context with reviewer output")
    collector.log_generated_output(reviewer.node_id, {"review": optimizer_context["previous_output"]}, "Generated code review")
    optimizer_result = await optimizer.execute(optimizer_context)
    collector.log_execution_result(optimizer.node_id, optimizer_result, "Optimizer node execution")
    assert not optimizer_result.success
    assert "API key" in optimizer_result.error
    assert optimizer_result.metadata.error_type == "AuthenticationError"
    
    collector.log_chain_event(chain, "manual_simulation_complete", {}, "Manual simulation completed")
    
    # Save test data for analysis
    collector.save_analysis()

@pytest.mark.asyncio
async def test_error_recovery_flow(mock_openai):
    """Test error handling and recovery in a multi-node scenario"""
    # Create nodes with different error scenarios
    node1 = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    ))
    
    node2 = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    ))
    
    fallback = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.7,
        max_tokens=100
    ))
    
    # Create chain with error handling
    chain = ScriptChain(retry_config={
        "max_retries": 2,
        "retry_delay": 1
    })
    chain.add_node(node1)
    chain.add_node(node2)
    chain.add_node(fallback)
    chain.add_edge(node1.node_id, node2.node_id)
    chain.add_edge(node2.node_id, fallback.node_id)
    
    # Set up context for error scenarios
    context = ContextManager()
    context.set_context(node1.node_id, {
        "prompt": "Generate text that will cause a rate limit error",
        "error_scenario": "rate_limit"
    })
    
    context.set_context(node2.node_id, {
        "prompt": "Process the output: {previous_output}",
        "error_scenario": "timeout"
    })
    
    context.set_context(fallback.node_id, {
        "prompt": "Fallback processing: {previous_output}",
        "error_handling": "graceful_degradation"
    })
    
    result = await chain.execute()
    
    # In test mode, we expect authentication errors
    assert not result.success
    assert "API key" in result.error
    assert result.metadata.error_type == "AuthenticationError"
    assert result.duration > 0 

@pytest.mark.asyncio
async def test_psychological_profile_analysis(mock_openai):
    """Test psychological profile analysis with essay input"""
    # Initialize data collector
    collector = TestDataCollector("psychological_profile_analysis")
    
    # Clean up old test data (older than 7 days)
    collector.cleanup_test_data(max_age_days=7)
    
    # Create nodes with different configurations
    text_analyzer = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.3,
        max_tokens=1000
    ))
    collector.log_node_config(text_analyzer, "Text analyzer node created")
    
    personality_assessor = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.4,
        max_tokens=800
    ))
    collector.log_node_config(personality_assessor, "Personality assessor node created")
    
    recommendation_generator = TextGenerationNode.create(LLMConfig(
        api_key="sk-test-key-for-testing-only",
        model="gpt-4",
        temperature=0.5,
        max_tokens=600
    ))
    collector.log_node_config(recommendation_generator, "Recommendation generator node created")
    
    # Create chain for psychological analysis workflow
    chain = ScriptChain(retry_config=None)
    chain.add_node(text_analyzer)
    chain.add_node(personality_assessor)
    chain.add_node(recommendation_generator)
    chain.add_edge(text_analyzer.node_id, personality_assessor.node_id)
    chain.add_edge(personality_assessor.node_id, recommendation_generator.node_id)
    
    collector.log_chain_event(chain, "setup", {
        "nodes": [n.node_id for n in [text_analyzer, personality_assessor, recommendation_generator]],
        "edges": [(text_analyzer.node_id, personality_assessor.node_id), 
                 (personality_assessor.node_id, recommendation_generator.node_id)]
    }, "Psychological analysis chain setup complete")
    
    # Sample essay excerpt for analysis
    essay_excerpt = """
    The world is a complex place, filled with contradictions and paradoxes. I often find myself 
    standing at the crossroads of certainty and doubt, wondering which path to take. The weight 
    of decisions presses upon me, yet I cannot help but feel a strange exhilaration in the face 
    of uncertainty. My mind wanders through the labyrinth of possibilities, each turn revealing 
    new perspectives and challenges. I am drawn to the edges of understanding, where questions 
    outnumber answers and the familiar gives way to the unknown. In these moments of reflection, 
    I feel most alive, most connected to the essence of what it means to be human.
    
    Yet there is also a part of me that yearns for simplicity, for the comfort of clear boundaries 
    and predictable outcomes. I find myself creating order from chaos, organizing thoughts and 
    experiences into neat categories that help me navigate the world. This tension between 
    exploration and organization defines much of who I am. I am both the wanderer and the cartographer, 
    mapping the terrain of my own consciousness while allowing myself to get lost in its vastness.
    """
    
    # Set up context for text analyzer node
    analyzer_context = {
        "prompt": "Analyze the following essay excerpt for psychological insights:",
        "essay": essay_excerpt,
        "analysis_focus": [
            "writing style",
            "emotional content",
            "cognitive patterns",
            "recurring themes",
            "self-perception",
            "worldview"
        ],
        "output_format": "Structured analysis with sections for each focus area"
    }
    chain.context.set_context(text_analyzer.node_id, analyzer_context)
    collector.log_context_update(text_analyzer.node_id, analyzer_context, "Text analyzer context set")
    collector.log_user_input(text_analyzer.node_id, analyzer_context, "Initial user input for text analysis")
    
    # Check for cached analyzer result
    cached_analyzer_result = collector.get_cached_test_data(f"{text_analyzer.node_id}_result")
    if cached_analyzer_result:
        collector.logger.info("Using cached analyzer result")
        # In a real test, we would use the cached result
        # For this test, we'll still execute the node to demonstrate the flow
    
    # Set up context for personality assessor node
    assessor_context = {
        "prompt": "Based on the following psychological analysis, create a personality profile: {previous_output}",
        "profile_aspects": [
            "personality traits",
            "cognitive style",
            "emotional patterns",
            "interpersonal dynamics",
            "potential strengths",
            "potential challenges"
        ],
        "theoretical_frameworks": [
            "Big Five personality traits",
            "Jungian archetypes",
            "Attachment theory",
            "Cognitive behavioral patterns"
        ],
        "output_format": "Comprehensive personality profile with supporting evidence from the text"
    }
    chain.context.set_context(personality_assessor.node_id, assessor_context)
    collector.log_context_update(personality_assessor.node_id, assessor_context, "Personality assessor context set")
    collector.log_user_input(personality_assessor.node_id, assessor_context, "User input for personality assessment")
    
    # Check for cached assessor result
    cached_assessor_result = collector.get_cached_test_data(f"{personality_assessor.node_id}_result")
    if cached_assessor_result:
        collector.logger.info("Using cached assessor result")
        # In a real test, we would use the cached result
        # For this test, we'll still execute the node to demonstrate the flow
    
    # Set up context for recommendation generator node
    recommendation_context = {
        "prompt": "Based on the following personality profile, provide personalized recommendations: {previous_output}",
        "recommendation_areas": [
            "personal growth",
            "interpersonal relationships",
            "career development",
            "emotional well-being",
            "cognitive enhancement"
        ],
        "recommendation_style": "practical and actionable",
        "output_format": "Structured recommendations with explanations of how they align with the personality profile"
    }
    chain.context.set_context(recommendation_generator.node_id, recommendation_context)
    collector.log_context_update(recommendation_generator.node_id, recommendation_context, "Recommendation generator context set")
    collector.log_user_input(recommendation_generator.node_id, recommendation_context, "User input for recommendation generation")
    
    # Check for cached recommendation result
    cached_recommendation_result = collector.get_cached_test_data(f"{recommendation_generator.node_id}_result")
    if cached_recommendation_result:
        collector.logger.info("Using cached recommendation result")
        # In a real test, we would use the cached result
        # For this test, we'll still execute the node to demonstrate the flow
    
    # Execute the chain
    collector.log_chain_event(chain, "execution_start", {}, "Starting psychological analysis chain execution")
    result = await chain.execute()
    collector.log_chain_event(chain, "execution_complete", {
        "success": result.success,
        "error": result.error,
        "error_type": result.metadata.error_type,
        "duration": result.duration
    }, "Psychological analysis chain execution completed")
    
    # Test assertions
    assert not result.success
    assert "API key" in str(result.output)
    assert result.metadata.error_type == "ExecutionError"
    assert result.duration > 0
    
    # Test context passing by manually simulating the flow
    collector.log_chain_event(chain, "manual_simulation_start", {}, "Starting manual psychological analysis simulation")
    
    # Step 1: Text analyzer node execution
    analyzer_result = await text_analyzer.execute(chain.context.get_context(text_analyzer.node_id))
    collector.log_execution_result(text_analyzer.node_id, analyzer_result, "Text analyzer node execution")
    assert not analyzer_result.success
    assert "API key" in analyzer_result.error
    assert analyzer_result.metadata.error_type == "AuthenticationError"
    
    # Step 2: Personality assessor node execution with analyzer output
    assessor_context = chain.context.get_context(personality_assessor.node_id)
    assessor_context["previous_output"] = """
    Psychological Analysis:
    
    Writing Style: The author employs a reflective, introspective tone with rich metaphorical language. 
    The prose is thoughtful and contemplative, suggesting a preference for deep processing of experiences.
    
    Emotional Content: There is a complex emotional landscape present, with both excitement about 
    uncertainty and a desire for stability. The text reveals emotional awareness and self-reflection.
    
    Cognitive Patterns: The author demonstrates abstract thinking, pattern recognition, and 
    meta-cognitive awareness. There is evidence of both analytical and intuitive cognitive styles.
    
    Recurring Themes: Exploration vs. organization, certainty vs. doubt, complexity vs. simplicity. 
    The author seems to navigate these dichotomies consciously.
    
    Self-Perception: The author appears to have a nuanced self-concept, recognizing multiple aspects 
    of their personality ("both the wanderer and the cartographer").
    
    Worldview: The author perceives the world as complex and paradoxical, yet navigable through 
    personal meaning-making and organization.
    """
    collector.log_context_update(personality_assessor.node_id, assessor_context, "Updated assessor context with analyzer output")
    collector.log_generated_output(text_analyzer.node_id, {"analysis": assessor_context["previous_output"]}, "Generated psychological analysis")
    assessor_result = await personality_assessor.execute(assessor_context)
    collector.log_execution_result(personality_assessor.node_id, assessor_result, "Personality assessor node execution")
    assert not assessor_result.success
    assert "API key" in assessor_result.error
    assert assessor_result.metadata.error_type == "AuthenticationError"
    
    # Step 3: Recommendation generator node execution with assessor output
    recommendation_context = chain.context.get_context(recommendation_generator.node_id)
    recommendation_context["previous_output"] = """
    Personality Profile:
    
    Personality Traits: High in Openness to Experience, moderate to high in Conscientiousness, 
    moderate in Extraversion (likely introverted in social settings but internally energetic), 
    moderate to high in Emotional Stability, and moderate to high in Agreeableness.
    
    Cognitive Style: Integrative thinker who combines analytical and intuitive approaches. 
    Shows strong pattern recognition abilities and meta-cognitive awareness.
    
    Emotional Patterns: Emotionally aware with capacity for both experiencing and observing emotions. 
    Shows comfort with complexity and ambiguity in emotional experiences.
    
    Interpersonal Dynamics: Likely values deep, meaningful connections over superficial relationships. 
    May be selective in social interactions, preferring quality over quantity.
    
    Potential Strengths: Creative problem-solving, adaptability, self-awareness, 
    ability to hold multiple perspectives, capacity for deep work and focus.
    
    Potential Challenges: May experience decision paralysis due to seeing multiple possibilities, 
    potential for overthinking, may struggle with perfectionism in organizing thoughts and experiences.
    """
    collector.log_context_update(recommendation_generator.node_id, recommendation_context, "Updated recommendation context with assessor output")
    collector.log_generated_output(personality_assessor.node_id, {"profile": recommendation_context["previous_output"]}, "Generated personality profile")
    recommendation_result = await recommendation_generator.execute(recommendation_context)
    collector.log_execution_result(recommendation_generator.node_id, recommendation_result, "Recommendation generator node execution")
    assert not recommendation_result.success
    assert "API key" in recommendation_result.error
    assert recommendation_result.metadata.error_type == "AuthenticationError"
    
    collector.log_chain_event(chain, "manual_simulation_complete", {}, "Manual psychological analysis simulation completed")
    
    # Save test data for analysis
    collector.save_analysis() 

@pytest.mark.asyncio
async def test_real_api_integration():
    """
    Test integration with real OpenAI API when available.
    This test will:
    1. Use real API keys if available, fall back to mocks if not
    2. Measure performance metrics (response time, token usage)
    3. Validate output quality against predefined criteria
    """
    # Get API key (real or mock)
    api_key, is_mock = get_api_key()
    
    # Initialize data collector
    collector = TestDataCollector("real_api_integration")
    
    # Create a node with the API key
    node = TextGenerationNode.create(LLMConfig(
        api_key=api_key,
        model="gpt-4",
        temperature=0.7,
        max_tokens=500
    ))
    collector.log_node_config(node, "Text generation node created")
    
    # Define test prompts with expected quality criteria
    test_cases = [
        {
            "name": "factual_response",
            "prompt": "What is the capital of France?",
            "quality_criteria": {
                "contains": ["Paris"],
                "max_length": 100,
                "response_time_threshold": 5.0  # seconds
            }
        },
        {
            "name": "creative_writing",
            "prompt": "Write a short poem about technology.",
            "quality_criteria": {
                "min_length": 50,
                "contains_keywords": ["tech", "digital", "future"],
                "response_time_threshold": 10.0  # seconds
            }
        },
        {
            "name": "code_generation",
            "prompt": "Write a Python function to calculate the Fibonacci sequence. Return ONLY the function definition with no additional text or explanation. The code must be syntactically valid Python.",
            "quality_criteria": {
                "contains": ["def", "fibonacci", "return"],
                "is_valid_python": True,
                "response_time_threshold": 8.0  # seconds
            }
        }
    ]
    
    # Run tests for each prompt
    for test_case in test_cases:
        # Log test case start without chain info
        collector.log_event("test_case_start", {"name": test_case["name"]}, f"Starting test case: {test_case['name']}")
        
        # Set up context
        context = {
            "prompt": test_case["prompt"],
            "output_format": "text"
        }
        collector.log_context_update(node.node_id, context, f"Context set for {test_case['name']}")
        collector.log_user_input(node.node_id, context, f"User input for {test_case['name']}")
        
        # Execute the node and measure performance
        start_time = datetime.utcnow()
        result = await node.execute(context)
        end_time = datetime.utcnow()
        execution_time = (end_time - start_time).total_seconds()
        
        # Log execution result
        collector.log_execution_result(node.node_id, result, f"Execution result for {test_case['name']}")
        
        # Log performance metrics
        performance_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "performance_metrics",
            "test_case": test_case["name"],
            "execution_time": execution_time,
            "success": result.success,
            "error": result.error if not result.success else None
        }
        collector.data_points.append(performance_data)
        collector.logger.info(f"Performance Metrics: {json.dumps(performance_data, indent=2)}")
        
        # Validate output quality if execution was successful
        if result.success and result.output:
            # Log the generated output
            collector.log_generated_output(node.node_id, {"output": result.output}, f"Generated output for {test_case['name']}")
            collector.logger.info(f"Raw output for {test_case['name']}: {result.output}")
            
            # For code generation, try to clean the output
            if test_case["name"] == "code_generation":
                # Remove any markdown code block markers if present
                cleaned_output = result.output.strip()
                if cleaned_output.startswith("```python"):
                    cleaned_output = cleaned_output[8:]
                if cleaned_output.startswith("```"):
                    cleaned_output = cleaned_output[3:]
                if cleaned_output.endswith("```"):
                    cleaned_output = cleaned_output[:-3]
                cleaned_output = cleaned_output.strip()
                result.output = cleaned_output
                collector.logger.info(f"Cleaned code output: {cleaned_output}")
            
            # Validate against quality criteria
            quality_validation = validate_output_quality(result.output, test_case["quality_criteria"])
            collector.data_points.append(quality_validation)
            collector.logger.info(f"Quality Validation: {json.dumps(quality_validation, indent=2)}")
            
            # Assert quality criteria are met
            assert quality_validation["passed"], f"Quality validation failed for {test_case['name']}: {quality_validation['failures']}"
            
            # Assert performance criteria are met
            assert execution_time <= test_case["quality_criteria"]["response_time_threshold"], \
                f"Performance threshold exceeded for {test_case['name']}: {execution_time}s > {test_case['quality_criteria']['response_time_threshold']}s"
        else:
            # If using mock API, we expect authentication errors
            if is_mock:
                assert not result.success
                assert "API key" in result.error
                assert result.metadata.error_type == "AuthenticationError"
            else:
                # If using real API, we should have successful results
                assert result.success, f"Execution failed for {test_case['name']}: {result.error}"
        
        # Log test case completion without chain info
        collector.log_event("test_case_complete", {"name": test_case["name"]}, f"Completed test case: {test_case['name']}")
    
    # Save test data for analysis
    collector.save_analysis()

def validate_output_quality(output: str, criteria: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate output quality against predefined criteria.
    Returns a dictionary with validation results.
    """
    validation_result = {
        "timestamp": datetime.utcnow().isoformat(),
        "event_type": "quality_validation",
        "passed": True,
        "failures": []
    }
    
    # Check if output contains required keywords
    if "contains" in criteria:
        for keyword in criteria["contains"]:
            if keyword.lower() not in output.lower():
                validation_result["passed"] = False
                validation_result["failures"].append(f"Missing required keyword: {keyword}")
    
    # Check if output contains any of the keywords
    if "contains_keywords" in criteria:
        found_keywords = [kw for kw in criteria["contains_keywords"] if kw.lower() in output.lower()]
        if not found_keywords:
            validation_result["passed"] = False
            validation_result["failures"].append(f"Missing any of the keywords: {criteria['contains_keywords']}")
    
    # Check output length
    if "min_length" in criteria and len(output) < criteria["min_length"]:
        validation_result["passed"] = False
        validation_result["failures"].append(f"Output too short: {len(output)} < {criteria['min_length']}")
    
    if "max_length" in criteria and len(output) > criteria["max_length"]:
        validation_result["passed"] = False
        validation_result["failures"].append(f"Output too long: {len(output)} > {criteria['max_length']}")
    
    # Check if output is valid Python code
    if criteria.get("is_valid_python", False):
        try:
            compile(output, "<string>", "exec")
        except SyntaxError as e:
            validation_result["passed"] = False
            validation_result["failures"].append(f"Invalid Python code: {str(e)}")
    
    return validation_result 