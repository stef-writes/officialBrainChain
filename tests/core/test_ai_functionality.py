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
from app.nodes.text_generation import TextGenerationNode
from app.models.config import LLMConfig, MessageTemplate
from app.models.node_models import NodeConfig, NodeMetadata, NodeExecutionResult
from app.chains.script_chain import ScriptChain
from app.utils.context import ContextManager
import time

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
            "usage": result.usage.dict() if result.usage else None,
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
    
    # Set up context using chain's context manager
    chain.context.set_context(node1.node_id, {"prompt": "Generate a creative story"})
    chain.context.set_context(node2.node_id, {"prompt": "Continue the story"})
    
    result = await chain.execute()
    
    # In test mode, we expect authentication errors
    assert not result.success
    assert "API key" in result.error
    assert result.metadata.error_type == "AuthenticationError"
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
async def test_script_chain_execution_with_error_handling():
    """Test script chain execution with error handling"""
    # Create nodes
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
    
    # Create chain
    chain = ScriptChain(retry_config=None)
    chain.add_node(node1)
    chain.add_node(node2)
    chain.add_edge(node1.node_id, node2.node_id)
    
    # Set up context with prompt
    chain.context.set_context(node1.node_id, {"prompt": "Test prompt"})
    
    # Execute chain
    result = await chain.execute()
    
    # Verify error handling
    assert not result.success
    assert result.metadata.error_type == "AuthenticationError"
    assert "API key" in str(result.error)

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
    assert result.metadata.error_type == "AuthenticationError"
    assert result.duration > 0

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
        "delay": 1
    })
    chain.add_node(node1)
    chain.add_node(node2)
    chain.add_node(fallback)
    chain.add_edge(node1.node_id, node2.node_id)
    chain.add_edge(node2.node_id, fallback.node_id)
    
    # Set up context for error scenarios
    chain.context.set_context(node1.node_id, {
        "prompt": "Generate text that will cause a rate limit error",
        "error_scenario": "rate_limit"
    })
    
    chain.context.set_context(node2.node_id, {
        "prompt": "Process the output: {previous_output}",
        "error_scenario": "timeout"
    })
    
    chain.context.set_context(fallback.node_id, {
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
async def test_real_api_integration():
    """Test integration with real OpenAI API."""
    # Get API key from environment
    api_key, _ = get_api_key()
    if not api_key:
        pytest.skip("No API key available")
        
    # Create node with real API configuration
    node = TextGenerationNode(
        config=NodeConfig(
            metadata=NodeMetadata(
                node_id="real_api_test",
                node_type="ai",
                version="1.0.0",
                description="Test with real API"
            ),
            llm_config=LLMConfig(
                model="gpt-4",  # Using gpt-4 instead of gpt-3.5-turbo
                temperature=0.7,
                max_tokens=100,
                api_key=api_key
            )
        )
    )
    
    # Execute with real API
    start_time = time.time()
    result = await node.execute({
        "prompt": "Write a haiku about testing. Start with 'Here's a haiku:' and then write the haiku on separate lines."
    })
    execution_time = time.time() - start_time
    
    # Verify response
    assert result.success
    assert "haiku" in result.output.lower()
    assert execution_time < 15.0  # Allow more time for real API call
    
    # Verify usage metadata
    assert result.usage is not None
    assert result.usage.prompt_tokens > 0
    assert result.usage.completion_tokens > 0
    assert result.usage.total_tokens > 0
    assert result.usage.api_calls == 1
    assert result.usage.model == "gpt-4"  # Changed from gpt-3.5-turbo to gpt-4

@pytest.mark.asyncio
async def test_concurrent_execution():
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