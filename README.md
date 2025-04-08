# Gaffer

A powerful workflow orchestration system for LLM (Large Language Model) nodes.

## Overview

Gaffer is a framework for creating and executing complex workflows involving Large Language Models (LLMs). It provides a flexible and extensible architecture for building AI-powered applications.

## Features

- **Directed Graph Workflows**: Create complex workflows with dependencies between nodes
- **Smart Context Management**: 
  - Token-aware optimization with dynamic summarization
  - Intelligent context prioritization based on importance
  - Automatic context truncation to stay within token limits
- **Parallel Workflow Execution**: Execute multiple nodes concurrently with proper dependency handling
- **Robust Error Handling**: Comprehensive error handling with detailed error types and messages
- **Retry Mechanism**: Handle API call failures and rate limits gracefully
- **Extensible Node System**: Create custom nodes for different types of operations
- **Comprehensive Testing**: Robust test suite with high coverage

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Gaffer.git
cd Gaffer

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Creating a Simple Chain

```python
from app.chains.script_chain import ScriptChain
from app.nodes.text_generation import TextGenerationNode
from app.models.config import LLMConfig

# Create nodes
node1 = TextGenerationNode.create(LLMConfig(
    api_key="your-api-key",
    model="gpt-4",
    temperature=0.7,
    max_tokens=100
))

node2 = TextGenerationNode.create(LLMConfig(
    api_key="your-api-key",
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

# Check result
print(result.output)
```

### Parallel Workflow Execution

```python
# Create multiple nodes
nodes = [TextGenerationNode.create(llm_config) for _ in range(3)]
node_ids = [node.node_id for node in nodes]

# Set context for each node
for i, node_id in enumerate(node_ids):
    chain.context.set_context(node_id, {
        "prompt": f"What is {i}+2? Answer in one word."
    })

# Execute workflow in parallel
results = await chain.execute_workflow(nodes, node_ids)

# Process results
for node_id, result in results.items():
    if result.success:
        print(f"Node {node_id}: {result.output}")
```

## Project Structure

- `app/`: Main application code
  - `api/`: API endpoints
  - `chains/`: Workflow orchestration
  - `models/`: Data models
  - `nodes/`: Node implementations
  - `utils/`: Utility functions
    - `context.py`: Smart context management
    - `retry.py`: Retry mechanism for API calls
    - `logging.py`: Enhanced logging utilities
- `tests/`: Test suite with real API integration tests

## Testing

```bash
# Run all tests
python -m pytest

# Run tests with coverage
python -m pytest --cov=app

# Run specific test file
python -m pytest tests/test_script_chain.py
```

## Features in Detail

### Smart Context Management

The context manager includes several optimizations:
- Token counting for different data types
- Dynamic context summarization for long texts
- Importance-based prioritization of context
- Automatic truncation to stay within token limits
- Parent context inheritance with optimization

### Error Handling

Comprehensive error handling includes:
- Authentication errors
- Rate limit handling
- API errors
- Network issues
- Invalid input validation

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 