# Gaffer

A powerful workflow orchestration system for LLM (Large Language Model) nodes.

## Overview

Gaffer is a framework for creating and executing complex workflows involving Large Language Models (LLMs). It provides a flexible and extensible architecture for building AI-powered applications.

## Features

- **Directed Graph Workflows**: Create complex workflows with dependencies between nodes
- **Context Management**: Maintain state across nodes with token-aware optimization
- **Retry Mechanism**: Handle API call failures gracefully
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

## Project Structure

- `app/`: Main application code
  - `api/`: API endpoints
  - `chains/`: Workflow orchestration
  - `models/`: Data models
  - `nodes/`: Node implementations
  - `utils/`: Utility functions
- `tests/`: Test suite

## Testing

```bash
# Run all tests
python -m pytest

# Run tests with coverage
python -m pytest --cov=app
```

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 