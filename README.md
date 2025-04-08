# Gaffer

A powerful workflow engine for building and executing AI-powered script chains.

## Features

- **Script Chain Execution**: Build and execute chains of AI nodes
- **Context Management**: Intelligent context handling with token awareness
- **Vector Store Integration**: Semantic context retrieval and storage
- **Callback System**: Comprehensive event tracking and monitoring
- **Token Optimization**: Smart token allocation and context optimization

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Gaffer.git
cd Gaffer

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Script Chain

```python
from app.chains.script_chain import ScriptChain
from app.models.node_models import NodeConfig

# Create a chain
chain = ScriptChain()

# Add nodes
chain.add_node(NodeConfig(
    id="node1",
    type="llm",
    model="gpt-4",
    prompt="Generate a creative story"
))

# Execute the chain
result = await chain.execute()
```

### Context Management

The context manager provides intelligent context handling with vector-based retrieval:

```python
from app.utils.context import ContextManager

# Initialize context manager
context_manager = ContextManager(max_context_tokens=4000)

# Set context for a node
context_manager.set_context("node1", {
    "system": "You are a creative writer",
    "output": "Once upon a time..."
})

# Get optimized context with vector retrieval
context = context_manager.get_context_with_optimization("node1")
```

### Callback System

Monitor chain execution with the callback system:

```python
from app.utils.debug_callback import DebugCallback

# Create a debug callback
callback = DebugCallback()

# Add callback to chain
chain.add_callback(callback)

# Execute chain
result = await chain.execute()

# Get events
events = callback.get_events()
```

## Vector Store

The vector store provides semantic context retrieval:

- **Context Storage**: Automatically stores node contexts with metadata
- **Semantic Search**: Find similar contexts using vector similarity
- **Token Optimization**: Smart allocation of token budget for vector results
- **Event Tracking**: Monitor vector operations through the callback system

## Development

### Running Tests

```bash
python -m pytest tests/
```

### Code Style

```bash
# Format code
black .

# Sort imports
isort .

# Type checking
mypy .
```

## License

MIT License

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
    - `callbacks.py`: Callback system for workflow monitoring
    - `debug_callback.py`: Debug implementation of callbacks
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

### Event Callbacks

The callback system provides comprehensive monitoring of workflow execution:
- Chain lifecycle events (start/end)
- Node execution events (start/complete/error)
- Context updates
- Customizable callback implementations
- Built-in debug callback for logging

### Error Handling

Comprehensive error handling includes:
- Authentication errors
- Rate limit handling
- API errors
- Network issues
- Invalid input validation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 