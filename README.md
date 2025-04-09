# Gaffer

A powerful workflow engine for building and executing AI-powered script chains with level-based parallel execution.

## Features

- **Enhanced Script Chain Execution**: Build and execute chains of AI nodes with level-based parallel processing
- **Graph-based Context Management**: Intelligent context handling with graph-based inheritance and token awareness
- **Vector Store Integration**: Semantic context retrieval and storage with similarity search
- **Comprehensive Callback System**: Event tracking, metrics collection, and debugging capabilities
- **Token Optimization**: Smart token allocation and context optimization with vector-based retrieval

## Core Components

### ScriptChain

The enhanced ScriptChain implementation provides:
- **Level-based Parallel Execution**: Nodes at the same level execute concurrently
- **Robust Error Handling**: Comprehensive error management with retries and fallbacks
- **Configurable Concurrency**: Control parallel execution with concurrency limits
- **Validation**: Automatic validation of workflow structure and dependencies

### GraphContextManager

Advanced context management with:
- **Graph-based Inheritance**: Context flows through the workflow graph
- **Vector Store Integration**: Semantic similarity search for context optimization
- **Token Awareness**: Smart token budget allocation and optimization
- **Error Tracking**: Comprehensive error logging and tracking

### Callback System

Three types of callbacks for different use cases:
- **LoggingCallback**: Production-grade logging of workflow events
- **MetricsCallback**: Performance metrics collection and analysis
- **DebugCallback**: Detailed debugging with event tracking and analysis

## Integration with LangChain and Pinecone

The Gaffer project now integrates LangChain and Pinecone to enhance context management and vector storage capabilities.

### LangChain
- Provides structured context management, reducing complexity and improving maintainability.
- Supports advanced prompt management and chaining, enhancing the flexibility of script chains.

### Pinecone
- Offers efficient vector storage and retrieval, improving the performance of semantic searches.
- Manages vector data, allowing for scalable and high-performance similarity search.

### Updated Features
- **Context Management**: Now uses LangChain for improved context handling and optimization.
- **Vector Store**: Integrated with Pinecone for efficient vector-based context retrieval.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/Gaffer.git
cd Gaffer

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Enhanced Script Chain

```python
from app.chains.script_chain import ScriptChain
from app.models.node_models import NodeConfig
from app.utils.callbacks import LoggingCallback, MetricsCallback

# Create a chain with configuration
chain = ScriptChain(
    concurrency_level=3,  # Execute up to 3 nodes in parallel
    retry_policy={
        'max_retries': 3,
        'delay': 1,
        'backoff': 2
    }
)

# Add nodes with levels
chain.add_node(NodeConfig(
    id="node1",
    type="llm",
    model="gpt-4",
    prompt="Generate a creative story",
    level=0  # Base level
))

chain.add_node(NodeConfig(
    id="node2",
    type="llm",
    model="gpt-4",
    prompt="Enhance the story",
    level=1,  # Will execute after level 0
    dependencies=["node1"]
))

# Add callbacks
chain.add_callback(LoggingCallback())
chain.add_callback(MetricsCallback())

# Execute the chain
result = await chain.execute()
```

### Graph-based Context Management

```python
from app.utils.context import GraphContextManager
import networkx as nx

# Create a workflow graph
graph = nx.DiGraph()
graph.add_edge("node1", "node2")

# Initialize context manager with graph
context_manager = GraphContextManager(
    max_tokens=4000,
    graph=graph,
    vector_store=your_vector_store
)

# Set and retrieve context
context_manager.set_context("node1", {
    "system": "You are a creative writer",
    "output": "Once upon a time..."
})

# Get optimized context with graph inheritance
context = context_manager.get_context_with_optimization("node2")
```

### Comprehensive Callback System

```python
from app.utils.callbacks import LoggingCallback, MetricsCallback
from app.utils.debug_callback import DebugCallback
import logging

# Production logging
logging_callback = LoggingCallback(log_level=logging.INFO)

# Metrics collection
metrics_callback = MetricsCallback()

# Debugging
debug_callback = DebugCallback()

# Add callbacks to chain
chain.add_callback(logging_callback)
chain.add_callback(metrics_callback)
chain.add_callback(debug_callback)

# Execute chain
result = await chain.execute()

# Access metrics
metrics = metrics_callback.get_metrics()
metrics_callback.export_metrics("execution_metrics.json")

# Access debug events
debug_events = debug_callback.get_events()
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
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=app
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
  - `chains/`: Workflow orchestration
    - `script_chain.py`: Enhanced ScriptChain implementation
  - `models/`: Data models and configurations
  - `utils/`: Utility functions
    - `context.py`: Graph-based context management
    - `callbacks.py`: Production callbacks (Logging, Metrics)
    - `debug_callback.py`: Debug callback implementation
    - `retry.py`: Configurable retry mechanism

## Features in Detail

### Level-based Parallel Execution

The ScriptChain supports parallel execution of nodes:
- Automatic level calculation based on dependencies
- Configurable concurrency limits
- Semaphore-based concurrency control
- Proper error propagation across levels

### Graph Context Management

The GraphContextManager provides:
- Graph-based context inheritance
- Vector-based context optimization
- Token-aware context merging
- Error tracking and logging
- Metadata management

### Callback System

Three types of callbacks for different needs:
- **LoggingCallback**: Production event logging
  - Configurable log levels
  - Structured log messages
  - Chain and node lifecycle events
- **MetricsCallback**: Performance monitoring
  - Execution timing
  - Token usage tracking
  - Success/failure rates
  - Vector store operations
- **DebugCallback**: Detailed debugging
  - Complete event history
  - Context update tracking
  - Vector operation monitoring
  - Detailed error information

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. 