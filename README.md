# Gaffer

A Python-based workflow orchestration system that provides a flexible framework for building and executing directed acyclic graphs (DAGs) of text generation nodes.

## Core Components

### ScriptChain

The primary orchestration engine that manages workflow execution and node dependencies.

#### Key Features
- Level-based parallel execution
- Context management with token tracking
- Configurable LLM integration
- Vector store integration for context retrieval
- Workflow validation for orphan nodes and disconnected components

#### Configuration
```python
llm_config = LLMConfig(
    model="gpt-4",
    api_key="your-api-key",
    temperature=0.7,
    max_tokens=1000,
    max_context_tokens=1000
)

script_chain = ScriptChain(
    llm_config=llm_config,
    vector_store=vector_store  # Optional
)
```

### Node System

#### TextGenerationNode
- Primary node type for text generation
- Supports async execution
- Integrates with OpenAI's GPT models
- Handles context window management
- Token usage tracking

#### Node Configuration
```python
node_config = NodeConfig(
    node_id="unique_id",
    prompt_template="Your prompt template",
    dependencies=["dependency1", "dependency2"],
    max_tokens=1000,
    temperature=0.7
)
```

### Context Management

#### VectorStore
- Abstract interface for vector storage
- Supports similarity search
- Configurable chunk size and overlap
- Metadata storage capabilities

#### ContextWindow
- Manages token limits
- Handles context truncation
- Supports metadata tracking
- Configurable window size

### Workflow Execution

#### Level-based Processing
1. Nodes are organized into execution levels
2. Each level executes in parallel
3. Dependencies are resolved automatically
4. Results are propagated to dependent nodes

#### Error Handling
- Graceful failure handling
- Error propagation
- Node-level error isolation
- Execution state preservation

### Testing Framework

#### Test Coverage
- Unit tests for core components
- Integration tests for workflow execution
- Mock implementations for external services
- Fixture-based test setup

#### Key Test Areas
- Node initialization
- Workflow validation
- Parallel execution
- Error handling
- Context management
- Vector store integration

## Technical Implementation

### Dependencies
- Python 3.8+
- langchain
- langchain-community
- pydantic
- pytest
- pytest-asyncio
- pytest-cov

### Project Structure
```
app/
├── api/
│   └── routes.py
├── chains/
│   └── script_chain.py
├── context/
│   └── vector.py
├── models/
│   ├── config.py
│   ├── node_models.py
│   └── vector_store.py
├── nodes/
│   └── text_generation.py
├── utils/
│   ├── callbacks.py
│   ├── context.py
│   └── logging.py
└── vector/
    └── store.py
```

### Key Classes and Interfaces

#### ScriptChain
- Manages workflow execution
- Handles node dependencies
- Controls parallel execution
- Manages context and state

#### TextGenerationNode
- Executes text generation
- Manages LLM interactions
- Handles prompt templating
- Tracks token usage

#### VectorStore
- Abstract base class for vector storage
- Defines interface for similarity search
- Manages document storage and retrieval

#### ContextWindow
- Manages token limits
- Handles context truncation
- Tracks metadata

### Configuration System

#### LLMConfig
- Model configuration
- API settings
- Token limits
- Generation parameters

#### NodeConfig
- Node identification
- Prompt templates
- Dependencies
- Generation parameters

### Error Handling

#### Retry Mechanism
- Configurable retry attempts
- Exponential backoff
- Error type filtering
- State preservation

#### Error Types
- ValidationError
- ExecutionError
- ContextError
- VectorStoreError

### Testing Infrastructure

#### Test Fixtures
- ScriptChain setup
- Node configuration
- Vector store mocking
- Context management

#### Test Categories
- Initialization tests
- Execution tests
- Error handling tests
- Integration tests

## Development

### Testing
```bash
# Run all tests
python -m pytest

# Run specific test file
python -m pytest tests/test_script_chain.py

# Run with coverage
python -m pytest --cov=app tests/
```

### Code Style
- PEP 8 compliance
- Type hints
- Docstring documentation
- Clear error messages

### Error Handling
- Graceful degradation
- Detailed error messages
- State preservation
- Recovery mechanisms

## Technical Limitations

### Context Management
- Fixed token limits
- Context window constraints
- Metadata storage limits

### Vector Store
- Abstract interface only
- No default implementation
- Requires custom implementation

### Node System
- Single node type (TextGenerationNode)
- Fixed dependency model
- Synchronous execution only

### Error Handling
- Basic retry mechanism
- Limited error recovery
- State preservation constraints

## Future Considerations

### Potential Enhancements
- Additional node types
- Enhanced error recovery
- Extended context management
- Vector store implementations
- Async node execution
- Dynamic workflow modification
- Enhanced monitoring
- Performance optimization 