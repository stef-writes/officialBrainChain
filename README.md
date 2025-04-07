# Gaffer - AI Workflow Orchestration System

Gaffer is a FastAPI-based workflow orchestration system for AI-powered text generation. It provides a flexible and extensible framework for building, executing, and monitoring AI workflows.

## Features

- **Node-Based Workflows**: Create workflows using a node-based architecture
- **AI Integration**: Seamlessly integrate with LLM providers like OpenAI
- **Extensible**: Plugin system for adding custom nodes and integrations
- **Version Control**: Version-controlled templates and configurations
- **Monitoring**: Track execution statistics and performance
- **Error Handling**: Robust error handling and retry mechanisms

## Project Structure

- `app/`: Main application code
  - `api/`: API routes and endpoints
  - `chains/`: Workflow chain implementations
  - `config/`: Configuration management
  - `core/`: Core functionality
  - `middleware/`: FastAPI middleware
  - `models/`: Data models
  - `nodes/`: Node implementations
  - `plugins/`: Plugin system
  - `schemas/`: API schemas
  - `services/`: Business logic services
  - `utils/`: Utility functions
- `examples/`: Example code
- `tests/`: Testing framework
  - `unit/`: Unit tests
  - `integration/`: Integration tests
  - `e2e/`: End-to-end tests
  - `fixtures/`: Test fixtures
  - `output/`: Test output

## Getting Started

### Prerequisites

- Python 3.9+
- OpenAI API key (or other LLM provider)

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -e .
   ```
3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

### Running the Application

```bash
uvicorn app.main:app --reload
```

### Running Tests

```bash
./run_tests.sh
```

## Usage

See the `examples/` directory for usage examples.

## License

MIT 