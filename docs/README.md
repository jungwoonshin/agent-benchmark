# Agent System

A sophisticated, multi-problem agent system in Python that can solve complex, multi-hop problems requiring multimodal inputs (images, audio, text), deep file parsing (PDF, Excel, .txt), and complex logical reasoning (math, riddles, data filtering).

The system uses LLM (OpenAI GPT-4) for all decision-making processes, avoiding hardcoded assumptions and enabling adaptive problem-solving.

## Architecture

The agent system implements a comprehensive architecture with the following modules:

### Core Components

```
src/
├── core/
│   ├── __init__.py                   # Core package exports
│   ├── agent.py                      # Main Agent class (orchestrator)
│   ├── llm_service.py               # LLM service for OpenAI API
│   ├── query_understanding.py        # Query Understanding Module
│   ├── problem_classifier.py         # Problem Classification System
│   ├── planner.py                    # Planning Module
│   ├── executor.py                   # Execution Engine
│   ├── search_result_processor.py   # Search Result Processing (NEW)
│   ├── state_manager.py              # Information State Manager
│   ├── reasoning_engine.py           # Reasoning Engine
│   ├── answer_synthesizer.py        # Answer Synthesis Module
│   ├── tool_belt.py                  # ToolBelt class with all tools
│   └── models.py                     # Data models (Attachment, SearchResult, RevisionData)
└── __init__.py                       # Main package exports
```

### System Modules

1. **Query Understanding Module**: Parses natural language, identifies requirements and dependencies
2. **Problem Classification System**: Classifies problems into types (Information Retrieval, Logical Deduction, Computational, etc.)
3. **Planning Module**: Generates execution strategies with dependency graphs
4. **Execution Engine**: Orchestrates tool operations based on the plan
5. **Information State Manager**: Tracks knowledge graph, progress, and context
6. **Reasoning Engine**: Handles pattern matching, constraint propagation, hypothesis generation
7. **Answer Synthesis Module**: Constructs final answers from partial results

## Features

### Two Types of Output

1. **Reasoning Monologue**: A human-readable, first-person "chain-of-thought" narrative returned as part of the `solve()` method output.
2. **Extensive Logging**: A detailed, developer-facing log stream for debugging, auditing, and tracing every tool call and internal decision.

### Available Tools

- `code_interpreter()`: Executes Python code in a sandboxed environment for math, logic, and data processing
- `search()`: Performs web or specialized searches
- `read_attachment()`: Smart file reader that extracts text from various formats (PDF, Excel, .txt)
- `analyze_media()`: Analyzes non-text media files (images, audio, video) using ML models
- `get_structured_data()`: Accesses specialized external APIs for structured data

## Setup

1. **Install dependencies**:
```bash
uv pip install -r requirements.txt
```

2. **Set up environment variables**:
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=gpt-5
```

## Usage

```python
import logging
from src.core import Agent, Attachment, ToolBelt

# Setup logging
logger = logging.getLogger('agent_system')
logging.basicConfig(level=logging.INFO)

# Create ToolBelt and Agent (with LLM support)
# Model is loaded from LLM_MODEL env var if not specified
tool_belt = ToolBelt()
agent = Agent(tool_belt=tool_belt, logger=logger)

# Solve a problem
problem = "What is 2 + 2? Show your work."
final_answer, reasoning_monologue = agent.solve(problem)

print(f"Answer: {final_answer}")
print(f"Reasoning: {reasoning_monologue}")
```

### With Attachments

```python
attachment = Attachment(
    filename="image.jpg",
    data=b"image_data_bytes"
)

problem = "What animal is in this image?"
final_answer, reasoning_monologue = agent.solve(problem, attachments=[attachment])
```

## Control Flow

The agent follows a structured control flow:

1. **PARSE**: Decompose query into atomic requirements using LLM
2. **CLASSIFY**: Classify problem type using LLM
3. **PLAN**: Generate execution strategy with dependency graph using LLM
4. **EXECUTE**: Run tool operations (respecting dependencies)
5. **MONITOR**: Track progress and detect issues
6. **SYNTHESIZE**: Construct answer from components using LLM
7. **VALIDATE**: Verify all requirements satisfied

## Example

See `example_usage.py` for a complete working example.

## Design Philosophy

The agent's "brain" is a **planner**. It solves problems by:
1. Breaking them down into smaller steps
2. Orchestrating a set of powerful tools
3. Using tools for:
   - **Data Retrieval**: Getting information it doesn't have (e.g., `search`, `read_attachment`, `get_structured_data`)
   - **Data Processing**: Performing complex logic, math, or data manipulation (e.g., `code_interpreter`)

The `reasoning_monologue` is the *narrative* of this plan. The `logger` is the *verbatim trace* of its execution.

## Key Features

- **Intelligent Retry Logic**: When validation fails, only failed subtasks are re-executed instead of the entire plan, reducing redundant computation by 50%+ (see [RETRY_OPTIMIZATION.md](RETRY_OPTIMIZATION.md))
- **Systematic Search Result Processing**: LLM-based relevance checking, automatic classification of web pages vs files, and intelligent dispatching to appropriate handlers (see [SEARCH_RESULT_PROCESSING.md](SEARCH_RESULT_PROCESSING.md))
- **LLM-Based Decision Making**: All strategic decisions are made by LLM, no hardcoded assumptions
- **Modular Architecture**: Each module is independent and composable
- **Adaptive Planning**: Execution plans are dynamically generated based on problem characteristics
- **State Tracking**: Comprehensive knowledge graph and progress monitoring
- **Error Handling**: Graceful degradation with partial solutions
- **Observability**: Extensive logging for debugging and auditing

## Design Principles

1. **Composability**: Each module works independently and combines flexibly
2. **Observability**: All decisions and reasoning steps are traceable
3. **Idempotency**: Repeated operations produce consistent results
4. **Graceful Degradation**: Partial solutions better than complete failure
5. **Resource Awareness**: Balance thoroughness with computational costs

## Status

The system is fully implemented with LLM-based decision making. All modules are functional and integrated. The system uses OpenAI GPT-4 by default but can be configured to use other models.

