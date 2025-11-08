# Agent System

A sophisticated, multi-problem agent system in Python designed to solve complex, multi-hop problems requiring multimodal inputs (images, audio, text), deep file parsing (PDF, Excel, .txt), and complex logical reasoning (math, riddles, data filtering).

The system uses Large Language Models (LLMs) for all decision-making processes, avoiding hardcoded assumptions and enabling adaptive problem-solving. It is optimized for GAIA dataset benchmark compatibility and general agent-related problem solving.

## Overview

The Agent System is built around a hierarchical planning architecture where an LLM-powered planner breaks down complex problems into subtasks, orchestrates a suite of specialized tools, and synthesizes final answers from partial results. The system maintains a knowledge graph throughout execution, tracks progress, and implements intelligent retry logic for failed subtasks.

## Architecture

### Core Components

```
src/
├── core/
│   └── agent.py                      # Main Agent class (orchestrator)
├── planning/
│   ├── query_understanding.py        # Query Understanding Module
│   ├── problem_classifier.py         # Problem Classification System
│   └── planner.py                    # Planning Module
├── execution/
│   ├── executor.py                   # Execution Engine
│   ├── search_handler.py             # Search operation handler
│   └── result_analyzer.py            # Result analysis and validation
├── state/
│   └── state_manager.py              # Information State Manager
├── synthesis/
│   ├── answer_synthesizer.py         # Answer Synthesis Module
│   └── validation.py                 # Answer Validation
├── llm/
│   └── llm_service.py                # LLM service for API interactions
├── tools/
│   ├── tool_belt.py                  # ToolBelt class with all tools
│   ├── search_tool.py                 # Web search tool
│   ├── browser_tool.py                # Web browser automation
│   ├── file_handler.py                # File reading and processing
│   ├── image_recognition.py           # Image analysis
│   ├── llm_reasoning.py               # LLM-based reasoning and computation
│   └── context_extractor.py          # Context extraction utilities
├── browser/
│   ├── search_result_processor.py     # Search result processing
│   ├── relevance_checker.py           # Relevance checking
│   └── content_type_classifier.py     # Content type detection
├── code_interpreter/
│   └── interpreter.py                 # Code execution utilities
├── models/
│   └── models.py                      # Data models (Attachment, SearchResult, etc.)
└── solver.py                          # GAIASolver wrapper for benchmark compatibility
```

### System Modules

1. **Query Understanding Module** (`query_understanding.py`): Parses natural language, identifies requirements, dependencies, and answer format specifications
2. **Problem Classification System** (`problem_classifier.py`): Classifies problems into types (Information Retrieval, Logical Deduction, Computational, Cross-Reference, etc.)
3. **Planning Module** (`planner.py`): Generates execution strategies with dependency graphs, handles hierarchical task decomposition
4. **Execution Engine** (`executor.py`): Orchestrates tool operations based on the plan, manages parallel and sequential execution
5. **Information State Manager** (`state_manager.py`): Tracks knowledge graph, progress, and context throughout execution
6. **Answer Synthesis Module** (`answer_synthesizer.py`): Constructs final answers from partial results using LLM reasoning
7. **Answer Validator** (`validation.py`): Validates answers against requirements and triggers retry logic for failed subtasks

## Features

### Intelligent Problem Solving

- **LLM-Based Decision Making**: All strategic decisions (planning, classification, synthesis) are made by LLM, no hardcoded assumptions
- **Hierarchical Task Decomposition**: Complex problems are broken down into manageable subtasks with dependency tracking
- **Adaptive Planning**: Execution plans are dynamically generated based on problem characteristics and intermediate results
- **Intelligent Retry Logic**: When validation fails, only failed subtasks are re-executed instead of the entire plan, reducing redundant computation by 50%+

### Multimodal Capabilities

- **Text Processing**: Deep parsing of PDFs, Excel files, text documents with section-aware extraction
- **Image Analysis**: Visual understanding using LLM vision capabilities
- **Web Browsing**: Automated web navigation with content extraction and relevance checking
- **File Downloads**: Automatic detection and handling of downloadable files from web pages

### Search and Information Retrieval

- **Systematic Search Result Processing**: LLM-based relevance checking, automatic classification of web pages vs files, intelligent dispatching to appropriate handlers
- **Content Type Detection**: Automatic classification of web content (HTML pages, PDFs, images, etc.)
- **Relevance Filtering**: Intelligent filtering of search results and document sections based on query relevance

### State Management

- **Knowledge Graph**: Maintains a graph of discovered facts with source attribution
- **Progress Tracking**: Monitors completed subtasks and pending requirements
- **Context Preservation**: Maintains context across multiple tool invocations

## Available Tools

The system provides a comprehensive set of tools through the `ToolBelt` class:

- **`llm_reasoning()`**: Performs calculations, data processing, and analysis using LLM reasoning (replaces deprecated code_interpreter)
- **`llm_reasoning_with_images()`**: Visual reasoning with image inputs
- **`search()`**: Performs web searches using Google Custom Search API
- **`read_attachment()`**: Smart file reader that extracts text and images from various formats (PDF, Excel, .txt, images)
- **`analyze_media()`**: Analyzes non-text media files (images, audio, video) using ML models
- **`browse()`**: Automated web browsing with content extraction
- **`download_file_from_url()`**: Downloads files from URLs with automatic format detection

## Setup

### Prerequisites

- Python 3.8+
- `uv` package manager (for dependency management)

### Installation

1. **Install dependencies using `uv`**:
```bash
uv pip install -r requirements.txt
```

2. **Set up environment variables**:
Create a `.env` file in the project root:
```
OPENAI_API_KEY=your_openai_api_key_here
LLM_MODEL=openai/gpt-oss-120b
GOOGLE_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_google_cse_id_here
```

The `LLM_MODEL` defaults to `openai/gpt-oss-120b` if not specified. For visual reasoning tasks, the system automatically uses appropriate vision-capable models.

## Usage

### Basic Usage

```python
import logging
from src.core import Agent, Attachment, ToolBelt

# Setup logging
logger = logging.getLogger('agent_system')
logging.basicConfig(level=logging.INFO)

# Create ToolBelt and Agent
tool_belt = ToolBelt()
agent = Agent(tool_belt=tool_belt, logger=logger)

# Solve a problem
problem = "What is the capital of France?"
final_answer = agent.solve(problem)

print(f"Answer: {final_answer}")
```

### With Attachments

```python
attachment = Attachment(
    filename="document.pdf",
    data=b"pdf_data_bytes"
)

problem = "What is the main topic discussed in this document?"
final_answer = agent.solve(problem, attachments=[attachment])
```

### GAIA Benchmark Compatibility

For GAIA benchmark evaluation, use the `GAIASolver` wrapper:

```python
from src.solver import GAIASolver

solver = GAIASolver()
result = solver.solve(question="Your question here", attachments=[...])

print(f"Answer: {result.answer}")
print(f"Confidence: {result.confidence}")
print(f"Sources: {result.sources}")
```

## Control Flow

The agent follows a structured control flow:

1. **PARSE**: Decompose query into atomic requirements using LLM
2. **CLASSIFY**: Classify problem type using LLM
3. **PLAN**: Generate execution strategy with dependency graph using LLM
4. **EXECUTE**: Run tool operations (respecting dependencies, supporting parallel execution)
5. **MONITOR**: Track progress and detect issues
6. **SYNTHESIZE**: Construct answer from components using LLM
7. **VALIDATE**: Verify all requirements satisfied, retry failed subtasks if needed

## Design Philosophy

The agent's "brain" is a **planner**. It solves problems by:

1. Breaking them down into smaller steps with clear dependencies
2. Orchestrating a set of powerful tools
3. Using tools for:
   - **Data Retrieval**: Getting information it doesn't have (e.g., `search`, `read_attachment`, `browse`)
   - **Data Processing**: Performing complex logic, math, or data manipulation (e.g., `llm_reasoning`)

The system maintains comprehensive logging for debugging, auditing, and tracing every tool call and internal decision.

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
6. **GAIA Compatibility**: Maintain compatibility with GAIA dataset benchmark requirements

## Testing

Run validation tests on GAIA dataset:

```bash
python test_validation.py
```

The test script supports loading specific cases by index or running a subset of cases for quick validation.

## Status

The system is fully implemented with LLM-based decision making. All modules are functional and integrated. The system uses OpenAI-compatible models by default (configurable via `LLM_MODEL` environment variable) and maintains compatibility with the GAIA benchmark evaluation framework.

## Documentation

Additional documentation is available in the `docs/` directory:

- [RETRY_OPTIMIZATION.md](RETRY_OPTIMIZATION.md): Details on intelligent retry logic
- [SEARCH_RESULT_PROCESSING.md](SEARCH_RESULT_PROCESSING.md): Search result processing workflow
- [SEARCH_FIRST_WORKFLOW.md](SEARCH_FIRST_WORKFLOW.md): Search-first execution strategy
- [QUICK_START_SEARCH_FIRST.md](QUICK_START_SEARCH_FIRST.md): Quick start guide
