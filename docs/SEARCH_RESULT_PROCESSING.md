# Search Result Processing

This document describes the systematic search result processing workflow implemented in the agent system.

## Overview

The `SearchResultProcessor` class provides a systematic approach to handling search results by:

1. **Checking Relevance** - Uses LLM to determine if each search result is relevant to the subtask
2. **Classifying Type** - Determines if the result is a web page or downloadable file
3. **Dispatching to Handler** - Routes to appropriate handler based on type
4. **Extracting Content** - Extracts and structures content for downstream processing

## Architecture

```
┌─────────────────┐
│   Executor      │
│  execute_subtask│
└────────┬────────┘
         │
         ├─ Search Tool
         │    └─> search_results[]
         │
         ├─ SearchResultProcessor
         │    │
         │    ├─ For each result:
         │    │   │
         │    │   ├─ 1. Check Relevance (LLM)
         │    │   │      └─> relevant: bool, reasoning: str
         │    │   │
         │    │   ├─ 2. Classify Type
         │    │   │      └─> is_file: bool, file_type: str
         │    │   │
         │    │   ├─ 3. Handle based on type:
         │    │   │   │
         │    │   │   ├─ If FILE:
         │    │   │   │   └─> download_file_from_url()
         │    │   │   │       └─> add to attachments
         │    │   │   │
         │    │   │   └─ If WEB PAGE:
         │    │   │       └─> browser_navigate()
         │    │   │           └─> extract content
         │    │   │
         │    │   └─ 4. Extract and structure content
         │    │
         │    └─> processing_result{
         │          processed_count,
         │          relevant_count,
         │          web_pages[],
         │          downloaded_files[],
         │          content_summary
         │        }
         │
         └─> result (to downstream tasks)
```

## Module Structure

Following the repository's modular design principles:

```
src/core/
├── search_result_processor.py   # NEW: Systematic search result handling
├── executor.py                   # Uses SearchResultProcessor
├── tool_belt.py                  # Provides browser_navigate, download_file_from_url
└── llm_service.py                # Provides LLM calls for relevance checking
```

## Usage

### Basic Usage in Executor

The `SearchResultProcessor` is automatically instantiated in the `Executor` and used for all search operations:

```python
# In Executor.__init__()
self.search_processor = SearchResultProcessor(
    llm_service=llm_service,
    tool_belt=tool_belt,
    logger=logger,
)

# In execute_subtask() when tool_name == 'search'
processing_result = self.search_processor.process_search_results(
    search_results=search_results,
    subtask_description=subtask.description,
    problem=problem,
    query_analysis=query_analysis,
    attachments=attachments,
    max_results_to_process=5,
)
```

### Processing Flow Example

```python
# Example: Processing search results for "Find Nature articles published in 2020"

# Input: search_results = [
#   SearchResult(title="Nature 2020 Archive", url="https://nature.com/archive/2020", ...),
#   SearchResult(title="2020 Report.pdf", url="https://example.com/report.pdf", ...),
#   SearchResult(title="Blog Post", url="https://blog.example.com", ...),
# ]

processing_result = search_processor.process_search_results(...)

# Output: processing_result = {
#   'processed_count': 3,
#   'relevant_count': 2,  # LLM determined blog post not relevant
#   'web_pages': [
#     {
#       'url': 'https://nature.com/archive/2020',
#       'title': 'Nature 2020 Archive',
#       'content': '...[extracted text from page]...'
#     }
#   ],
#   'downloaded_files': [
#     {
#       'url': 'https://example.com/report.pdf',
#       'type': 'pdf',
#       'content': '...[extracted text from PDF]...'
#     }
#   ],
#   'content_summary': '[Web Page: Nature 2020 Archive]...\n\n[File: 2020 Report.pdf]...'
# }
```

## Key Features

### 1. LLM-Based Relevance Checking

Each search result is evaluated for relevance using the LLM with:
- **Subtask context** - What the current subtask is trying to accomplish
- **Problem context** - The original problem being solved
- **Requirements** - Explicit and implicit requirements from query analysis
- **Constraints** - Temporal, spatial, and categorical constraints

```python
def _check_relevance(
    self,
    search_result: SearchResult,
    subtask_description: str,
    problem: str,
    query_analysis: Optional[Dict[str, Any]] = None,
) -> tuple[bool, str]:
    """
    Returns:
        (is_relevant: bool, reasoning: str)
    """
```

### 2. Intelligent Type Classification

Automatically classifies search results as:

**File Types:**
- `pdf` - PDF documents
- `doc` - Word documents (.doc, .docx)
- `spreadsheet` - Excel/CSV files
- `image` - Image files
- `archive` - Compressed archives
- `text` - Plain text files

**Non-File:**
- `webpage` - Regular web pages

Classification is based on:
- URL patterns (file extensions)
- Title analysis
- URL path indicators (/pdf/, /download/, etc.)

### 3. Appropriate Handler Dispatching

**For Files:**
- Downloads the file using `tool_belt.download_file_from_url()`
- Adds to `attachments` list for downstream tasks
- Extracts text content using `tool_belt.read_attachment()`

**For Web Pages:**
- Navigates using `tool_belt.browser_navigate()` (Selenium-based)
- Determines appropriate extraction action based on subtask:
  - `extract_count` - For counting/statistics tasks
  - `find_table` - For structured data extraction
  - `search_text` - For text search tasks
  - `extract_text` - Default text extraction
- Extracts structured content (text, tables, counts, etc.)

### 4. Content Structuring

All extracted content is structured into a unified format:

```python
{
    'processed_count': int,      # Total results processed
    'relevant_count': int,        # Results deemed relevant by LLM
    'web_pages': [                # Processed web pages
        {
            'url': str,
            'title': str,
            'content': str
        },
        ...
    ],
    'downloaded_files': [         # Downloaded files
        {
            'url': str,
            'type': str,
            'content': str
        },
        ...
    ],
    'content_summary': str        # Aggregated content from all sources
}
```

## Configuration

### Max Results to Process

Control how many search results are processed:

```python
processing_result = search_processor.process_search_results(
    search_results=search_results,
    ...,
    max_results_to_process=5,  # Default: 5
)
```

### Extraction Actions

The processor automatically determines extraction actions, but you can customize the logic in:

```python
def _determine_extraction_action(self, subtask_description: str) -> Optional[str]:
    """Map subtask keywords to extraction actions."""
```

## Benefits

### 1. **Systematic Processing**
Every search result is processed in a consistent, predictable way.

### 2. **Intelligent Filtering**
LLM-based relevance checking reduces noise and focuses on useful results.

### 3. **Automatic Type Handling**
Files are downloaded, web pages are navigated - all automatically.

### 4. **Content Extraction**
Extracts and structures content for easy use in downstream tasks.

### 5. **Modular Design**
Following repository principles:
- Single responsibility (search result processing)
- Separate module (`search_result_processor.py`)
- Clear interfaces
- Reusable across different contexts

## Integration with GAIA Dataset

This systematic approach maintains compatibility with GAIA dataset problems by:

1. **Preserving Tool Capabilities** - All existing tool functionality remains available
2. **Enhanced Multi-Source Handling** - Better handles problems requiring multiple information sources
3. **Improved File Handling** - Automatically downloads and processes files from search results
4. **Better Web Navigation** - Systematically navigates to relevant pages using Selenium

## Future Enhancements

Potential improvements:

1. **Parallel Processing** - Process multiple results concurrently
2. **Caching** - Cache relevance checks and content extraction
3. **Smart Retry** - Retry failed navigations/downloads with different strategies
4. **Content Ranking** - Rank extracted content by relevance for synthesis
5. **Format-Specific Handlers** - Specialized handlers for different file types

## See Also

- `src/core/executor.py` - Main execution engine
- `src/core/tool_belt.py` - Tool implementations
- `src/core/browser_navigator.py` - Web navigation utilities
- `SELENIUM_INTEGRATION.md` - Browser automation details

