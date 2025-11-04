# Search-First Workflow Implementation

## Overview

The agent system now enforces a **search-first workflow** for all subtasks. Every subtask that needs to gather information will:

1. **Always start with a search** using the subtask description
2. **Process each search result** systematically:
   - Check relevance using LLM
   - Classify as web page or file
   - Navigate to web pages using Selenium
   - Download files and extract content
3. **Return structured results** for downstream processing

## Key Principle

**"Search First, Then Process"**

Instead of directly navigating to websites or trying to access resources, the system:
- Searches for relevant information
- Evaluates what it finds
- Takes appropriate action based on what's discovered

## Implementation Details

### 1. Executor Changes (`src/core/executor.py`)

Added logic to convert any tool (except `code_interpreter`, `read_attachment`, `analyze_media`) to the search-first workflow:

```python
# IMPORTANT: For most subtasks, always search first, then process results
# Only skip search for: code_interpreter, read_attachment, analyze_media
should_search_first = tool_name not in [
    'code_interpreter',
    'read_attachment',
    'analyze_media',
]

# If tool is browser_navigate or unknown, convert to search
if should_search_first and tool_name != 'search':
    self.logger.info(
        f'Converting tool "{tool_name}" to search-first workflow. '
        f'Will search with subtask description, then process results.'
    )
    tool_name = 'search'
    # Use subtask description as query if no query provided
    if 'query' not in parameters:
        parameters['query'] = subtask.description
```

**What this means:**
- If planner assigns `browser_navigate` → Converted to `search`
- If planner assigns unknown tool → Converted to `search`
- Subtask description becomes the search query
- Search results are then processed systematically

### 2. Planner Changes (`src/core/planner.py`)

Updated planner prompts to guide the LLM to prefer `search` over `browser_navigate`:

**Old approach:**
```
- browser_navigate: Use when you need to navigate to a specific website...
- search: Use when you need to discover information sources...
```

**New approach:**
```
- search: ALWAYS PREFERRED as the primary tool for information gathering
  * After search, the system will automatically:
    - Check relevance of each result using LLM
    - Classify results as web pages or files
    - Navigate to web pages using browser automation
    - Download files and extract content
  * Search handles ALL scenarios: archives, databases, websites, files, documents

- browser_navigate: DEPRECATED - Do not use directly
```

**Key changes:**
1. Search is now the primary tool for ALL information gathering
2. Browser navigation happens automatically within search processing
3. Planner is instructed NOT to use `browser_navigate` directly
4. Retry context updated to suggest better search queries instead of switching tools

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    SUBTASK EXECUTION                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ Check tool type             │
         │                             │
         │ • code_interpreter? → Run   │
         │ • read_attachment? → Read   │
         │ • analyze_media? → Analyze  │
         │ • OTHER? → Convert to       │
         │           SEARCH            │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ Execute Search              │
         │ (with subtask description)  │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ SearchResultProcessor       │
         │                             │
         │ For each result:            │
         │  1. Check relevance (LLM)   │
         │  2. Classify type           │
         │  3. If web page → Navigate  │
         │  4. If file → Download      │
         │  5. Extract content         │
         └─────────────┬───────────────┘
                       │
                       ▼
         ┌─────────────────────────────┐
         │ Return Structured Results   │
         │                             │
         │ • Web pages (navigated)     │
         │ • Files (downloaded)        │
         │ • Content summary           │
         │ • Relevance filtering       │
         └─────────────────────────────┘
```

## Examples

### Example 1: Finding arXiv Papers

**Old approach (could fail):**
```
Tool: browser_navigate
URL: https://arxiv.org
Action: Navigate and search
```

**New approach (always works):**
```
Tool: search
Query: "arXiv AI regulation papers submitted June 2022"

→ Search finds relevant URLs
→ LLM checks relevance
→ System navigates to arXiv automatically
→ Extracts paper information
→ Downloads PDFs if needed
```

### Example 2: Getting Statistics from Website

**Old approach:**
```
Tool: browser_navigate
URL: https://nature.com/archive/2020
Action: extract_count
```

**New approach:**
```
Tool: search
Query: "Nature articles published in 2020"

→ Search finds Nature archive page
→ LLM confirms relevance
→ System navigates with Selenium
→ Extracts count from page
→ Returns structured data
```

### Example 3: Accessing Files

**Old approach:**
```
Tool: browser_navigate
URL: https://example.com/report.pdf
Action: download
```

**New approach:**
```
Tool: search
Query: "climate change report 2020 PDF"

→ Search finds PDF URL
→ System classifies as file
→ Downloads automatically
→ Extracts text content
→ Adds to attachments
```

## Benefits

### 1. **Consistency**
Every information-gathering subtask follows the same pattern:
- Search → Evaluate → Process → Return

### 2. **Robustness**
- If direct URL doesn't work, search finds alternatives
- If resource moved, search finds new location
- If format changes, system adapts

### 3. **Intelligence**
- LLM evaluates relevance before processing
- Filters out non-relevant results
- Focuses effort on useful information

### 4. **Automatic Handling**
- Web pages: Navigated automatically with Selenium
- Files: Downloaded and processed automatically
- Archives: Searched and navigated automatically
- No manual URL construction needed

### 5. **Better Error Recovery**
- If direct navigation fails, search provides alternatives
- If file download fails, other sources may be found
- If resource unavailable, related resources discovered

## Edge Cases Handled

### Case 1: Planner Still Uses browser_navigate
**Solution:** Executor automatically converts to search workflow

### Case 2: Direct URL Known
**Solution:** Search still finds it, plus alternatives

### Case 3: Complex Multi-Step Navigation
**Solution:** Search finds entry points, processor handles navigation

### Case 4: File vs Web Page Ambiguity
**Solution:** SearchResultProcessor classifies automatically

### Case 5: Multiple Relevant Sources
**Solution:** LLM ranks and selects most relevant

## Configuration

No configuration needed - the workflow is enforced automatically:

- ✅ Planner guided to use `search`
- ✅ Executor converts other tools to `search`
- ✅ SearchResultProcessor handles all result types
- ✅ Logging shows conversion when it happens

## Debugging

### Check Logs

Look for these log messages:

```
Converting tool "browser_navigate" to search-first workflow.
Will search with subtask description, then process results.
```

```
Processing X search results systematically...
```

```
Result [1/X] Title is RELEVANT. Reason: ...
```

```
Result [1/X] Title is a web page. Navigating with Selenium...
```

```
Result [2/X] Title is a file (type: pdf). Downloading...
```

### Verify Search-First Behavior

1. Check that search happens first for information-gathering tasks
2. Verify results are processed (relevance checked)
3. Confirm appropriate handlers used (navigate or download)
4. Review content extraction quality

## Compatibility

### Backward Compatibility

- ✅ Existing plans still work (converted automatically)
- ✅ Old browser_navigate calls converted to search
- ✅ All tools remain available
- ✅ GAIA dataset compatibility maintained

### Future Plans

- Plans generated going forward will use `search` directly
- Over time, all subtasks will naturally follow search-first pattern
- No migration needed

## Summary

The search-first workflow ensures:

1. **Every subtask searches first** (except code/attachment operations)
2. **Results are evaluated intelligently** using LLM
3. **Appropriate handlers are dispatched automatically** (navigate or download)
4. **Content is extracted and structured** for downstream use
5. **The system is more robust** and adaptive to changes

This creates a **consistent, intelligent, and reliable** approach to information gathering that works across all types of resources: web pages, files, archives, databases, and more.

