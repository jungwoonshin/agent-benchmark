# PDF Processing Fix: Guide LLM to Use Search Queries

## Problem

The LLM was trying to generate Python code that imports `PyPDF2` to parse PDF files:
```
Execution Error: ImportError: Import of 'PyPDF2' is not allowed for security reasons
```

This happened because:
1. The LLM was instructed to use `code_interpreter` for various tasks
2. When it needed to process PDFs, it tried to import PDF parsing libraries
3. These libraries are not in the whitelist of safe modules

## Solution

Implemented a multi-layered approach to guide the LLM away from PDF parsing code and toward using search queries and attachment tools:

### 1. Enhanced Error Messages

**File**: `src/core/tool_belt.py`

Added specific error handling for PDF import errors that provides helpful suggestions:

```python
except ImportError as e:
    if 'PyPDF2' in error_msg or 'pdf' in error_msg.lower():
        suggestion = (
            "IMPORT ERROR: PDF processing libraries are not available. "
            "Instead of using code to parse PDFs, use the 'read_attachment' tool. "
            "If you need to extract information from a PDF, first search for the PDF or use the read_attachment tool with the attachment index."
        )
        return f'{error_msg}\n\n💡 SUGGESTION: {suggestion}'
```

### 2. Updated Planner Prompts

**File**: `src/core/planner.py`

Added explicit guidance in the planner system prompt:

- **CRITICAL**: When you need to find PDFs or extract information from PDFs, use search with specific queries. The system will automatically download PDFs and extract text content. DO NOT try to parse PDFs with code_interpreter.
- **For PDF processing**: Use search to find and download PDFs, then use read_attachment to extract information. DO NOT use code_interpreter to parse PDFs - PDF parsing libraries are not available.
- **code_interpreter**: Use ONLY for computation, data processing, and analysis (using built-in Python functions). DO NOT use for PDF parsing or file operations - use search and read_attachment instead.

### 3. Updated Parameter Determination Prompts

**File**: `src/core/executor.py`

Added critical constraints to the parameter determination system prompt:

```
CRITICAL CONSTRAINTS:
- **code_interpreter limitations**: Can only use built-in Python functions and whitelisted modules (math, json, datetime, re, etc.). 
  - CANNOT import: PyPDF2, pdfplumber, os, sys, subprocess, or any file parsing libraries
  - For PDF processing: DO NOT generate code that imports PDF libraries. Instead, suggest using search + read_attachment tools
  - For file operations: Use search to find files, then read_attachment to extract content
- **Use search for information gathering**: When you need to find PDFs, documents, or information, generate a specific search query instead of code
- **PDF processing**: Use search to locate PDFs, then read_attachment to extract text. Never try to parse PDFs with code.
```

Also added guidance for generating better search queries:
```
For search: {"query": "search query", "num_results": 5, "search_type": "web"}
  - Generate specific, detailed search queries that include key terms, dates, and requirements
  - Example: Instead of "AI papers", use "arXiv AI regulation papers submitted June 2022"
```

### 4. Automatic Fallback to Search

**File**: `src/core/executor.py`

Added automatic detection and conversion when code execution attempts PDF parsing:

```python
# Check if result indicates PDF import error - convert to search if needed
if isinstance(result, str) and ('PyPDF2' in result or 'pdfplumber' in result.lower() or 'pdf processing' in result.lower()):
    # Convert to search automatically
    search_query = subtask.description
    if 'pdf' not in search_query.lower():
        search_query = f"{search_query} PDF"
    # Execute search and process results
```

## Workflow After Fix

### Before (Broken):
1. LLM generates code: `import PyPDF2; pdf = PyPDF2.PdfFileReader(...)`
2. Code execution fails: `ImportError: Import of 'PyPDF2' is not allowed`
3. Task fails with error

### After (Fixed):
1. **LLM Guidance**: Planner prompts guide LLM to use search for PDFs
2. **Parameter Generation**: Parameter determination prevents PDF import code
3. **Error Handling**: If PDF import is attempted, clear error message with suggestion
4. **Automatic Fallback**: System automatically converts PDF parsing attempts to search
5. **Search Execution**: System executes search query and processes results (downloads PDFs, extracts text)

## Example: Finding arXiv Papers

### Before:
```python
# LLM might try:
import PyPDF2
# Error: Import not allowed
```

### After:
```json
{
  "tool": "search",
  "query": "arXiv AI regulation papers submitted June 2022",
  "num_results": 5,
  "search_type": "web"
}
```

The system will:
1. Execute the search
2. Find relevant arXiv papers
3. Download PDFs automatically
4. Extract text content
5. Make content available via read_attachment

## Files Modified

1. **src/core/tool_belt.py** (lines 237-258)
   - Added ImportError handling with PDF-specific suggestions

2. **src/core/planner.py** (lines 65-97, 105-109)
   - Added PDF processing guidance to tool selection guidelines
   - Updated tool descriptions to emphasize search for PDFs

3. **src/core/executor.py** (lines 344-362, 126-165)
   - Added critical constraints to parameter determination
   - Added automatic fallback from PDF parsing to search
   - Enhanced search query generation guidance

## Benefits

1. **Prevents Errors**: LLM won't try to import restricted modules
2. **Better Guidance**: Clear instructions on when to use search vs code
3. **Automatic Recovery**: System automatically converts PDF parsing attempts to search
4. **Better Search Queries**: Guidance encourages specific, detailed search queries
5. **Proper Workflow**: Uses the intended workflow: search → download → read_attachment

## Testing

The fix has been implemented and should:
- ✅ Prevent PDF import errors
- ✅ Guide LLM to generate search queries instead
- ✅ Automatically fallback to search if PDF parsing is attempted
- ✅ Generate more specific search queries with dates and requirements

