# Search Content Integration Fix

## Issue

The user asked: **"Is search result of web page and file content being included in synthesis step?"**

## Current State

YES, the content IS being passed to synthesis, but it's not highlighted prominently enough.

### Data Flow

1. **SearchResultProcessor** extracts content:
   ```python
   processing_result = {
       'processed_count': 5,
       'relevant_count': 3,
       'web_pages': [
           {'url': '...', 'title': '...', 'content': '...[extracted text]...'}
       ],
       'downloaded_files': [
           {'url': '...', 'type': 'pdf', 'content': '...[extracted PDF text]...'}
       ],
       'content_summary': '[Web Page: ...]...\n\n[File: ...]...'  # ← KEY FIELD!
   }
   ```

2. **Executor** packages this into result:
   ```python
   result = {
       'search_results': search_results,  # Original results (just URLs/snippets)
       'processing_summary': processing_result,
       'content': processing_result.get('content_summary', ''),  # ← EXTRACTED CONTENT
       'web_pages': processing_result.get('web_pages', []),
       'downloaded_files': processing_result.get('downloaded_files', []),
   }
   ```

3. **Synthesis** receives execution_results which includes this dict

### Problem

The `_format_result_content()` method in `answer_synthesizer.py` doesn't have special handling for this structure, so it just JSON dumps the entire dict. The `content` field (which contains all the extracted web page and file content) is buried in the JSON.

## Solution

Add special handling in `_format_result_content()` to highlight search result content prominently.

### Implementation

In `/Users/jungwoonshin/github/agent-system/src/core/answer_synthesizer.py` around line 523, add this BEFORE the browser_navigate error handling:

```python
# Special handling for search results with processed content (from SearchResultProcessor)
if 'content' in result and 'processing_summary' in result:
    highlighted = '=== PROCESSED SEARCH RESULTS (WEB PAGES + FILES) ===\n'
    
    # Show summary stats
    summary = result.get('processing_summary', {})
    highlighted += f"Processed: {summary.get('processed_count', 0)} results\n"
    highlighted += f"Relevant: {summary.get('relevant_count', 0)} results\n"
    highlighted += f"Web Pages: {len(result.get('web_pages', []))} navigated\n"
    highlighted += f"Files: {len(result.get('downloaded_files', []))} downloaded\n\n"
    
    # Show the aggregated content (this is the key extracted content!)
    content = result.get('content', '')
    if content:
        highlighted += '=== EXTRACTED CONTENT FROM WEB PAGES AND FILES ===\n'
        # Limit length to avoid token limits
        max_content_length = 3000
        if len(content) > max_content_length:
            highlighted += content[:max_content_length] + '\n\n...[content truncated for length]...\n'
        else:
            highlighted += content + '\n'
    else:
        highlighted += '[No content extracted]\n'
    
    highlighted += '\n=== FULL RESULT DETAILS ===\n'
    highlighted += json.dumps(
        result, indent=2, ensure_ascii=False, default=str
    )[:2000]  # Limit JSON dump to avoid overwhelming output
    return highlighted
```

### Location

Insert this code in `answer_synthesizer.py` at line 523, right before this line:
```python
# Special handling for browser_navigate errors - show diagnostics clearly
if not result.get('success', True) and 'diagnostics' in result:
```

## Benefits

1. ✅ **Content is Prominently Displayed** - Extracted web page and file content shown at top
2. ✅ **Clear Section Headers** - "EXTRACTED CONTENT FROM WEB PAGES AND FILES" stands out
3. ✅ **Summary Stats** - Shows how many results were processed, how many web pages navigated, how many files downloaded
4. ✅ **Length Management** - Truncates very long content to avoid token limits
5. ✅ **Better Synthesis** - LLM can easily find and use the extracted content

## Example Output

### Before (Content Buried in JSON)
```
Subtask step_1:
{
  "search_results": [...],
  "processing_summary": {...},
  "content": "very long text buried here...",
  "web_pages": [...],
  "downloaded_files": [...]
}
```

### After (Content Highlighted)
```
Subtask step_1:
=== PROCESSED SEARCH RESULTS (WEB PAGES + FILES) ===
Processed: 5 results
Relevant: 3 results
Web Pages: 2 navigated
Files: 1 downloaded

=== EXTRACTED CONTENT FROM WEB PAGES AND FILES ===
[Web Page: ArXiv AI Regulation Paper]
URL: https://arxiv.org/pdf/2207.01510
Content:
This paper discusses AI regulation in the European Union...
[Figure shows three axes with labels: X-axis: individualism-collectivism, 
Y-axis: hierarchical-egalitarian, Z-axis: freedom-control]

[File: supplementary-data.pdf]
Additional findings about regulatory frameworks...

=== FULL RESULT DETAILS ===
{...truncated JSON...}
```

## Impact

With this fix:
- ✅ Web page content from Selenium navigation is visible to synthesis
- ✅ Downloaded file content is visible to synthesis
- ✅ Content is prominently displayed, not buried in JSON
- ✅ LLM can easily extract information for answer synthesis
- ✅ Better success rate on problems requiring web navigation or file downloads

## Testing

To verify:
1. Run a test that searches and navigates to web pages
2. Check synthesis logs for "EXTRACTED CONTENT FROM WEB PAGES AND FILES"
3. Verify final answer uses content from navigated pages/downloaded files

```bash
uv run python test_validation.py
grep "EXTRACTED CONTENT FROM WEB PAGES AND FILES" logs/log.txt
```

