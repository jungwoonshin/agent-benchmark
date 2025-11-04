# Quick Start: Search-First Workflow

## What Changed?

Your agent system now **automatically searches first** for every subtask that needs information. Here's what you need to know:

## TL;DR

✅ **Every information-gathering subtask now starts with a search**  
✅ **Search results are evaluated by LLM for relevance**  
✅ **Web pages are navigated automatically with Selenium**  
✅ **Files are downloaded and processed automatically**  
✅ **No code changes needed - it works automatically**

## How to Use

### Nothing Changes for You!

The system handles everything automatically:

```python
# Your existing code works as-is
from src.core import Agent, ToolBelt

tool_belt = ToolBelt()
agent = Agent(tool_belt=tool_belt, logger=logger)

# This now automatically:
# 1. Searches for relevant information
# 2. Evaluates each result with LLM
# 3. Navigates to web pages or downloads files
# 4. Extracts and structures content
answer, reasoning = agent.solve("Find arXiv papers from June 2022")
```

## What Happens Behind the Scenes

### Old Behavior
```
Subtask: "Navigate to arXiv and find papers"
Tool: browser_navigate
→ Tries to navigate directly
→ Might fail if URL wrong
→ No alternatives
```

### New Behavior
```
Subtask: "Navigate to arXiv and find papers"
Tool: Automatically converted to 'search'
→ Searches: "arXiv papers June 2022"
→ Finds multiple results
→ LLM evaluates each for relevance
→ Navigates to relevant pages
→ Downloads any PDFs found
→ Returns structured content
```

## Key Benefits

### 1. More Reliable
- Searches find current/working URLs
- Multiple options if one fails
- Adapts to changed resources

### 2. More Intelligent
- LLM filters irrelevant results
- Focuses on high-value information
- Better resource utilization

### 3. More Automatic
- No manual URL construction
- Auto-classifies files vs web pages
- Handles downloads automatically
- Extracts content automatically

## Monitoring

### Check Your Logs

You'll see new log messages like:

```
Converting tool "browser_navigate" to search-first workflow.
Processing 5 search results systematically...
Result [1/5] "ArXiv Papers" is RELEVANT. Reason: Directly addresses the task...
Result [1/5] is a web page. Navigating with Selenium...
Successfully extracted content from https://arxiv.org/...
```

### Verify It's Working

Look for these indicators:
- ✅ "Converting tool..." messages (automatic conversion happening)
- ✅ "Processing X search results..." (search execution)
- ✅ "Result is RELEVANT..." (LLM evaluation)
- ✅ "Navigating..." or "Downloading..." (appropriate handling)
- ✅ Content extraction logs

## Examples

### Example 1: Finding Papers
```python
problem = "Find the AI regulation paper submitted to arXiv in June 2022"
answer, _ = agent.solve(problem)

# Behind the scenes:
# 1. Searches "arXiv AI regulation papers June 2022"
# 2. Finds relevant results
# 3. LLM confirms relevance
# 4. Navigates to arXiv with Selenium
# 5. Extracts paper information
# 6. Downloads PDF if needed
# 7. Returns structured data
```

### Example 2: Getting Statistics
```python
problem = "How many Nature articles were published in 2020?"
answer, _ = agent.solve(problem)

# Behind the scenes:
# 1. Searches "Nature articles published 2020"
# 2. Finds Nature archive page
# 3. LLM confirms relevance
# 4. Navigates with Selenium
# 5. Extracts count from page
# 6. Returns answer
```

### Example 3: Processing Files
```python
problem = "Analyze the climate report from 2020"
answer, _ = agent.solve(problem)

# Behind the scenes:
# 1. Searches "climate report 2020 PDF"
# 2. Finds PDF URL
# 3. Classifies as file
# 4. Downloads automatically
# 5. Extracts text
# 6. Processes content
```

## Troubleshooting

### Issue: Search Returns No Results

**Check:** Is Google API key configured?
```bash
# In .env file:
GOOGLE_API_KEY=your_key_here
GOOGLE_CX=your_cx_here
```

### Issue: Navigation Failing

**Check:** Is Selenium/ChromeDriver installed?
```bash
uv pip install selenium
```

### Issue: Want to Debug

**Enable debug logging:**
```python
logging.basicConfig(level=logging.DEBUG)
```

## Configuration

### Adjust Processing Limits

Edit `src/core/executor.py`:
```python
processing_result = self.search_processor.process_search_results(
    ...
    max_results_to_process=5,  # Change this number
)
```

### Adjust Search Results Count

Edit `src/core/executor.py`:
```python
num_results = parameters.get('num_results', 5)  # Change default here
```

## Documentation

For more details, see:

- **`SEARCH_FIRST_WORKFLOW.md`** - Complete workflow explanation
- **`SEARCH_RESULT_PROCESSING.md`** - Technical documentation
- **`CHANGES_SUMMARY.md`** - What changed and why
- **`WORKFLOW_DIAGRAM.txt`** - Visual workflow diagram

## Need Help?

### View Logs
```bash
tail -f logs/log.txt
```

### Run Examples
```bash
uv run python example_search_processing.py
```

### Run Tests
```bash
uv run python test_validation.py
```

## Summary

Your agent now:
1. ✅ **Always searches first** for information
2. ✅ **Evaluates results intelligently** with LLM
3. ✅ **Handles files and web pages automatically**
4. ✅ **Works without code changes** on your part
5. ✅ **Provides better reliability** and results

Just use the agent as before - the search-first workflow runs automatically! 🚀

