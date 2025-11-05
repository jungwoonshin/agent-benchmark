# Dependency Access KeyError Fix

## Issue Description

The system was experiencing a `KeyError: 'step_2'` when executing code_interpreter tasks that reference dependency results. The error occurred in the `guarded_getitem` function when trying to access context variables.

### Error Traceback
```
KeyError: 'step_2'
Traceback (most recent call last):
  File "/Users/jungwoonshin/github/agent-system/src/core/tool_belt.py", line 352, in code_interpreter
    exec(byte_code, execution_globals, safe_locals)
  File "<string>", line 5, in <module>
  File "/Users/jungwoonshin/github/agent-system/src/core/tool_belt.py", line 25, in guarded_getitem
    return container[key]
KeyError: 'step_2'
```

## Root Cause

The issue was caused by a mismatch between how dependency results were stored in the context and how the LLM-generated code was trying to access them:

1. **Subtask ID Format**: Subtask IDs are generated as `step_1`, `step_2`, etc. (with underscores)
2. **Storage Format**: The code was storing dependency results with simplified keys by removing underscores:
   ```python
   simplified_key = dep_id.replace('step_', 'step')  # step_1 -> step1
   context[simplified_key] = result
   ```
3. **Access Pattern**: The LLM was generating code that tried to access `context['step_2']` (with underscore)
4. **Result**: The key `step_2` didn't exist in context, only `step2` (without underscore)

## Fixes Applied

### 1. Executor.py - Context Key Storage (Lines 137-209)

**Fixed**: Added dependency results under BOTH original AND simplified keys for maximum compatibility:

```python
# Add with original key (e.g., step_1, step_2)
context[dep_id] = result
# Also add with simplified key (e.g., step1, step2) for backward compatibility
simplified_key = dep_id.replace('step_', 'step')
if simplified_key != dep_id:
    context[simplified_key] = result
```

**Added**: Comprehensive dependency validation before code execution:
- Track missing dependencies (failed or incomplete)
- Prevent execution if dependencies are not satisfied
- Clear error messages indicating which dependencies are missing
- Logging of available context keys for debugging

### 2. Executor.py - LLM Prompt Update (Line 602)

**Fixed**: Updated the system prompt to accurately reflect how dependencies are accessible:

**Before**:
```
context['step1'] or context['dependency_results']['step_1']
```

**After**:
```
context['step_1'] or context['dependency_results']['step_1']
```

This clarifies that dependencies use their ORIGINAL names (with underscores) at the root level of context.

### 3. Tool_belt.py - Improved Error Messages (Lines 23-33)

**Fixed**: Enhanced `guarded_getitem` to provide helpful error messages:

```python
def guarded_getitem(container, key):
    """Safe item access wrapper with improved error messages."""
    try:
        return container[key]
    except KeyError:
        if isinstance(container, dict):
            available_keys = list(container.keys())
            error_msg = f"Key '{key}' not found in container. Available keys: {available_keys}"
            raise KeyError(error_msg)
        raise
```

This helps developers quickly identify what keys are available when a KeyError occurs.

## How Dependencies Are Now Accessible

After the fix, dependency results from subtask `step_1` can be accessed in THREE ways:

1. **Original key**: `context['step_1']` (with underscore) ✓ RECOMMENDED
2. **Simplified key**: `context['step1']` (without underscore) ✓ Backward compatible
3. **Via dependency_results**: `context['dependency_results']['step_1']` ✓ Explicit

This ensures compatibility regardless of how the LLM generates the code.

## Similar Issues Prevented

The following additional improvements prevent similar issues:

### Missing Dependency Detection
- Code now checks if ALL dependencies are completed before execution
- Returns clear error if any dependency is missing/failed/incomplete
- Logs available vs. missing dependencies for debugging

### Better Logging
- Added debug logging of available context keys
- Logs dependency status (completed, failed, pending)
- Tracks which dependencies are added to context

### Error Categorization
- New error type: `missing_dependencies`
- Distinguishes dependency errors from code execution errors
- Proper subtask failure marking with error metadata

## Testing Recommendations

To test the fix:

1. **Test Case 1**: Code accessing dependencies with underscores
   ```python
   result = context['step_1']['search_results']
   ```

2. **Test Case 2**: Code accessing dependencies without underscores
   ```python
   result = context['step1']['search_results']
   ```

3. **Test Case 3**: Code accessing via dependency_results
   ```python
   result = context['dependency_results']['step_1']
   ```

4. **Test Case 4**: Missing dependency handling
   - Ensure code_interpreter returns clear error when dependency is missing
   - Verify subtask is marked as failed with appropriate error type

## Files Modified

1. `/Users/jungwoonshin/github/agent-system/src/core/executor.py`
   - Lines 137-209: Enhanced dependency handling and validation
   - Line 602: Updated LLM prompt for clarity

2. `/Users/jungwoonshin/github/agent-system/src/core/tool_belt.py`
   - Lines 23-33: Improved error messages in guarded_getitem

## Impact

- **Backward Compatible**: Old code using simplified keys still works
- **Forward Compatible**: New code using original keys now works
- **Better UX**: Clear error messages when keys are missing
- **Reliability**: Prevents execution with missing dependencies

---

# Additional Fix: Typing Module Import

## Issue Description

The `typing` module was being blocked with the error:
```
Import of 'typing' is not allowed for security reasons
```

This prevented code from importing common typing utilities like `List`, `Dict`, `Optional`, `Union`, etc.

## Root Cause

The `typing` module was not in the `safe_modules` whitelist in `tool_belt.py`, even though it's a safe, standard library module that only provides type hint utilities.

## Fix Applied

Added `typing` to the safe_modules whitelist in `/Users/jungwoonshin/github/agent-system/src/core/tool_belt.py`:

```python
safe_modules = {
    'math',
    'json',
    'datetime',
    're',
    'itertools',
    'collections',
    'functools',
    'operator',
    'statistics',
    'typing',  # Safe module for type hints (List, Dict, Optional, etc.)
}
```

Also updated LLM prompts in `executor.py` to inform the LLM that typing module is available.

## What Works Now

✅ Import typing module: `from typing import List, Dict, Optional, Union, Tuple`
✅ Use typing in code logic (e.g., checking types, using TypeVar)
✅ Import any typing utilities without security errors

## Limitations

⚠️ **Type Annotations Not Supported**: Due to RestrictedPython limitations, type annotations (`:` syntax) cannot be used:
```python
# This will fail with "AnnAssign statements are not allowed"
value: Optional[int] = 10

# This works fine
value = 10  # Type can be inferred or documented in comments
```

This is a RestrictedPython limitation, not our code. The module can be imported and used, but the Python type annotation syntax is not supported by RestrictedPython's restricted execution environment.

## Files Modified

1. `/Users/jungwoonshin/github/agent-system/src/core/tool_belt.py`
   - Line 117: Added `typing` to safe_modules

2. `/Users/jungwoonshin/github/agent-system/src/core/executor.py`
   - Lines 589, 599: Updated LLM prompts to list typing in available modules

---

# Additional Fix: Attribute Access on Serialized SearchResult Objects

## Issue Description

The code was experiencing `AttributeError: 'dict' object has no attribute 'snippet'` when trying to access SearchResult attributes on serialized dictionaries:

```python
# This fails when result is a serialized dictionary:
for result in search_results:
    snippet = result.snippet  # AttributeError!
```

## Root Cause

1. SearchResult objects are serialized to dictionaries via `_serialize_result_for_code()` before being passed to code_interpreter
2. The LLM-generated code was using attribute access (`.snippet`) instead of dictionary access (`['snippet']`)
3. Some internal code also assumed SearchResult objects but could receive dictionaries

## Fixes Applied

### 1. Updated LLM Prompts (executor.py, Lines 603-607)

Added explicit warning in the system prompt:

```python
  - **SEARCH RESULTS**: When accessing search results from dependencies, note that SearchResult objects are serialized as dictionaries. Use dictionary access:
    * result['snippet'] NOT result.snippet
    * result['url'] NOT result.url
    * result['title'] NOT result.title
    * For lists of search results: for item in results_list: snippet = item['snippet']
```

### 2. Handle Both Types in Filtering (executor.py, Lines 813-822)

Added support for both SearchResult objects and dictionaries:

```python
elif isinstance(result, dict):
    # Handle already-serialized SearchResult dictionaries
    results_list.append({
        'index': i,
        'title': result.get('title', ''),
        'snippet': result.get('snippet', ''),
        'url': result.get('url', ''),
    })
```

### 3. Handle Both Types in Search Result Processor (search_result_processor.py, Lines 101-109)

Convert dictionaries back to SearchResult objects for uniform processing:

```python
if isinstance(result, dict):
    # Convert dict back to SearchResult for uniform processing
    from .models import SearchResult as SR
    result = SR(
        snippet=result.get('snippet', ''),
        url=result.get('url', ''),
        title=result.get('title', ''),
        relevance_score=result.get('relevance_score', 0.0)
    )
```

## What This Fixes

✅ **LLM-generated code** now uses dictionary access instead of attribute access  
✅ **Internal code** handles both SearchResult objects and serialized dictionaries  
✅ **Search result filtering** works with both types  
✅ **Search result processing** converts dicts back to objects for uniform handling

## Files Modified

1. `/Users/jungwoonshin/github/agent-system/src/core/executor.py`
   - Lines 603-607: Added SEARCH RESULTS warning to LLM prompt
   - Lines 813-822: Handle dictionary results in filtering

2. `/Users/jungwoonshin/github/agent-system/src/core/search_result_processor.py`
   - Lines 101-109: Convert dictionary results back to SearchResult objects

