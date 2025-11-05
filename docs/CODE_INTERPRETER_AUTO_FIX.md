# Code Interpreter Auto-Fix Feature

## Overview

The executor now automatically attempts to fix code errors when `code_interpreter` fails, using LLM to analyze the error and generate corrected code. This improves success rates by automatically recovering from common coding mistakes.

## How It Works

### Automatic Error Detection and Retry

When `code_interpreter` encounters an error, the system:

1. **Detects the error** - Checks for:
   - `Name Error:` - Undefined variables
   - `Execution Error:` - Runtime errors
   - `Import Error:` - Unallowed imports
   - `Syntax Error:` - Syntax mistakes

2. **Attempts to fix** - Uses LLM to:
   - Analyze the error message
   - Understand the original code intent
   - Generate corrected code that fixes the specific error

3. **Retries execution** - Automatically retries with the fixed code

4. **Limits retries** - Maximum 2 retry attempts to avoid infinite loops

### Implementation Details

**Location**: `src/core/executor.py`

**Key Method**: `_fix_code_error()` (lines 653-750)

**Trigger**: When `code_interpreter` returns an error string (lines 223-311)

## Features

### Error Types Handled

1. **NameError** - Missing variables
   - Fix: Checks if variable exists in context
   - Suggests using `context['key']` syntax for context variables

2. **ImportError** - Unallowed imports
   - Fix: Removes the import or suggests whitelisted alternatives

3. **SyntaxError** - Syntax mistakes
   - Fix: Corrects syntax errors

4. **ExecutionError** - Runtime errors
   - Fix: Fixes logic errors based on error message

### Retry Logic

```python
max_retries = 2
retry_count = subtask.metadata.get('code_fix_retry_count', 0)

if retry_count < max_retries:
    fixed_code = self._fix_code_error(code, error_reason, context, problem, subtask)
    if fixed_code:
        result = self.tool_belt.code_interpreter(fixed_code, context)
        # Check if retry succeeded
```

### Success Handling

- If retry succeeds: Clears error metadata, continues with successful result
- If retry fails: Increments retry count, tries again (up to max)
- If all retries fail: Marks subtask as failed with structured error dict

## Example Flow

### Before (Old Behavior)
```
1. Code executes: result = step_1 * 0.04
2. Error: Name Error: name 'step_1' is not defined
3. Subtask marked as failed
4. Error string returned as result
```

### After (New Behavior)
```
1. Code executes: result = step_1 * 0.04
2. Error: Name Error: name 'step_1' is not defined. Available context: ['step_1']
3. LLM analyzes error: "step_1 is in context, should use context['step_1']"
4. Fixed code: result = context['step_1'] * 0.04
5. Retry execution with fixed code
6. Success! Result: 40.08
7. Subtask completes successfully
```

## Benefits

1. **Higher Success Rate** - Automatically recovers from common mistakes
2. **Better User Experience** - Fewer failed subtasks due to simple errors
3. **Self-Healing** - System fixes its own code generation mistakes
4. **Transparent** - All retry attempts are logged for debugging

## Configuration

- **Max Retries**: 2 (hardcoded, can be adjusted)
- **Retry Tracking**: Stored in `subtask.metadata['code_fix_retry_count']`
- **Original Code**: Stored in `subtask.metadata['original_code']`
- **Last Error**: Stored in `subtask.metadata['last_error']`

## Logging

All retry attempts are logged:
- `INFO`: "Attempting to fix code error (retry X/2)..."
- `INFO`: "Retrying code execution with fixed code (attempt X)"
- `INFO`: "Code fix retry X succeeded!" (if successful)
- `WARNING`: "Code fix retry X also failed" (if retry also fails)

## Error Handling

If code fixing fails:
- LLM call fails → Returns `None`, skips retry
- Fixed code is empty → Logs warning, skips retry
- Fixed code same as original → Skips retry (no change)
- All retries exhausted → Marks subtask as failed with structured error

## Limitations

1. **Only fixes code_interpreter errors** - Other tool errors not handled
2. **Max 2 retries** - Prevents infinite loops
3. **LLM-dependent** - Requires LLM to successfully fix the code
4. **Error message quality** - Depends on clear error messages from code_interpreter

## Future Enhancements

Potential improvements:
- Increase max retries for specific error types
- Cache common fixes for similar errors
- Learn from successful fixes to improve future attempts
- Support for more error types (TypeError, ValueError, etc.)





