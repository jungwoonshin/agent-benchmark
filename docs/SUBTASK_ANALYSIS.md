# Subtask Result Handling Analysis: Why Answers Are Consistently Wrong

## Executive Summary

After analyzing the subtask-related code in `src/core/`, I've identified **4 critical bugs** that cause incorrect answers:

1. **Failed subtasks are excluded from execution_results** - Missing subtasks lead to incomplete information
2. **Error strings are treated as valid results** - Error messages get synthesized into answers
3. **State inconsistency** - Failed subtasks exist in state_manager but not in execution_results
4. **No error filtering in synthesis** - Error strings are passed to LLM without filtering

## Critical Issues Identified

### Issue #1: Failed Subtasks Not Included in execution_results

**Location**: `src/core/executor.py`, lines 979-993

**Problem**:
```python
try:
    result = self.execute_subtask(subtask, problem, attachments, query_analysis)
    results[subtask.id] = result
    completed_ids.add(subtask.id)
    progress_made = True
except Exception as e:
    self.logger.error(f'Failed to execute {subtask.id}: {e}')
    # Try to continue with other tasks
    continue  # ❌ BUG: Failed subtask is never added to results!
```

**Impact**:
- When `execute_subtask()` raises an exception, it's caught but the failure is never recorded in `results[subtask.id]`
- The subtask IS marked as failed in `state_manager` (via `fail_subtask()` in `execute_subtask()` line 498)
- But it's missing from `execution_results` dict returned by `execute_plan()`
- This means:
  - Synthesis doesn't know the subtask failed
  - Synthesis uses incomplete information
  - Dependencies that need this subtask's result won't find it
  - The system appears to have succeeded when it actually failed

**Example Flow**:
```
1. execute_plan() calls execute_subtask(step_1)
2. step_1 fails with exception
3. execute_subtask() calls fail_subtask(step_1) → state_manager knows it failed
4. execute_subtask() re-raises exception
5. execute_plan() catches exception, logs error, continues
6. results[step_1] is NEVER set → execution_results missing step_1
7. step_2 depends on step_1 → tries to access step_1 result → fails or gets None
8. Synthesis receives execution_results without step_1 → incomplete information → wrong answer
```

### Issue #2: Error Strings Returned as "Results"

**Location**: `src/core/executor.py`, lines 177-188, 218-230

**Problem**:
```python
# Line 188: Returns error string instead of raising exception
if missing_dependencies:
    error_msg = f'Cannot execute code_interpreter - missing or incomplete dependencies: ...'
    self.state_manager.fail_subtask(subtask.id, error_msg)
    return f'Error: {error_msg}'  # ❌ BUG: Error string returned as "result"

# Line 218-230: Code execution errors return error strings
if result.startswith('Name Error:') or result.startswith('Execution Error:'):
    error_reason = result[:500]
    self.state_manager.fail_subtask(subtask.id, error_reason)
    # ❌ BUG: No return here - the error string result continues to be used
```

**Impact**:
- Error strings like `"Error: Cannot execute code_interpreter - missing dependencies: step_1, step_2"` become the "result"
- These error strings are stored in `results[subtask.id]` and passed to synthesis
- The synthesizer treats these as valid data and tries to extract answers from them
- The LLM may misinterpret error messages as actual information
- Even though the subtask is marked as failed in state_manager, the error string is treated as a valid result

**Example Flow**:
```
1. step_1 executes code_interpreter
2. Missing dependencies detected → returns "Error: Cannot execute..."
3. fail_subtask() called → state_manager marks as failed
4. But "Error: Cannot execute..." is returned and stored in results[step_1]
5. Synthesis receives execution_results = {"step_1": "Error: Cannot execute..."}
6. LLM tries to extract answer from error message → wrong answer
```

### Issue #3: State Inconsistency Between execution_results and state_manager

**Location**: `src/core/executor.py` and `src/core/agent.py`

**Problem**:
- Failed subtasks are tracked in `state_manager.subtasks[subtask_id].status = 'failed'`
- But they're missing from `execution_results` dict (Issue #1)
- When synthesis happens, it only sees `execution_results`, not `state_manager.subtasks`
- This creates two sources of truth that are out of sync

**Impact**:
- Validation (`validation.py`) reads from `execution_results` (line 63) but could also check `state_manager.subtasks`
- The system doesn't know which source to trust
- Retry logic may not work correctly because it checks `state_manager.get_failed_subtasks()` but synthesis uses `execution_results`

### Issue #4: No Error Filtering in Synthesis

**Location**: `src/core/answer_synthesizer.py`, lines 487-496

**Problem**:
```python
execution_summary_str = '## Execution Results Summary\n'
for subtask_id, result in execution_summary.items():
    if isinstance(result, dict):
        result_str = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    else:
        result_str = str(result)
    execution_summary_str += f'\n### Subtask {subtask_id}:\n{result_str}\n'
    # ❌ BUG: No filtering of error strings or failed subtasks
```

**Impact**:
- Error strings from Issue #2 are included in the synthesis prompt
- The LLM sees error messages like "Error: Cannot execute..." as if they were valid data
- No indication that these are errors, not actual results
- The synthesizer tries to extract answers from error messages

## Root Cause Analysis

The fundamental issue is **inconsistent error handling**:

1. **Some errors raise exceptions** (caught in `execute_plan()`, missing from results)
2. **Some errors return error strings** (included in results, treated as valid data)
3. **No unified error result format** (should be a structured error dict)
4. **Synthesis doesn't distinguish errors from results** (no filtering)

## Recommended Fixes

### Fix #1: Include Failed Subtasks in execution_results

**File**: `src/core/executor.py`, `execute_plan()` method

**Change**:
```python
try:
    result = self.execute_subtask(subtask, problem, attachments, query_analysis)
    results[subtask.id] = result
    completed_ids.add(subtask.id)
    progress_made = True
except Exception as e:
    self.logger.error(f'Failed to execute {subtask.id}: {e}')
    # ✅ FIX: Store failure in results with structured error format
    results[subtask.id] = {
        'error': str(e),
        'error_type': type(e).__name__,
        'status': 'failed'
    }
    # Note: fail_subtask() already called in execute_subtask(), so state is consistent
    continue
```

### Fix #2: Standardize Error Results

**File**: `src/core/executor.py`, `execute_subtask()` method

**Change**: Instead of returning error strings, return structured error dicts:

```python
# Line 188: Instead of returning error string
if missing_dependencies:
    error_msg = f'Cannot execute code_interpreter - missing or incomplete dependencies: ...'
    self.state_manager.fail_subtask(subtask.id, error_msg)
    # ✅ FIX: Return structured error dict
    return {
        'error': error_msg,
        'error_type': 'missing_dependencies',
        'status': 'failed',
        'subtask_id': subtask.id
    }

# Line 218-230: For code execution errors
if result.startswith('Name Error:') or result.startswith('Execution Error:'):
    error_reason = result[:500]
    self.state_manager.fail_subtask(subtask.id, error_reason)
    # ✅ FIX: Return structured error dict
    return {
        'error': error_reason,
        'error_type': 'execution_error',
        'status': 'failed',
        'subtask_id': subtask.id
    }
```

### Fix #3: Filter Errors in Synthesis

**File**: `src/core/answer_synthesizer.py`, `synthesize()` method

**Change**: Filter out error results before passing to LLM:

```python
execution_summary_str = '## Execution Results Summary\n'
for subtask_id, result in execution_summary.items():
    # ✅ FIX: Skip error results
    if isinstance(result, dict) and result.get('status') == 'failed':
        self.logger.warning(f'Skipping failed subtask {subtask_id} from synthesis')
        continue
    
    # ✅ FIX: Check if result is an error string
    if isinstance(result, str) and result.startswith('Error:'):
        self.logger.warning(f'Skipping error result from subtask {subtask_id}: {result[:100]}')
        continue
    
    if isinstance(result, dict):
        result_str = json.dumps(result, indent=2, ensure_ascii=False, default=str)
    else:
        result_str = str(result)
    execution_summary_str += f'\n### Subtask {subtask_id}:\n{result_str}\n'
```

### Fix #4: Ensure Failed Subtasks Are Included from state_manager

**File**: `src/core/agent.py`, `solve()` method

**Change**: Merge failed subtasks from state_manager into execution_results:

```python
# After execution_results = self.executor.execute_plan(...)
# ✅ FIX: Include failed subtasks from state_manager
for subtask_id, subtask in self.state_manager.subtasks.items():
    if subtask.status == 'failed' and subtask_id not in execution_results:
        # Include failed subtask in execution_results for completeness
        execution_results[subtask_id] = {
            'error': subtask.metadata.get('error', 'Unknown error'),
            'error_type': subtask.metadata.get('error_type', 'unknown'),
            'status': 'failed',
            'subtask_id': subtask_id
        }
```

## Testing Recommendations

1. **Test failed subtask handling**: Create a plan where one subtask fails, verify it's included in execution_results
2. **Test error string filtering**: Verify error strings are not passed to synthesis
3. **Test dependency chain with failures**: Verify dependent subtasks handle missing/failed dependencies correctly
4. **Test synthesis with mixed results**: Some successful, some failed subtasks

## Impact Assessment

**Severity**: **CRITICAL** - These bugs directly cause incorrect answers by:
- Excluding critical failed subtasks from synthesis
- Treating error messages as valid data
- Creating incomplete execution summaries
- Confusing the LLM with error messages

**Affected Components**:
- `executor.py` - Subtask execution and result collection
- `answer_synthesizer.py` - Answer synthesis from results
- `agent.py` - Orchestration and result passing

**Estimated Fix Complexity**: Medium (4 focused changes across 3 files)





