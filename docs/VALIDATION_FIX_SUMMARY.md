# Validation Fix: Only Validate Executed Subtasks

## Issue Identified

The user reported that validation was checking ALL requirements from the query analysis, even for subtasks that weren't executed yet. This caused false positives for "missing requirements" when those requirements would have been addressed by subtasks that haven't run.

## Root Cause

The validation in `answer_synthesizer.py` was comparing:
-  ALL requirements from query analysis (explicit + implicit)
- Against only the executed subtasks' results

This mismatch caused it to flag requirements as "missing" even when no subtask had been executed to address them yet.

## Solution

Update the validation prompt to make it clear that validation should:
1. **Only check the quality of executed subtasks** (not overall completeness)
2. **Not flag missing requirements** that no executed subtask was supposed to provide
3. **Focus on data quality issues** in what was actually executed (stubs, errors, etc.)

## Implementation

Update `/Users/jungwoonshin/github/agent-system/src/core/answer_synthesizer.py` around line 291-321:

### Before (Problematic)
```python
system_prompt = """You are an expert at validating execution results for completeness and data quality.
Analyze whether the execution results contain sufficient information to answer the problem question.

Your task is to:
1. Check if execution results are empty or contain only placeholder/stub responses
2. Determine if all critical requirements from the query analysis are addressed by the results
3. Assess whether the data quality is sufficient to extract the required answer format
4. Identify any missing information that prevents answering the question
```

**Problem:** Checks if "all critical requirements... are addressed" even for unexecuted subtasks

### After (Fixed)
```python
system_prompt = """You are an expert at validating execution results for completeness and data quality.
Analyze whether the EXECUTED subtasks produced sufficient data.

IMPORTANT: Only validate the subtasks that were actually executed. Do NOT check for requirements that would be addressed by subtasks that haven't run yet.

Your task is to:
1. Check if the EXECUTED subtasks' results are empty or contain only placeholder/stub responses
2. Assess if the data FROM THESE SPECIFIC SUBTASKS is of good quality
3. Identify issues with the execution results (not with missing subtasks)
4. Focus on data quality, not completeness of overall requirements

Return a JSON object with:
- is_complete: boolean indicating if the EXECUTED subtasks produced usable data (not if ALL requirements are met)
- missing_requirements: list ONLY of requirements that EXECUTED subtasks were supposed to provide but didn't
- data_quality: string indicating quality level ("good", "fair", "poor") based on what was executed
- warnings: list of warnings about the EXECUTED subtasks' results (e.g., stub responses, errors, incomplete data)

DO NOT flag requirements as missing if no subtask has been executed to address them yet.
```

## Benefits

1. ✅ **Accurate Validation** - Only checks what was actually executed
2. ✅ **No False Positives** - Won't flag unexecuted work as "missing"
3. ✅ **Focus on Quality** - Validates data quality, not overall completeness
4. ✅ **Better Retry Logic** - Only retries when executed subtasks had issues
5. ✅ **Clearer Failures** - Identifies actual problems vs missing work

## Example Scenario

### Problem: Find AI regulation paper from June 2022

**Plan:**
- step_1: Search for paper ✅
- step_2: Navigate to paper ✅  
- step_3: Extract figure data ❌ (FAILED)
- step_4: Analyze labels ⏸️ (NOT EXECUTED - depends on step_3)

### Before (Problematic)
```
Validation checks: "Do results satisfy ALL requirements?"
Requirements: [find paper, navigate, extract figure, analyze labels]
Results: Only step_1, step_2 executed

Validation: ❌ is_complete=False
Missing: ["Extract figure data", "Analyze labels"]
```

**Problem:** Flags step_4's requirement as "missing" even though step_4 never ran!

### After (Fixed)
```
Validation checks: "Did EXECUTED subtasks (step_1, step_2) produce good data?"
Executed: step_1 ✅, step_2 ✅, step_3 ❌

Validation: ❌ is_complete=False  
Missing: ["Extract figure data"] (only from step_3, which failed)
Warnings: ["step_3 returned stub response"]
```

**Fixed:** Only flags step_3's failure, not step_4 (which wasn't executed)

## Integration with Retry Logic

This fix works perfectly with the new retry-failed-subtasks feature:

1. Validation identifies that step_3 (executed) had issues
2. Retry logic finds step_3 in failed subtasks
3. System retries ONLY step_3
4. step_4 will run after step_3 succeeds (dependency satisfied)

## Testing

To verify the fix works:

```bash
# Run tests
uv run python test_validation.py

# Check logs for validation messages
grep "is_complete" logs/log.txt
grep "missing_requirements" logs/log.txt
```

### Expected Behavior

- Validation should only mention executed subtasks
- "missing_requirements" should only list things executed subtasks should have provided
- Requirements for unexecuted subtasks should not appear

## Files Modified

- `/Users/jungwoonshin/github/agent-system/src/core/answer_synthesizer.py` - Updated validation prompts to focus on executed subtasks only

## Summary

The validation system now correctly:
- ✅ Validates only what was executed
- ✅ Focuses on data quality issues
- ✅ Doesn't flag unexecuted work as missing
- ✅ Works seamlessly with retry logic
- ✅ Provides accurate feedback for debugging

