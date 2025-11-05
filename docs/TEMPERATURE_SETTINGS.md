# LLM Temperature Settings Guide

This document describes the temperature settings used for different LLM tasks in the agent system.

## Temperature Strategy

Temperature controls the randomness/creativity of LLM outputs:
- **0.0**: Maximum determinism - same input → same output (>99% consistency)
- **0.3**: High consistency with slight natural variation
- **0.5**: Balanced between consistency and creativity
- **0.7**: Creative but focused
- **1.0+**: High creativity and randomness

## Temperature Settings by Task

### 🎯 Temperature = 0.0 (Maximum Determinism)

**Use case**: Tasks requiring exact same output every time

| File | Method/Function | Purpose |
|------|----------------|---------|
| `answer_synthesizer.py` | `synthesize()` | Final answer formatting and synthesis |
| `validation.py` | `validate_answer()` | Answer correctness validation |
| `browser_navigator.py` | `extract_with_llm()` | Precise data extraction from web pages |

**Why 0.0?**
- Answer formatting must be consistent (e.g., "word" format should always extract the same word)
- Validation should be deterministic for reproducibility
- Data extraction requires precision without variation

---

### 📊 Temperature = 0.3 (High Consistency)

**Use case**: Tasks requiring consistent but natural-sounding outputs

| File | Method/Function | Purpose |
|------|----------------|---------|
| `answer_synthesizer.py` | `build_monologue()` | Reasoning monologue generation |
| `problem_classifier.py` | `classify()` | Problem type classification |
| `query_understanding.py` | `analyze_query()` | Query requirements analysis |
| `reasoning_engine.py` | `analyze_patterns()` | Pattern recognition and analysis |
| `reasoning_engine.py` | `narrow_solution_space()` | Constraint propagation |
| `reasoning_engine.py` | `generate_hypotheses()` | Solution hypothesis generation |
| `executor.py` | `_is_result_relevant()` | Search result relevance checking |
| `search_result_processor.py` | `_check_relevance()` | Result relevance evaluation |

**Why 0.3?**
- Need consistency but some natural language variation is acceptable
- Classification should be stable but not rigid
- Reasoning should be logical and reproducible
- Relevance checking should be consistent for similar inputs

---

### ⚖️ Temperature = 0.5 (Balanced)

**Use case**: Tasks requiring flexibility in approach

| File | Method/Function | Purpose |
|------|----------------|---------|
| `planner.py` | `create_plan()` | Execution plan generation |

**Why 0.5?**
- Planning benefits from exploring different valid approaches
- Need balance between consistency and creative problem-solving
- Different plans can be equally valid for the same problem

---

### 🎨 Temperature = 0.7 (Creative but Focused)

**Use case**: Tasks requiring creativity while staying on target

| File | Method/Function | Purpose |
|------|----------------|---------|
| `executor.py` | `_determine_tool_parameters()` | Tool parameter generation |
| `executor.py` | `_identify_downloadable_resources()` | Resource identification |
| `executor.py` | `_filter_search_results_by_llm()` | Search result filtering |

**Why 0.7?**
- Parameter generation needs flexibility for different problem types
- Resource identification benefits from creative interpretation
- Search filtering needs to consider multiple perspectives
- Still focused enough to avoid irrelevant outputs

---

## Quick Reference Table

| Temperature | Task Type | Examples |
|-------------|-----------|----------|
| **0.0** | Deterministic output required | Answer synthesis, validation, data extraction |
| **0.3** | Consistent analysis | Classification, reasoning, relevance checking |
| **0.5** | Balanced creativity | Planning, strategy generation |
| **0.7** | Creative but focused | Parameter generation, resource identification |

## Best Practices

### When to Use Low Temperature (0.0 - 0.3)
- ✅ Final answer generation
- ✅ Validation and verification
- ✅ Structured data extraction
- ✅ Classification tasks
- ✅ Logical reasoning
- ✅ Format-sensitive outputs

### When to Use Medium Temperature (0.5)
- ✅ Planning and strategy
- ✅ Multi-step problem decomposition
- ✅ Exploring solution approaches

### When to Use High Temperature (0.7+)
- ✅ Creative problem interpretation
- ✅ Flexible parameter generation
- ✅ Exploratory search
- ✅ Brainstorming alternatives

### When NOT to Change Temperature
- ❌ Don't increase temperature for tasks requiring consistency
- ❌ Don't decrease temperature too much for creative tasks (can be too rigid)
- ❌ Don't use temperature > 1.0 unless you specifically need high randomness

## Testing Temperature Settings

To verify temperature effectiveness:

1. **For deterministic tasks (temp=0.0)**: Run the same input multiple times and verify outputs are identical
2. **For consistent tasks (temp=0.3)**: Run multiple times and verify outputs are similar but naturally varied
3. **For creative tasks (temp=0.7)**: Verify outputs are diverse but still relevant to the task

## Impact on System Performance

### Accuracy
- Lower temperature → More predictable, potentially more accurate for well-defined tasks
- Higher temperature → More exploration, potentially better for ambiguous problems

### Speed
- Temperature setting has **minimal impact** on LLM inference speed

### Cost
- No significant cost difference between temperature settings
- All settings require the same number of tokens

## Future Considerations

As the system evolves, consider:
- **A/B testing**: Compare different temperature settings for specific tasks
- **Dynamic temperature**: Adjust based on problem difficulty or uncertainty
- **User preferences**: Allow users to prefer more conservative (low temp) or creative (high temp) approaches





