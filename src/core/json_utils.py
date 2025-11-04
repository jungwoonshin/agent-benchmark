"""JSON utilities for extracting JSON from LLM responses."""

import json
import re
from typing import Any, Dict


def extract_json_from_text(text: str) -> str:
    """
    Extract JSON from text that might contain markdown code blocks or extra text.

    Args:
        text: Text that may contain JSON.

    Returns:
        Extracted JSON string.
    """
    # Try to find JSON in markdown code blocks first
    json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if json_match:
        return json_match.group(1)

    # Try to find JSON object directly - use balanced brace matching for nested JSON
    # Find the first { and then find the matching }
    brace_count = 0
    start_idx = -1
    for i, char in enumerate(text):
        if char == '{':
            if start_idx == -1:
                start_idx = i
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0 and start_idx != -1:
                return text[start_idx : i + 1]

    # Fallback: try simple regex (original behavior)
    json_match = re.search(r'\{.*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)

    # Return original text if no JSON found
    return text


def safe_json_loads(text: str, default: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Safely parse JSON from text, with fallback to default.

    Args:
        text: Text that may contain JSON.
        default: Default value to return if parsing fails.

    Returns:
        Parsed JSON dictionary or default.
    """
    try:
        json_text = extract_json_from_text(text)
        return json.loads(json_text)
    except (json.JSONDecodeError, ValueError):
        return default or {}
