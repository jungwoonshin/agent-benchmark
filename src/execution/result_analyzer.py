"""Execution result analyzer for backward compatibility."""

# Import the actual ExecutionResultAnalyzer from code_interpreter
# This is used for code execution error handling (legacy support)
from ..code_interpreter.error_analysis import ExecutionResultAnalyzer

__all__ = ['ExecutionResultAnalyzer']
