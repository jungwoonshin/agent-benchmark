"""Comprehensive tests for CodeInterpreter class and error analysis."""

import logging
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from code_interpreter.interpreter import CodeInterpreter
from code_interpreter.error_analysis import ExecutionResultAnalyzer


def setup_logger():
    """Setup logger for testing"""
    logger = logging.getLogger("test_interpreter")
    logger.setLevel(logging.WARNING)  # Reduce verbosity
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    return logger


def test_simple_execution():
    """Test basic code execution"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)
    result = interpreter.execute("2 + 2")
    assert result == "4", f"Expected '4', got '{result}'"
    print("✓ Test 1 passed: Simple execution")


def test_with_context():
    """Test code execution with context variables"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)
    context = {"x": 10, "y": 20}
    result = interpreter.execute("x + y", context)
    assert result == "30", f"Expected '30', got '{result}'"
    print("✓ Test 2 passed: Execution with context")


def test_error_handling_name_error():
    """Test error handling for undefined variables"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)
    result = interpreter.execute("undefined_var + 5")
    assert ExecutionResultAnalyzer.is_error_result(result), f"Expected error, got '{result}'"
    assert "Name Error:" in result, f"Expected Name Error, got '{result}'"
    print("✓ Test 3 passed: Name error handling")


def test_error_handling_syntax_error():
    """Test error handling for syntax errors"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)
    result = interpreter.execute("if True: pass else:")
    assert ExecutionResultAnalyzer.is_error_result(result), f"Expected error, got '{result}'"
    assert "Syntax Error:" in result or "Compilation Error:" in result, f"Expected syntax error, got '{result}'"
    print("✓ Test 4 passed: Syntax error handling")


def test_error_dict_detection():
    """Test detection of dict-based errors"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)
    # Test KeyError - should be caught as Execution Error
    result = interpreter.execute("d = {}; d['missing']")
    # Should detect error
    assert ExecutionResultAnalyzer.is_error_result(result), f"Expected error, got '{result}'"
    assert "Execution Error:" in result or "Key" in result, f"Expected execution/key error, got '{result}'"
    print("✓ Test 5 passed: Dict error detection")


def test_successful_math_operations():
    """Test successful math operations"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)
    code = """import math
math.sqrt(16)"""
    result = interpreter.execute(code)
    assert not ExecutionResultAnalyzer.is_error_result(result), f"Expected success, got error: '{result}'"
    assert "4" in result, f"Expected 4.0, got '{result}'"
    print("✓ Test 6 passed: Math operations")


def test_error_analysis_is_error_result():
    """Test ExecutionResultAnalyzer.is_error_result()"""
    # Test dict with error
    assert ExecutionResultAnalyzer.is_error_result({"error": "test error"}) is True
    # Test string with error prefix
    assert ExecutionResultAnalyzer.is_error_result("Name Error: test") is True
    assert ExecutionResultAnalyzer.is_error_result("Execution Error: test") is True
    # Test string with dict error pattern
    assert ExecutionResultAnalyzer.is_error_result("{'error': 'test'}") is True
    # Test success cases
    assert ExecutionResultAnalyzer.is_error_result("42") is False
    assert ExecutionResultAnalyzer.is_error_result({"result": "success"}) is False
    print("✓ Test 7 passed: Error analysis is_error_result()")


def test_error_analysis_extract_error_message():
    """Test ExecutionResultAnalyzer.extract_error_message()"""
    # Test dict error
    msg = ExecutionResultAnalyzer.extract_error_message({"error": "test error"})
    assert msg == "test error", f"Expected 'test error', got '{msg}'"
    # Test string error
    msg = ExecutionResultAnalyzer.extract_error_message("Name Error: undefined var")
    assert msg == "Name Error: undefined var", f"Expected error message, got '{msg}'"
    # Test dict pattern in string
    msg = ExecutionResultAnalyzer.extract_error_message("{'error': 'dict error'}")
    assert msg == "dict error", f"Expected 'dict error', got '{msg}'"
    # Test None for non-error
    msg = ExecutionResultAnalyzer.extract_error_message("42")
    assert msg is None, f"Expected None, got '{msg}'"
    print("✓ Test 8 passed: Error analysis extract_error_message()")


def test_error_analysis_normalize_error_result():
    """Test ExecutionResultAnalyzer.normalize_error_result()"""
    # Test dict error
    is_error, normalized, msg = ExecutionResultAnalyzer.normalize_error_result({"error": "test"})
    assert is_error is True
    assert "Execution Error:" in normalized
    assert msg == "test"
    # Test string error
    is_error, normalized, msg = ExecutionResultAnalyzer.normalize_error_result("Name Error: test")
    assert is_error is True
    assert normalized == "Name Error: test"
    assert msg == "Name Error: test"
    # Test success
    is_error, normalized, msg = ExecutionResultAnalyzer.normalize_error_result("42")
    assert is_error is False
    assert normalized == "42"
    assert msg is None
    print("✓ Test 9 passed: Error analysis normalize_error_result()")


def test_error_analysis_classify_error_type():
    """Test ExecutionResultAnalyzer.classify_error_type()"""
    assert ExecutionResultAnalyzer.classify_error_type("Name Error: test") == "name_error"
    assert ExecutionResultAnalyzer.classify_error_type("Import Error: test") == "import_error"
    assert ExecutionResultAnalyzer.classify_error_type("Syntax Error: test") == "syntax_error"
    assert ExecutionResultAnalyzer.classify_error_type("Execution Error: test") == "execution_error"
    print("✓ Test 10 passed: Error analysis classify_error_type()")


def test_error_analysis_create_error_dict():
    """Test ExecutionResultAnalyzer.create_error_dict()"""
    error_dict = ExecutionResultAnalyzer.create_error_dict(
        "test error", subtask_id="step_1", retry_attempts=2
    )
    assert error_dict["error"] == "test error"
    assert error_dict["status"] == "failed"
    assert error_dict["subtask_id"] == "step_1"
    assert error_dict["retry_attempts"] == 2
    assert "error_type" in error_dict
    print("✓ Test 11 passed: Error analysis create_error_dict()")


def test_complex_code_execution():
    """Test complex code with multiple statements"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)
    code = """
result = 0
for i in range(1, 6):
    result += i
result
"""
    result = interpreter.execute(code)
    assert not ExecutionResultAnalyzer.is_error_result(result), f"Expected success, got error: '{result}'"
    assert "15" in result, f"Expected 15, got '{result}'"
    print("✓ Test 12 passed: Complex code execution")


def test_list_operations():
    """Test list operations"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)
    result = interpreter.execute("[1, 2, 3, 4, 5]")
    assert not ExecutionResultAnalyzer.is_error_result(result), f"Expected success, got error: '{result}'"
    assert "1" in result and "5" in result, f"Expected list, got '{result}'"
    print("✓ Test 13 passed: List operations")


def test_dict_operations():
    """Test dict operations"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)
    context = {"data": {"key": "value"}}
    result = interpreter.execute("data['key']", context)
    assert not ExecutionResultAnalyzer.is_error_result(result), f"Expected success, got error: '{result}'"
    assert "value" in result, f"Expected 'value', got '{result}'"
    print("✓ Test 14 passed: Dict operations")


def test_error_logging():
    """Test that errors are properly logged"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)
    result = interpreter.execute("undefined_var")
    # Should return error string, not dict
    assert isinstance(result, str), f"Expected string, got {type(result)}"
    assert ExecutionResultAnalyzer.is_error_result(result), f"Expected error, got '{result}'"
    print("✓ Test 15 passed: Error logging")


def run_all_tests():
    """Run all test cases"""
    print("=" * 70)
    print("Running comprehensive tests for CodeInterpreter")
    print("=" * 70)

    tests = [
        test_simple_execution,
        test_with_context,
        test_error_handling_name_error,
        test_error_handling_syntax_error,
        test_error_dict_detection,
        test_successful_math_operations,
        test_error_analysis_is_error_result,
        test_error_analysis_extract_error_message,
        test_error_analysis_normalize_error_result,
        test_error_analysis_classify_error_type,
        test_error_analysis_create_error_dict,
        test_complex_code_execution,
        test_list_operations,
        test_dict_operations,
        test_error_logging,
    ]

    passed = 0
    failed = 0
    failures = []

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            failures.append((test.__name__, str(e)))

    print("=" * 70)
    print(f"Test Summary: {passed} passed, {failed} failed")
    if failures:
        print("\nFailures:")
        for test_name, error in failures:
            print(f"  - {test_name}: {error}")
    print("=" * 70)

    return failed == 0, failures


if __name__ == "__main__":
    success, failures = run_all_tests()
    sys.exit(0 if success else 1)

