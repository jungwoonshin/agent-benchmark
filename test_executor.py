"""Comprehensive tests for Executor class and code execution with retry logic."""

import logging
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

# Mock webdriver_manager before any imports
sys.modules['webdriver_manager'] = MagicMock()
sys.modules['webdriver_manager.chrome'] = MagicMock()

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path.parent))  # Add parent to allow relative imports

from src.execution import Executor
from src.state import InformationStateManager, Subtask
from src.code_interpreter.error_analysis import ExecutionResultAnalyzer


def setup_logger():
    """Setup logger for testing"""
    logger = logging.getLogger("test_executor")
    logger.setLevel(logging.WARNING)  # Reduce verbosity
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    return logger


def create_mock_tool_belt():
    """Create a mock ToolBelt"""
    tool_belt = Mock()
    return tool_belt


def create_mock_llm_service():
    """Create a mock LLMService"""
    llm_service = Mock()
    return llm_service


def create_executor():
    """Create an Executor instance with mocked dependencies"""
    logger = setup_logger()
    tool_belt = create_mock_tool_belt()
    llm_service = create_mock_llm_service()
    state_manager = InformationStateManager(logger)

    executor = Executor(tool_belt, llm_service, state_manager, logger)
    return executor


def test_execute_code_with_retry_success():
    """Test successful code execution without retries"""
    executor = create_executor()
    tool_belt = executor.tool_belt

    # Mock successful execution
    tool_belt.code_interpreter.return_value = "42"

    subtask = Subtask(
        id="step_1",
        description="Test task",
        status="in_progress",
        metadata={},
    )
    executor.state_manager.subtasks[subtask.id] = subtask

    result = executor._execute_code_with_retry(
        code="2 * 21",
        context={},
        problem="Test problem",
        subtask=subtask,
        max_retries=10,
    )

    assert result == "42", f"Expected '42', got '{result}'"
    assert not ExecutionResultAnalyzer.is_error_result(result)
    assert tool_belt.code_interpreter.call_count == 1
    print("✓ Test 1 passed: Code execution success")


def test_execute_code_with_retry_error_then_success():
    """Test code execution that fails once then succeeds"""
    executor = create_executor()
    tool_belt = executor.tool_belt

    # Mock error then success
    tool_belt.code_interpreter.side_effect = [
        "Name Error: name 'x' is not defined",
        "42",  # Success on retry
    ]

    # Mock code fixing
    executor._fix_code_error = Mock(return_value="2 * 21")

    subtask = Subtask(
        id="step_2",
        description="Test task",
        status="in_progress",
        metadata={},
    )
    executor.state_manager.subtasks[subtask.id] = subtask

    result = executor._execute_code_with_retry(
        code="x * 21",
        context={},
        problem="Test problem",
        subtask=subtask,
        max_retries=10,
    )

    assert result == "42", f"Expected '42', got '{result}'"
    assert not ExecutionResultAnalyzer.is_error_result(result)
    assert tool_belt.code_interpreter.call_count == 2
    assert executor._fix_code_error.call_count == 1
    print("✓ Test 2 passed: Code execution with retry success")


def test_execute_code_with_retry_failure():
    """Test code execution that fails after all retries"""
    executor = create_executor()
    tool_belt = executor.tool_belt

    # Mock consistent error
    tool_belt.code_interpreter.return_value = "Name Error: name 'x' is not defined"

    # Mock code fixing that returns same code (can't fix)
    executor._fix_code_error = Mock(return_value="x * 21")

    subtask = Subtask(
        id="step_3",
        description="Test task",
        status="in_progress",
        metadata={},
    )
    executor.state_manager.subtasks[subtask.id] = subtask

    result = executor._execute_code_with_retry(
        code="x * 21",
        context={},
        problem="Test problem",
        subtask=subtask,
        max_retries=2,
    )

    # Should return error dict
    assert ExecutionResultAnalyzer.is_error_result(result)
    assert isinstance(result, dict)
    assert result["status"] == "failed"
    assert "error" in result
    assert "error_type" in result
    assert tool_belt.code_interpreter.call_count >= 1
    print("✓ Test 3 passed: Code execution failure after retries")


def test_handle_code_execution_success():
    """Test handling successful code execution"""
    executor = create_executor()
    subtask = Subtask(
        id="step_4",
        description="Test task",
        status="in_progress",
        metadata={"error": "previous error", "error_type": "name_error"},
    )
    executor.state_manager.subtasks[subtask.id] = subtask

    result = executor._handle_code_execution_success("42", retry_count=2, subtask=subtask)

    assert result == "42"
    # Metadata should be cleared
    assert "error" not in subtask.metadata
    assert "error_type" not in subtask.metadata
    assert "code_fix_retry_count" not in subtask.metadata
    print("✓ Test 4 passed: Handle code execution success")


def test_handle_code_execution_failure():
    """Test handling code execution failure"""
    executor = create_executor()
    subtask = Subtask(
        id="step_5",
        description="Test task",
        status="in_progress",
        metadata={},
    )
    executor.state_manager.subtasks[subtask.id] = subtask

    result = executor._handle_code_execution_failure(
        result="Name Error: undefined",
        last_error="Name Error: undefined",
        retry_count=3,
        subtask=subtask,
    )

    assert isinstance(result, dict)
    assert result["status"] == "failed"
    assert result["error_type"] == "name_error"
    assert result["subtask_id"] == "step_5"
    assert result["retry_attempts"] == 3
    assert subtask.metadata["error"] == "Name Error: undefined"
    assert subtask.metadata["error_type"] == "name_error"
    assert subtask.status == "failed" or "step_5" in executor.state_manager.dead_ends
    print("✓ Test 5 passed: Handle code execution failure")


def test_update_code_retry_metadata():
    """Test updating retry metadata"""
    executor = create_executor()
    subtask = Subtask(
        id="step_6",
        description="Test task",
        status="in_progress",
        metadata={},
    )

    executor._update_code_retry_metadata(
        subtask=subtask,
        original_code="original",
        retry_count=1,
        error_reason="error",
        fixed_code="fixed",
    )

    assert subtask.metadata["code_fix_retry_count"] == 1
    assert subtask.metadata["original_code"] == "original"
    assert subtask.metadata["last_error"] == "error"
    assert subtask.metadata["last_fixed_code"] == "fixed"
    print("✓ Test 6 passed: Update code retry metadata")


def test_clear_code_error_metadata():
    """Test clearing error metadata"""
    executor = create_executor()
    subtask = Subtask(
        id="step_7",
        description="Test task",
        status="in_progress",
        metadata={
            "error": "test error",
            "error_type": "name_error",
            "code_fix_retry_count": 2,
        },
    )

    executor._clear_code_error_metadata(subtask)

    assert "error" not in subtask.metadata
    assert "error_type" not in subtask.metadata
    assert "code_fix_retry_count" not in subtask.metadata
    print("✓ Test 7 passed: Clear code error metadata")


def test_error_dict_format_in_retry():
    """Test handling dict-based errors in retry loop"""
    executor = create_executor()
    tool_belt = executor.tool_belt

    # Mock dict-based error
    tool_belt.code_interpreter.return_value = {"error": "name 'dict' is not defined"}

    executor._fix_code_error = Mock(return_value=None)

    subtask = Subtask(
        id="step_8",
        description="Test task",
        status="in_progress",
        metadata={},
    )
    executor.state_manager.subtasks[subtask.id] = subtask

    result = executor._execute_code_with_retry(
        code="dict()",
        context={},
        problem="Test problem",
        subtask=subtask,
        max_retries=2,
    )

    # Should detect and handle dict error
    assert ExecutionResultAnalyzer.is_error_result(result)
    assert isinstance(result, dict)
    assert result["status"] == "failed"
    print("✓ Test 8 passed: Error dict format in retry")


def test_string_error_pattern_in_retry():
    """Test handling string error patterns in retry loop"""
    executor = create_executor()
    tool_belt = executor.tool_belt

    # Mock string error pattern
    tool_belt.code_interpreter.return_value = "{'error': 'test error'}"

    executor._fix_code_error = Mock(return_value=None)

    subtask = Subtask(
        id="step_9",
        description="Test task",
        status="in_progress",
        metadata={},
    )
    executor.state_manager.subtasks[subtask.id] = subtask

    result = executor._execute_code_with_retry(
        code="test()",
        context={},
        problem="Test problem",
        subtask=subtask,
        max_retries=2,
    )

    # Should detect string error pattern
    assert ExecutionResultAnalyzer.is_error_result(result)
    print("✓ Test 9 passed: String error pattern in retry")


def test_max_retries_limit():
    """Test that max retries limit is respected"""
    executor = create_executor()
    tool_belt = executor.tool_belt

    # Mock consistent error
    tool_belt.code_interpreter.return_value = "Name Error: test"

    executor._fix_code_error = Mock(return_value="fixed_code")

    subtask = Subtask(
        id="step_10",
        description="Test task",
        status="in_progress",
        metadata={},
    )
    executor.state_manager.subtasks[subtask.id] = subtask

    result = executor._execute_code_with_retry(
        code="test",
        context={},
        problem="Test problem",
        subtask=subtask,
        max_retries=3,
    )

    # Should respect max retries
    assert tool_belt.code_interpreter.call_count <= 4  # Initial + 3 retries
    assert ExecutionResultAnalyzer.is_error_result(result)
    print("✓ Test 10 passed: Max retries limit")


def test_error_classification():
    """Test error type classification in failure handling"""
    executor = create_executor()

    test_cases = [
        ("Name Error: test", "name_error"),
        ("Import Error: test", "import_error"),
        ("Syntax Error: test", "syntax_error"),
        ("Execution Error: test", "execution_error"),
    ]

    for error_message, expected_type in test_cases:
        subtask = Subtask(
            id=f"step_{expected_type}",
            description="Test task",
            status="in_progress",
            metadata={},
        )
        executor.state_manager.subtasks[subtask.id] = subtask

        result = executor._handle_code_execution_failure(
            result=error_message,
            last_error=error_message,
            retry_count=0,
            subtask=subtask,
        )

        assert result["error_type"] == expected_type, f"Expected {expected_type}, got {result['error_type']}"

    print("✓ Test 11 passed: Error classification")


def run_all_tests():
    """Run all test cases"""
    print("=" * 70)
    print("Running comprehensive tests for Executor")
    print("=" * 70)

    tests = [
        test_execute_code_with_retry_success,
        test_execute_code_with_retry_error_then_success,
        test_execute_code_with_retry_failure,
        test_handle_code_execution_success,
        test_handle_code_execution_failure,
        test_update_code_retry_metadata,
        test_clear_code_error_metadata,
        test_error_dict_format_in_retry,
        test_string_error_pattern_in_retry,
        test_max_retries_limit,
        test_error_classification,
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

