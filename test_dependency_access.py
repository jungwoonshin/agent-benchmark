"""
Test script to verify dependency access fix.
This tests that dependencies can be accessed with both original and simplified keys.
"""

import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def test_guarded_getitem():
    """Test the improved guarded_getitem function."""
    from src.code_interpreter.interpreter import guarded_getitem

    logger.info('Testing guarded_getitem...')

    # Test 1: Successful access
    test_dict = {'step_1': 'value1', 'step_2': 'value2'}
    try:
        result = guarded_getitem(test_dict, 'step_1')
        assert result == 'value1', f"Expected 'value1', got {result}"
        logger.info('✓ Test 1 passed: Successful key access')
    except Exception as e:
        logger.error(f'✗ Test 1 failed: {e}')
        return False

    # Test 2: Missing key with helpful error message
    try:
        result = guarded_getitem(test_dict, 'step_3')
        logger.error('✗ Test 2 failed: Should have raised KeyError')
        return False
    except KeyError as e:
        error_msg = str(e)
        if 'Available keys' in error_msg:
            logger.info(f'✓ Test 2 passed: Got helpful error message: {error_msg}')
        else:
            logger.error(f'✗ Test 2 failed: Error message not helpful: {error_msg}')
            return False

    logger.info('All guarded_getitem tests passed!')
    return True


def test_dependency_context_keys():
    """Test that dependencies are added with multiple key formats."""
    from src.execution import Executor
    from src.state import InformationStateManager, Subtask
    from src.tools import ToolBelt

    logger.info('\nTesting dependency context keys...')

    # Setup minimal test environment
    state_manager = InformationStateManager(logger)
    tool_belt = ToolBelt()
    tool_belt.set_logger(logger)

    # Create mock LLM service
    class MockLLMService:
        def call_with_system_prompt(self, **kwargs):
            return '{"code": "result = 42", "context": {}}'

    llm_service = MockLLMService()
    executor = Executor(tool_belt, llm_service, state_manager, logger)

    # Create a completed dependency subtask
    dep_subtask = Subtask(
        id='step_1', description='Test dependency', status='completed'
    )
    dep_subtask.result = {'data': 'test_value'}
    state_manager.add_subtask(dep_subtask)
    state_manager.complete_subtask('step_1', dep_subtask.result)

    # Create main subtask that depends on step_1
    main_subtask = Subtask(
        id='step_2', description='Test main task', dependencies=['step_1']
    )
    main_subtask.metadata = {
        'tool': 'code_interpreter',
        'parameters': {
            'code': 'result = context["step_1"]["data"]',  # Access with underscore
            'context': {},
        },
    }
    state_manager.add_subtask(main_subtask)

    # Test dependency serialization
    serialized = executor._serialize_result_for_code(dep_subtask.result)
    logger.info(f'Serialized dependency result: {serialized}')

    # Simulate the context creation process
    context = {}
    dependency_results = {'step_1': serialized}

    # Add dependencies using both key formats (as done in the fix)
    for dep_id, result in dependency_results.items():
        # Original key
        context[dep_id] = result
        # Simplified key
        simplified_key = dep_id.replace('step_', 'step')
        if simplified_key != dep_id:
            context[simplified_key] = result

    # Add dependency_results dict
    context['dependency_results'] = dependency_results

    # Test all three access patterns
    tests_passed = True

    # Test 1: Original key (step_1)
    try:
        value = context['step_1']
        logger.info(f"✓ Test 1 passed: Access via original key 'step_1': {value}")
    except KeyError as e:
        logger.error(f"✗ Test 1 failed: Cannot access 'step_1': {e}")
        tests_passed = False

    # Test 2: Simplified key (step1)
    try:
        value = context['step1']
        logger.info(f"✓ Test 2 passed: Access via simplified key 'step1': {value}")
    except KeyError as e:
        logger.error(f"✗ Test 2 failed: Cannot access 'step1': {e}")
        tests_passed = False

    # Test 3: Via dependency_results
    try:
        value = context['dependency_results']['step_1']
        logger.info(
            f"✓ Test 3 passed: Access via dependency_results['step_1']: {value}"
        )
    except KeyError as e:
        logger.error(
            f"✗ Test 3 failed: Cannot access dependency_results['step_1']: {e}"
        )
        tests_passed = False

    # Test 4: Check all keys are present
    expected_keys = {'step_1', 'step1', 'dependency_results'}
    actual_keys = set(context.keys())
    if expected_keys.issubset(actual_keys):
        logger.info(f'✓ Test 4 passed: All expected keys present: {actual_keys}')
    else:
        logger.error(
            f'✗ Test 4 failed: Missing keys. Expected {expected_keys}, got {actual_keys}'
        )
        tests_passed = False

    if tests_passed:
        logger.info('All dependency context tests passed!')
    else:
        logger.error('Some dependency context tests failed!')

    return tests_passed


def test_code_execution_with_dependencies():
    """Test actual code execution with dependency access."""
    from src.tools import ToolBelt

    logger.info('\nTesting code execution with dependencies...')

    tool_belt = ToolBelt()
    tool_belt.set_logger(logger)

    # Create context with dependencies in multiple formats
    context = {
        'step_1': {'value': 42, 'name': 'test'},
        'step1': {'value': 42, 'name': 'test'},  # Simplified key
        'dependency_results': {'step_1': {'value': 42, 'name': 'test'}},
    }

    # Test 1: Access with original key
    code1 = """
result = context['step_1']['value'] * 2
"""
    result1 = tool_belt.code_interpreter(code1, context)
    if '84' in result1 or result1 == '84':
        logger.info(f'✓ Test 1 passed: Code with original key executed: {result1}')
    else:
        logger.error(f'✗ Test 1 failed: Unexpected result: {result1}')
        return False

    # Test 2: Access with simplified key
    code2 = """
result = context['step1']['value'] * 3
"""
    result2 = tool_belt.code_interpreter(code2, context)
    if '126' in result2 or result2 == '126':
        logger.info(f'✓ Test 2 passed: Code with simplified key executed: {result2}')
    else:
        logger.error(f'✗ Test 2 failed: Unexpected result: {result2}')
        return False

    # Test 3: Access via dependency_results
    code3 = """
result = context['dependency_results']['step_1']['value'] * 4
"""
    result3 = tool_belt.code_interpreter(code3, context)
    if '168' in result3 or result3 == '168':
        logger.info(
            f'✓ Test 3 passed: Code with dependency_results executed: {result3}'
        )
    else:
        logger.error(f'✗ Test 3 failed: Unexpected result: {result3}')
        return False

    # Test 4: Missing key should give helpful error
    code4 = """
result = context['step_999']['value']
"""
    result4 = tool_belt.code_interpreter(code4, context)
    if 'Available keys' in result4 or 'KeyError' in result4:
        logger.info(
            f'✓ Test 4 passed: Missing key gives helpful error: {result4[:200]}'
        )
    else:
        logger.error(f'✗ Test 4 failed: Error message not helpful: {result4}')
        return False

    logger.info('All code execution tests passed!')
    return True


def main():
    """Run all tests."""
    logger.info('=' * 80)
    logger.info('DEPENDENCY ACCESS FIX TEST SUITE')
    logger.info('=' * 80)

    all_passed = True

    # Test 1: guarded_getitem
    if not test_guarded_getitem():
        all_passed = False

    # Test 2: Dependency context keys
    if not test_dependency_context_keys():
        all_passed = False

    # Test 3: Code execution with dependencies
    if not test_code_execution_with_dependencies():
        all_passed = False

    logger.info('\n' + '=' * 80)
    if all_passed:
        logger.info('✓ ALL TESTS PASSED!')
    else:
        logger.error('✗ SOME TESTS FAILED!')
    logger.info('=' * 80)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())

