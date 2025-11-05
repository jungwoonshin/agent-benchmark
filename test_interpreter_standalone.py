"""Standalone test code for interpreter.py using cases from log.txt"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from code_interpreter.interpreter import CodeInterpreter


def setup_logger():
    """Setup logger for testing"""
    logger = logging.getLogger("test_interpreter")
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(handler)
    return logger


def test_simple_execution():
    """Test case from log.txt: Simple string assignment"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)

    # Test case from log: line 605
    code = "species = 'clownfish'\nspecies"
    result = interpreter.execute(code)
    assert "clownfish" in result, f"Expected 'clownfish' in result, got: {result}"
    print(f"✓ Test 1 passed: Simple execution - {result[:50]}")


def test_code_with_context():
    """Test case from log.txt: Code with context variables"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)

    # Test case from log: line 1393 - zip codes formatting
    code = """zip_codes = ['94133', '98105', '33139']
result = ','.join(zip_codes)
result"""
    context = {"step_3": {"search_results": []}}
    result = interpreter.execute(code, context)
    assert "94133,98105,33139" in result, f"Expected zip codes in result, got: {result}"
    print(f"✓ Test 2 passed: Code with context - {result[:50]}")


def test_math_operations():
    """Test mathematical operations"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)

    code = """import math
result = math.sqrt(16) + math.pi
result"""
    result = interpreter.execute(code)
    assert "4" in result or "7.14" in result or "7.1" in result, f"Expected math result, got: {result}"
    print(f"✓ Test 3 passed: Math operations - {result[:50]}")


def test_list_operations():
    """Test list operations"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)

    code = """numbers = [1, 2, 3, 4, 5]
average = sum(numbers) / len(numbers)
average"""
    result = interpreter.execute(code)
    assert "3" in result or "3.0" in result, f"Expected average 3, got: {result}"
    print(f"✓ Test 4 passed: List operations - {result[:50]}")


def test_string_operations():
    """Test string operations"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)

    code = """text = "Finding Nemo"
result = text.lower().replace(" ", "_")
result"""
    result = interpreter.execute(code)
    assert "finding_nemo" in result, f"Expected 'finding_nemo', got: {result}"
    print(f"✓ Test 5 passed: String operations - {result[:50]}")


def test_dict_access():
    """Test dictionary access with context"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)

    code = """result = step_1
result"""
    context = {"step_1": "clownfish"}
    result = interpreter.execute(code, context)
    assert "clownfish" in result, f"Expected 'clownfish' in result, got: {result}"
    print(f"✓ Test 6 passed: Dict access - {result[:50]}")


def test_error_handling_undefined_variable():
    """Test error handling for undefined variables (from log.txt line 5204)"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)

    # This should fail with NameError
    code = "result = undefined_variable"
    result = interpreter.execute(code)
    assert "Name Error" in result or "NameError" in result, f"Expected NameError, got: {result}"
    print(f"✓ Test 7 passed: Undefined variable error - {result[:50]}")


def test_error_handling_import_restriction():
    """Test import restriction (from log.txt line 5204 - 're' is not defined)"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)

    # 're' should be available, but let's test a restricted import
    code = """import os
result = os.getcwd()"""
    result = interpreter.execute(code)
    assert "Import Error" in result or "ImportError" in result or "not allowed" in result.lower(), \
        f"Expected ImportError for 'os', got: {result}"
    print(f"✓ Test 8 passed: Import restriction - {result[:50]}")


def test_regex_operations():
    """Test regex operations (re module should be available)"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)

    code = """import re
pattern = r'\\d{5}'
text = 'The zip code is 94133'
match = re.search(pattern, text)
result = match.group(0) if match else 'No match'
result"""
    result = interpreter.execute(code)
    assert "94133" in result, f"Expected '94133' in result, got: {result}"
    print(f"✓ Test 9 passed: Regex operations - {result[:50]}")


def test_json_operations():
    """Test JSON operations"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)

    code = """import json
data = {'zip_codes': ['94133', '98105'], 'count': 2}
result = json.dumps(data)
result"""
    result = interpreter.execute(code)
    assert "94133" in result and "98105" in result, f"Expected JSON with zip codes, got: {result}"
    print(f"✓ Test 10 passed: JSON operations - {result[:50]}")


def test_complex_data_processing():
    """Test complex data processing from search results"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)

    code = """# Process search results
zip_codes = ['94133', '98105', '33139']
# Filter and format
filtered = [z for z in zip_codes if len(z) == 5]
result = ','.join(sorted(filtered))
result"""
    context = {}
    result = interpreter.execute(code, context)
    assert "33139" in result and "94133" in result and "98105" in result, \
        f"Expected all zip codes, got: {result}"
    print(f"✓ Test 11 passed: Complex data processing - {result[:50]}")


def test_augmented_assignment():
    """Test augmented assignments (should work with RestrictedPython)"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)

    code = """x = 5
x += 3
x *= 2
result = x
result"""
    result = interpreter.execute(code)
    assert "16" in result, f"Expected 16, got: {result}"
    print(f"✓ Test 12 passed: Augmented assignment - {result[:50]}")


def test_print_output():
    """Test print statements"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)

    code = """print("Processing data...")
print("Zip codes: 94133, 98105")
result = "done"
result"""
    result = interpreter.execute(code)
    assert "Processing data" in result or "done" in result, \
        f"Expected print output or result, got: {result}"
    print(f"✓ Test 13 passed: Print output - {result[:100]}")


def test_exception_handling():
    """Test exception handling"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)

    code = """try:
    result = 1 / 0
except ZeroDivisionError as e:
    result = f"Error: {str(e)}"
result"""
    result = interpreter.execute(code)
    assert "Error" in result or "ZeroDivisionError" in result, \
        f"Expected error handling, got: {result}"
    print(f"✓ Test 14 passed: Exception handling - {result[:50]}")


def test_multiple_statements():
    """Test multiple statements"""
    logger = setup_logger()
    interpreter = CodeInterpreter(logger)

    code = """species = 'clownfish'
location = 'Florida'
year = 2018
result = f"{species} found in {location} in {year}"
result"""
    result = interpreter.execute(code)
    assert "clownfish" in result and "Florida" in result, \
        f"Expected combined result, got: {result}"
    print(f"✓ Test 15 passed: Multiple statements - {result[:50]}")


def run_all_tests():
    """Run all test cases"""
    print("=" * 70)
    print("Running standalone tests for interpreter.py")
    print("=" * 70)

    tests = [
        test_simple_execution,
        test_code_with_context,
        test_math_operations,
        test_list_operations,
        test_string_operations,
        test_dict_access,
        test_error_handling_undefined_variable,
        test_error_handling_import_restriction,
        test_regex_operations,
        test_json_operations,
        test_complex_data_processing,
        test_augmented_assignment,
        test_print_output,
        test_exception_handling,
        test_multiple_statements,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} failed: {e}")
            failed += 1

    print("=" * 70)
    print(f"Test Summary: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)





