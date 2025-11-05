"""
Test script to verify typing module can be imported in code_interpreter.
"""

import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
)

logger = logging.getLogger(__name__)


def test_typing_import():
    """Test that typing module can be imported and used."""
    from src.core.tool_belt import ToolBelt

    logger.info('Testing typing module import in code_interpreter...')

    tool_belt = ToolBelt()
    tool_belt.set_logger(logger)

    # Test 1: Import typing module
    code1 = """
from typing import List, Dict, Optional

result = "typing module imported successfully"
"""
    result1 = tool_belt.code_interpreter(code1, {})
    if 'imported successfully' in result1 and 'Error' not in result1:
        logger.info(f'✓ Test 1 passed: typing module imported: {result1}')
    else:
        logger.error(f'✗ Test 1 failed: {result1}')
        return False

    # Test 2: Import multiple typing types
    code2 = """
from typing import List, Dict, Optional, Union, Tuple, Set

# Just verify they can all be imported
result = "All typing types imported successfully"
"""
    result2 = tool_belt.code_interpreter(code2, {})
    if 'imported successfully' in result2 and 'Error' not in result2:
        logger.info(f'✓ Test 2 passed: Multiple typing types imported: {result2}')
    else:
        logger.error(f'✗ Test 2 failed: {result2}')
        return False

    # Test 3: Use typing types in code (without annotations)
    code3 = """
from typing import List, Dict

# Create variables without type annotations (RestrictedPython limitation)
items = [1, 2, 3, 4, 5]
data = {'sum': sum(items), 'count': len(items)}
result = data
"""
    result3 = tool_belt.code_interpreter(code3, {})
    if 'sum' in result3 and '15' in result3 and 'Error' not in result3:
        logger.info(f'✓ Test 3 passed: typing module usable in code: {result3}')
    else:
        logger.error(f'✗ Test 3 failed: {result3}')
        return False

    # Test 4: Verify typing doesn't break existing code
    code4 = """
from typing import Optional

test_dict = {'a': 10, 'b': 20}
value = test_dict.get('a')  # No type annotation
result = value
"""
    result4 = tool_belt.code_interpreter(code4, {})
    if '10' in result4 and 'Error' not in result4:
        logger.info(f"✓ Test 4 passed: typing import doesn't break code: {result4}")
    else:
        logger.error(f'✗ Test 4 failed: {result4}')
        return False

    logger.info('All typing module tests passed!')
    return True


def main():
    """Run typing import tests."""
    logger.info('=' * 80)
    logger.info('TYPING MODULE IMPORT TEST')
    logger.info('=' * 80)

    if test_typing_import():
        logger.info('\n' + '=' * 80)
        logger.info('✓ ALL TYPING TESTS PASSED!')
        logger.info('=' * 80)
        return 0
    else:
        logger.error('\n' + '=' * 80)
        logger.error('✗ TYPING TESTS FAILED!')
        logger.error('=' * 80)
        return 1


if __name__ == '__main__':
    sys.exit(main())
