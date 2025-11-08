"""Test GAIA validation cases"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, Optional

from src.solver import GAIASolver
from src.utils import setup_logging


def load_validation_cases(
    file_path: str, num_cases: int = 10, indexes: Optional[list[int]] = None
) -> list:
    """Load validation cases from JSON file

    Args:
        file_path: Path to the JSON file containing validation cases
        num_cases: Number of cases to load (used when indexes is None)
        indexes: Optional list of case indexes to select. If provided, selects
                 those specific cases instead of first num_cases.

    Returns:
        List of selected validation cases
    """
    logging.info(f'Loading validation cases from {file_path}')
    with open(file_path, 'r') as f:
        cases = json.load(f)

    if indexes is not None:
        # Validate indexes
        max_index = len(cases) - 1
        invalid_indexes = [idx for idx in indexes if idx < 0 or idx > max_index]
        if invalid_indexes:
            raise ValueError(
                f'Invalid indexes: {invalid_indexes}. Valid range is 0-{max_index}'
            )
        selected_cases = [cases[idx] for idx in indexes]
        logging.info(
            f'Loaded {len(cases)} total cases, selected {len(selected_cases)} '
            f'cases at indexes: {indexes}'
        )
        return selected_cases
    else:
        logging.info(f'Loaded {len(cases)} total cases, using first {num_cases}')
        return cases[:num_cases]


def test_case(
    solver: GAIASolver, case: Dict[str, Any], case_num: int
) -> Dict[str, Any]:
    """Test a single validation case"""
    question = case.get('Question', '')
    expected_answer = case.get('Final answer', '')
    task_id = case.get('task_id', '')
    level = case.get('Level', 0)

    print(f'\n{"=" * 80}')
    print(f'TEST CASE {case_num}')
    print(f'{"=" * 80}')
    print(f'Task ID: {task_id}')
    print(f'Level: {level}')
    print(f'\nQuestion: {question}')
    print(f'\nExpected Answer: {expected_answer}')
    print('\nSolving...')

    logging.info('=-' * 80)
    logging.info(f'Starting TEST CASE {case_num}')
    logging.info(f'Task ID: {task_id}, Level: {level}')
    logging.info(f'Question: {question}')
    logging.info(f'Expected Answer: {expected_answer}')

    try:
        logging.info('Calling solver.solve()...')
        answer = solver.solve(question)

        print('\n--- RESULT ---')
        print(f'Answer: {answer.answer}')
        print(f'Confidence: {answer.confidence:.2f}')
        print(f'Sources: {len(answer.sources)} source(s)')
        if answer.sources:
            for i, source in enumerate(answer.sources[:3], 1):
                print(f'  {i}. {source}')

        # Log full answer without any truncation
        full_answer = answer.answer if answer.answer else '(empty)'
        logging.info(f'Got Answer: {full_answer}')
        logging.info(f'Confidence: {answer.confidence:.2f}')
        logging.info(f'Number of sources: {len(answer.sources)}')

        # Check if answer is correct (simple string matching)
        # Only check match if we have a non-empty answer
        # Convert to string first to handle cases where answer might be an integer
        answer_str = str(answer.answer) if answer.answer is not None else ''
        got_answer = answer_str.strip() if answer_str else ''
        expected = expected_answer.strip()

        # Task is successful only if LLM generated a proper answer (not a refusal message)
        # Check for refusal patterns
        refusal_patterns = [
            'unable to answer',
            'cannot answer',
            'failed to',
            'task(s) failed',
            'prevented gathering',
        ]
        is_refusal = any(pattern in got_answer.lower() for pattern in refusal_patterns)

        # Success requires: non-empty answer, confidence > 0, and NOT a refusal message
        is_successful = got_answer and answer.confidence > 0.0 and not is_refusal

        # Match requires both: exact match AND successful answer
        answer_match = False
        if is_successful:
            answer_match = expected.lower().strip() == got_answer.lower().strip()

        result = {
            'task_id': task_id,
            'question': question,
            'expected': expected_answer,
            'got': answer.answer,
            'confidence': answer.confidence,
            'match': answer_match,
            'success': is_successful,
        }

        print('\n--- EVALUATION ---')
        print(f'Match: {"✓ YES" if answer_match else "✗ NO"}')
        print(f'Success: {"✓ YES" if result["success"] else "✗ NO"}')

        logging.info(
            f'TEST CASE {case_num} - Match: {answer_match}, Success: {result["success"]}'
        )

        return result

    except Exception as e:
        print('\n--- ERROR ---')
        print(f'Error: {str(e)}')
        import traceback

        traceback.print_exc()

        logging.error(f'TEST CASE {case_num} - ERROR: {str(e)}')
        logging.error('Traceback:', exc_info=True)

        return {
            'task_id': task_id,
            'question': question,
            'expected': expected_answer,
            'got': None,
            'error': str(e),
            'match': False,
            'success': False,
        }


def main():
    """Test 10 validation cases"""
    # Setup logging first
    setup_logging()

    # Add a separator for new test runs
    logging.info('=' * 80)
    logging.info(f'NEW TEST RUN - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    logging.info('=' * 80)

    print('=' * 80)
    print('GAIA VALIDATION TESTING')
    print('=' * 80)

    logging.info('=' * 80)
    logging.info('GAIA VALIDATION TESTING')
    logging.info('=' * 80)

    # Initialize solver
    print('\nInitializing GAIA Solver...')
    logging.info('Initializing GAIA Solver...')
    solver = GAIASolver()
    tool_names = ', '.join(solver.tool_registry.list_tool_names())
    print(f'Registered tools: {tool_names}')
    logging.info(f'Registered tools: {tool_names}')

    # Load validation cases
    validation_file = 'gaia_dataset/validation.json'
    print(f'\nLoading validation cases from {validation_file}...')
    cases = load_validation_cases(
        validation_file,
        num_cases=10,
        # indexes=[2,3,4,5,6,7],  # [2, 3, 5, 6, 7]
    )
    print(f'Loaded {len(cases)} cases')

    # Test each case
    results = []
    for i, case in enumerate(cases, 1):
        result = test_case(solver, case, i)
        results.append(result)

    # Summary
    print(f'\n{"=" * 80}')
    print('SUMMARY')
    print(f'{"=" * 80}')

    logging.info('=' * 80)
    logging.info('TEST SUMMARY')
    logging.info('=' * 80)

    total = len(results)
    successful = sum(1 for r in results if r.get('success', False))
    matched = sum(1 for r in results if r.get('match', False))

    print(f'Total cases: {total}')
    print(
        f'Successful executions: {successful}/{total} ({successful / total * 100:.1f}%)'
    )
    print(f'Answer matches: {matched}/{total} ({matched / total * 100:.1f}%)')

    logging.info(f'Total cases: {total}')
    logging.info(
        f'Successful executions: {successful}/{total} ({successful / total * 100:.1f}%)'
    )
    logging.info(f'Answer matches: {matched}/{total} ({matched / total * 100:.1f}%)')

    print(f'\n{"=" * 80}')
    print('DETAILED RESULTS')
    print(f'{"=" * 80}')

    logging.info('=' * 80)
    logging.info('DETAILED RESULTS')
    logging.info('=' * 80)

    for i, result in enumerate(results, 1):
        print(f'\nCase {i}:')
        print(f'  Task ID: {result["task_id"]}')
        print(f'  Expected: {result["expected"]}')
        print(f'  Got: {result["got"]}')
        print(f'  Match: {"✓" if result["match"] else "✗"}')
        print(f'  Success: {"✓" if result["success"] else "✗"}')
        if 'error' in result:
            print(f'  Error: {result["error"]}')

        logging.info(
            f'Case {i}: Task ID={result["task_id"]}, Match={result["match"]} (Expected: {result["expected"]} / Got: {result["got"]}), Success={result["success"]}'
        )
        if 'error' in result:
            logging.error(f'Case {i} Error: {result["error"]}')

    logging.info('=' * 80)
    logging.info('TEST RUN COMPLETED')
    logging.info('=' * 80)


if __name__ == '__main__':
    main()
