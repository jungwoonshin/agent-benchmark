"""Code interpreter for safe Python code execution using RestrictedPython."""

import io
import json
import logging
import math
import sys
from typing import Any, Dict, Optional

from RestrictedPython import (  # type: ignore
    compile_restricted,
    limited_builtins,
    safe_globals,
)

from .error_analysis import ExecutionResultAnalyzer


def guarded_getitem(container, key):
    """Safe item access wrapper with improved error messages."""
    # Handle case where container is a string (should use integer indices)
    if isinstance(container, str):
        if isinstance(key, int):
            try:
                return container[key]
            except IndexError:
                raise IndexError(
                    f'String index {key} out of range. String length: {len(container)}'
                )
        else:
            raise TypeError(
                f"string indices must be integers, not '{type(key).__name__}'"
            )

    try:
        return container[key]
    except KeyError:
        # Provide helpful error message with available keys
        if isinstance(container, dict):
            available_keys = list(container.keys())
            error_msg = (
                f"Key '{key}' not found in container. Available keys: {available_keys}"
            )
            raise KeyError(error_msg)
        raise
    except TypeError as e:
        # Re-raise with more context if it's a type error
        if 'indices must be' in str(e):
            raise TypeError(
                f'Cannot use {type(key).__name__} as index for {type(container).__name__}'
            ) from e
        raise


def guarded_iter(iterable):
    """Safe iterator wrapper."""
    return iter(iterable)


class CodeInterpreter:
    """Code interpreter for safe Python code execution."""

    def __init__(self, logger: logging.Logger):
        """
        Initialize the code interpreter.

        Args:
            logger: Logger instance for logging execution details
        """
        self.logger = logger

    def execute(
        self, python_code: str, context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Executes Python code in a sandboxed environment using RestrictedPython.
        This is the primary tool for all math, logic, and complex data processing.

        Args:
            python_code: Python code to execute
            context: Optional context dictionary with variables available to the code

        Returns:
            String representation of the execution result or error message
        """
        self.logger.info('Code interpreter called.')
        self.logger.debug(f'Executing code snippet: {python_code[:150]}...')

        if context is None:
            context = {}

        # Define a custom printer to capture output
        class _Printer:
            def __init__(self, output_buffer):
                self.output_buffer = output_buffer

            def write(self, text):
                self.output_buffer.write(text)

            def _call_print(self, *objects, sep=' ', end='\n'):
                # This is the method RestrictedPython will call for print
                # It should write to the captured output buffer
                self.write(sep.join(map(str, objects)) + end)

        # Allow all imports - note: this removes security restrictions
        # Use with caution in production environments
        def safe_import(name, *args, **kwargs):
            """Import function that allows all modules."""
            try:
                return __import__(name, *args, **kwargs)
            except ImportError as e:
                # Re-raise the original ImportError with more context
                raise ImportError(f"Failed to import '{name}': {e}")

        try:
            # Prepare safe execution environment
            safe_builtins = limited_builtins.copy()
            safe_builtins['__import__'] = safe_import
            safe_builtins['__name__'] = 'restricted_module'
            safe_builtins['__metaclass__'] = type
            safe_builtins['_getattr_'] = getattr
            # Add RestrictedPython utility functions for iteration and item access
            safe_builtins['_getiter_'] = guarded_iter
            safe_builtins['_getitem_'] = guarded_getitem
            safe_builtins['_write_'] = lambda x: x  # Allow writing to variables

            # Add RestrictedPython utility functions for in-place operations
            # These are needed for augmented assignments like +=, -=, *=, etc.
            # RestrictedPython generates code that calls these functions
            def _inplacevar_(obj, *args):
                """Helper for RestrictedPython in-place variable operations.
                RestrictedPython uses this for augmented assignments like x += y.
                Can be called as _inplacevar_(operator, obj1, obj2) where operator
                is a string like '+=' or '*='.
                """
                # If called with operator string as first arg (RestrictedPython pattern)
                if args and isinstance(obj, str) and obj.endswith('='):
                    op = obj.rstrip('=')
                    obj1 = args[0] if len(args) > 0 else obj
                    obj2 = args[1] if len(args) > 1 else 0
                    # Use _inplace_ logic to compute result
                    if op == '+':
                        return obj1 + obj2
                    elif op == '-':
                        return obj1 - obj2
                    elif op == '*':
                        return obj1 * obj2
                    elif op == '/':
                        return obj1 / obj2
                    elif op == '//':
                        return obj1 // obj2
                    elif op == '%':
                        return obj1 % obj2
                    elif op == '**':
                        return obj1**obj2
                    elif op == '<<':
                        return obj1 << obj2
                    elif op == '>>':
                        return obj1 >> obj2
                    elif op == '&':
                        return obj1 & obj2
                    elif op == '|':
                        return obj1 | obj2
                    elif op == '^':
                        return obj1 ^ obj2
                    else:
                        return obj1
                # Otherwise, just return the object (simple assignment case)
                return obj

            def _inplace_(op, obj1, obj2):
                """Helper for RestrictedPython in-place operations.
                RestrictedPython uses this for augmented assignments.
                Note: RestrictedPython may pass operators with '=' suffix (e.g., '+=').
                """
                try:
                    # Strip '=' if present to normalize operator
                    op_normalized = op.rstrip('=')

                    if op_normalized == '+':
                        return obj1 + obj2
                    elif op_normalized == '-':
                        return obj1 - obj2
                    elif op_normalized == '*':
                        return obj1 * obj2
                    elif op_normalized == '/':
                        return obj1 / obj2
                    elif op_normalized == '//':
                        return obj1 // obj2
                    elif op_normalized == '%':
                        return obj1 % obj2
                    elif op_normalized == '**':
                        return obj1**obj2
                    elif op_normalized == '<<':
                        return obj1 << obj2
                    elif op_normalized == '>>':
                        return obj1 >> obj2
                    elif op_normalized == '&':
                        return obj1 & obj2
                    elif op_normalized == '|':
                        return obj1 | obj2
                    elif op_normalized == '^':
                        return obj1 ^ obj2
                    else:
                        # Fallback: try to call the operation directly
                        return obj1
                except (IndexError, KeyError):
                    # Handle cases where augmented assignment attempts to modify
                    # a non-existent index in a list or key in a dictionary
                    self.logger.warning(
                        f"Augmented assignment failed for non-existent item/index with operation '{op}'. Skipping."
                    )
                    return obj1  # Return original object to prevent error
                except Exception:
                    # If operation fails, return the original object
                    return obj1

            # Add these to both safe_builtins and safe_locals for RestrictedPython
            safe_builtins['_inplacevar_'] = _inplacevar_
            safe_builtins['_inplace_'] = _inplace_

            # Add common built-in types to safe_builtins
            safe_builtins['set'] = set
            safe_builtins['sorted'] = sorted
            # Add essential built-in functions for type checking
            safe_builtins['isinstance'] = isinstance
            safe_builtins['type'] = type

            # Create a buffer to capture print output
            output_buffer = io.StringIO()

            # Create a print function that writes directly to output_buffer
            # Store buffer in a list to ensure it's not garbage collected
            # and remains accessible in all execution contexts
            buffer_ref = [output_buffer]

            # Create a print result object that RestrictedPython might expect
            class PrintResult:
                """Result object that RestrictedPython might expect from _print_."""

                def __init__(self, buffer_ref):
                    self.buffer_ref = buffer_ref

                def _call_print(self, *objects, sep=' ', end='\n'):
                    """Write to buffer."""
                    try:
                        buf = self.buffer_ref[0]
                        if buf is not None:
                            buf.write(sep.join(map(str, objects)) + end)
                    except (AttributeError, IndexError, TypeError):
                        pass
                    return None

            def _print_func(*objects, sep=' ', end='\n'):
                """Print function wrapper that writes to output buffer."""
                # Write directly to buffer
                try:
                    buf = buffer_ref[0]
                    if buf is not None:
                        buf.write(sep.join(map(str, objects)) + end)
                except (AttributeError, IndexError, TypeError):
                    pass
                # Return a PrintResult object that has _call_print
                # RestrictedPython might call ._call_print on the result
                return PrintResult(buffer_ref)

            safe_builtins['_print_'] = _print_func

            # RestrictedPython creates a 'printed' variable for print statements
            # Create a mock object that has _call_print to satisfy RestrictedPython
            class PrintedList(list):
                """List that RestrictedPython uses for print statements."""

                def _call_print(self, *args, **kwargs):
                    """Dummy method - actual printing is handled by _print_."""
                    return None

            printed = PrintedList()  # RestrictedPython will use this

            safe_locals = {
                '__builtins__': safe_builtins,
                'math': math,
                'json': json,
                'printed': printed,  # Required by RestrictedPython for print statements
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict,
                'tuple': tuple,
                'set': set,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'any': any,
                'all': all,
                'isinstance': isinstance,  # Essential for type checking
                'type': type,  # Essential for type checking
                'print': _print_func,  # Also make our custom print available as 'print'
                # Add RestrictedPython utility functions to locals as well
                '_inplacevar_': _inplacevar_,
                '_inplace_': _inplace_,
                # Add all standard built-in exceptions
                'Exception': Exception,
                'BaseException': BaseException,
                'ValueError': ValueError,
                'TypeError': TypeError,
                'KeyError': KeyError,
                'IndexError': IndexError,
                'AttributeError': AttributeError,
                'NameError': NameError,
                'ZeroDivisionError': ZeroDivisionError,
                'ImportError': ImportError,
                'RuntimeError': RuntimeError,
                'NotImplementedError': NotImplementedError,
                'StopIteration': StopIteration,
                'AssertionError': AssertionError,
                'OSError': OSError,
                'IOError': IOError,
                'EOFError': EOFError,
                'MemoryError': MemoryError,
                'RecursionError': RecursionError,
                'SystemError': SystemError,
                'LookupError': LookupError,
                'ArithmeticError': ArithmeticError,
            }

            # Add context dictionary itself as a variable so code can access context['key']
            safe_locals['context'] = context

            # Also add context variables as individual keys for easier access
            # But first, ensure we serialize any non-serializable objects
            for key, value in context.items():
                # Skip if value is a string and we're trying to use it as a dict
                # This prevents the "string indices must be integers" error
                if isinstance(value, str):
                    # Strings should be added as-is, not treated as dicts
                    safe_locals[key] = value
                else:
                    safe_locals[key] = value

            # Create a variable to capture the result
            # If code has explicit return or result assignment, use that
            # Otherwise, try to capture the last expression
            result_var = 'output_result'  # Use a valid name (no underscores at start)

            # Prepare code - if it doesn't assign to result or return, wrap it
            code_lines = python_code.strip().split('\n')

            # Check if code already has result assignment or return
            has_result = any(
                'result' in line or 'return' in line for line in code_lines
            )

            if not has_result and len(code_lines) == 1:
                # Single expression - assign to result
                modified_code = f'{result_var} = {python_code.strip()}'
            elif not has_result:
                # Multiple lines - add result assignment at the end if last line is expression
                last_line = code_lines[-1].strip()
                if last_line and not last_line.startswith(
                    ('if', 'for', 'while', 'def', 'class', 'import', 'from')
                ):
                    code_lines[-1] = f'{result_var} = {last_line}'
                    modified_code = '\n'.join(code_lines)
                else:
                    modified_code = python_code
            else:
                modified_code = python_code

            self.logger.debug(f'Modified code for execution: {modified_code[:200]}...')

            # Compile with RestrictedPython for safety
            compiled = compile_restricted(
                modified_code, filename='<string>', mode='exec'
            )

            # Check for compilation errors
            # compile_restricted returns a named tuple with 'code' and 'errors' attributes
            if hasattr(compiled, 'errors') and compiled.errors:
                error_msg = '; '.join(compiled.errors)
                self.logger.error(f'Code compilation failed: {error_msg}')
                return f'Compilation Error: {error_msg}'

            # Get the actual code object
            byte_code = compiled.code if hasattr(compiled, 'code') else compiled

            # Capture stdout
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                # Update safe_globals with our custom __import__
                execution_globals = safe_globals.copy()
                execution_globals['__builtins__'] = safe_builtins
                execution_globals['__import__'] = safe_import
                # Add RestrictedPython utility functions to globals as well
                execution_globals['_getiter_'] = guarded_iter
                execution_globals['_getitem_'] = guarded_getitem
                execution_globals['_write_'] = lambda x: x
                execution_globals['_print_'] = (
                    _print_func  # Assign custom print function
                )
                # Add in-place operation utilities to globals
                execution_globals['_inplacevar_'] = _inplacevar_
                execution_globals['_inplace_'] = _inplace_
                # Add common built-ins to globals
                execution_globals['set'] = set
                execution_globals['sorted'] = sorted
                execution_globals['isinstance'] = isinstance
                execution_globals['type'] = type
                # Add modules to globals for direct usage
                execution_globals['math'] = math
                execution_globals['json'] = json
                # Add all standard built-in exceptions to globals
                execution_globals['Exception'] = Exception
                execution_globals['BaseException'] = BaseException
                execution_globals['ValueError'] = ValueError
                execution_globals['TypeError'] = TypeError
                execution_globals['KeyError'] = KeyError
                execution_globals['IndexError'] = IndexError
                execution_globals['AttributeError'] = AttributeError
                execution_globals['NameError'] = NameError
                execution_globals['ZeroDivisionError'] = ZeroDivisionError
                execution_globals['ImportError'] = ImportError
                execution_globals['RuntimeError'] = RuntimeError
                execution_globals['NotImplementedError'] = NotImplementedError
                execution_globals['StopIteration'] = StopIteration
                execution_globals['AssertionError'] = AssertionError
                execution_globals['OSError'] = OSError
                execution_globals['IOError'] = IOError
                execution_globals['EOFError'] = EOFError
                execution_globals['MemoryError'] = MemoryError
                execution_globals['RecursionError'] = RecursionError
                execution_globals['SystemError'] = SystemError
                execution_globals['LookupError'] = LookupError
                execution_globals['ArithmeticError'] = ArithmeticError

                # Execute the code
                exec(byte_code, execution_globals, safe_locals)

                # Get the result
                output = output_buffer.getvalue()  # Get captured output

                # Try to get result in order of preference:
                # 1. Explicitly assigned result variable
                # 2. Our output_result variable
                # 3. Captured print output
                # 4. Any variable named 'answer' or 'output'
                result = None

                if 'result' in safe_locals and safe_locals['result'] is not None:
                    result = safe_locals['result']
                elif result_var in safe_locals and safe_locals[result_var] is not None:
                    result = safe_locals[result_var]
                elif output.strip():
                    result = output.strip()
                elif 'answer' in safe_locals and safe_locals['answer'] is not None:
                    result = safe_locals['answer']
                elif 'output' in safe_locals and safe_locals['output'] is not None:
                    result = safe_locals['output']

                if result is None:
                    # No explicit result found, capture all computed variables with their values
                    available_vars = {}
                    # Exclude built-in names and context/internal variables
                    excluded_names = {
                        '__builtins__',
                        '__name__',
                        '__metaclass__',
                        'context',
                        'dependency_results',
                        'math',
                        'json',
                        'datetime',
                        're',
                        'itertools',
                        'collections',
                        'functools',
                        'operator',
                        'statistics',
                        'str',
                        'int',
                        'float',
                        'bool',
                        'list',
                        'dict',
                        'tuple',
                        'set',
                        'len',
                        'range',
                        'enumerate',
                        'zip',
                        'sum',
                        'min',
                        'max',
                        'abs',
                        'round',
                        'sorted',
                        'any',
                        'all',
                        'print',
                    }

                    for k, v in safe_locals.items():
                        # Include user-defined variables (not starting with _)
                        if (
                            not k.startswith('_')
                            and k not in excluded_names
                            and k not in safe_locals.get('__builtins__', {})
                        ):
                            # Serialize values to JSON-serializable format
                            try:
                                # Try to convert to JSON-serializable format
                                if isinstance(v, (str, int, float, bool, type(None))):
                                    available_vars[k] = v
                                elif isinstance(v, (list, tuple)):
                                    # Convert lists/tuples, limiting size for readability
                                    available_vars[k] = list(v)[
                                        :100
                                    ]  # Limit to 100 items
                                elif isinstance(v, dict):
                                    # Convert dicts, limiting size
                                    limited_dict = {}
                                    for i, (dk, dv) in enumerate(v.items()):
                                        if i >= 50:  # Limit to 50 key-value pairs
                                            break
                                        if isinstance(
                                            dv, (str, int, float, bool, type(None))
                                        ):
                                            limited_dict[dk] = dv
                                        else:
                                            limited_dict[dk] = str(dv)[
                                                :200
                                            ]  # Truncate complex values
                                    available_vars[k] = limited_dict
                                else:
                                    # For other types, convert to string representation
                                    available_vars[k] = str(v)[
                                        :500
                                    ]  # Truncate long strings
                            except Exception:
                                # If serialization fails, use string representation
                                available_vars[k] = str(v)[:500]

                    if available_vars:
                        # Return structured result with actual values
                        try:
                            # Try to return as JSON for structured data
                            result = json.dumps(available_vars, indent=2, default=str)
                        except Exception:
                            # Fallback to formatted string if JSON fails
                            result = 'Code executed. Computed variables:\n'
                            for k, v in available_vars.items():
                                result += f'  {k} = {v}\n'
                    else:
                        result = 'Code executed successfully but produced no output.'

                # Analyze result to determine if it's an error and normalize it
                is_error, result_str, error_message = (
                    ExecutionResultAnalyzer.normalize_error_result(result)
                )

                # Log appropriately based on result
                if is_error:
                    self.logger.error(
                        f'Code execution failed. Error: {error_message[:500] if error_message else "Unknown error"}'
                    )
                else:
                    self.logger.info(
                        f'Code executed successfully. Output: {result_str[:200]}...'
                    )
                return result_str

            finally:
                # Restore stdout
                sys.stdout = old_stdout

        except SyntaxError as e:
            self.logger.error(f'Code syntax error: {e}')
            return f'Syntax Error: {e}'
        except NameError as e:
            self.logger.error(f'Code name error (undefined variable): {e}')
            return f'Name Error: {e}. Available context: {list(context.keys())}'
        except ImportError as e:
            # All imports are allowed, so this is likely a missing module issue
            self.logger.error(f'Code import error: {e}')
            return f'Import Error: {e}. The module may not be installed or may not exist. You may need to install it or use an alternative approach.'
        except Exception as e:
            self.logger.error(f'Code execution FAILED: {e}', exc_info=True)
            return f'Execution Error: {type(e).__name__}: {e}'
