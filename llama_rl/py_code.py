import re
import multiprocessing
import queue  # For queue.Empty
import time
import logging
import traceback
import io
import contextlib
import math  # Example allowed module
import ast
import importlib
import os
import datetime
import textwrap  # Import for dedent
from typing import Optional, Tuple, Dict, Any, List, Union

# Basic logger setup (customize as needed)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - [%(processName)s] - %(message)s",
)


# --- Target Function for Multiprocessing (MUST be top-level) ---
def _execute_code_subprocess_target(
    code: str,
    result_queue: multiprocessing.Queue,
    restricted_globals: Dict,
    timeout: float,  # Pass timeout for internal checks if needed, though primary timeout is join()
):
    """
    Target function for the execution process. Captures output and errors.

    WARNING: Runs 'exec'. Security depends heavily on restricted_globals.
    """
    start_time = time.monotonic()
    exec_success = False
    exec_error_repr = None
    exec_error_type = None
    captured_output = ""
    elapsed_time = 0.0

    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()

    try:
        # Redirect stdout/stderr
        with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(
            stderr_capture
        ):
            restricted_locals = {}
            # Compile first (syntax should be ok if we got here)
            compiled_code = compile(code, "<subprocess_string>", "exec")
            # Execute!
            exec(compiled_code, restricted_globals, restricted_locals)
        elapsed_time = time.monotonic() - start_time
        exec_success = True
        # Basic internal timeout check (less reliable than parent process join timeout)
        if elapsed_time > timeout:
            logging.warning(
                f"Subprocess: Execution finished but exceeded timeout {timeout:.1f}s (took {elapsed_time:.2f}s)"
            )
            # We can't reliably stop it here, parent process handles timeout via join
            # exec_success = False # Optionally mark as failed if slow?

    except Exception as e:
        elapsed_time = time.monotonic() - start_time
        exec_success = False
        exec_error_repr = repr(e)
        exec_error_type = type(e).__name__
        logging.warning(f"Subprocess: Execution failed: {exec_error_type}: {e}")
        # Append traceback to stderr capture if needed
        stderr_capture.write("\n--- Traceback ---\n")
        stderr_capture.write(traceback.format_exc())

    finally:
        # Retrieve captured output
        stdout_val = stdout_capture.getvalue()
        stderr_val = stderr_capture.getvalue()
        if stdout_val:
            captured_output += "--- stdout ---\n" + stdout_val
        if stderr_val:
            # Add newline if stdout also exists
            if stdout_val:
                captured_output += "\n"
            captured_output += "--- stderr ---\n" + stderr_val

        # Put pickleable results onto the queue
        try:
            result_queue.put(
                (
                    exec_success,
                    exec_error_type,
                    exec_error_repr,
                    captured_output,
                    elapsed_time,
                )
            )
        except Exception as q_err:
            logging.error(f"Subprocess: Failed to put result onto queue: {q_err}")
            # Try putting error info if possible
            try:
                result_queue.put(
                    (
                        False,
                        type(q_err).__name__,
                        repr(q_err),
                        captured_output,
                        elapsed_time,
                    )
                )
            except:
                pass  # Ignore secondary queue error


# --- CodeEvaluator Class ---
class CodeEvaluator:
    """
    Evaluates Python code extracted from a string using a separate process.

    Includes import checking, syntax checking, execution with timeout,
    output capturing, and optional code saving.

    WARNING: Uses 'exec' in a subprocess. Review security implications.
    """

    # (Keep DEFAULT_ALLOWED_MODULES_DICT and DEFAULT_SAFE_BUILTINS as before)
    DEFAULT_ALLOWED_MODULES_DICT = {"math": math}
    DEFAULT_SAFE_BUILTINS = {
        "print": print,
        "len": len,
        "range": range,
        "abs": abs,
        "min": min,
        "max": max,
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        "None": None,
        "True": True,
        "False": False,
        "isinstance": isinstance,
        "round": round,
        "sorted": sorted,
        "sum": sum,
        "pow": pow,
        "divmod": divmod,
        "enumerate": enumerate,
        "zip": zip,
        "__import__": None,
        "open": None,
        "eval": None,
        "exec": None,
        "input": None,
        "exit": None,
        "quit": None,
        "help": None,
        "dir": None,
        "getattr": None,
        "setattr": None,
        "delattr": None,
        "globals": None,
        "locals": None,
        "compile": None,
        "memoryview": None,
        "object": None,
        "property": None,
        "classmethod": None,
        "staticmethod": None,
        "super": None,
        "type": type,
    }

    def __init__(
        self,
        timeout: float = 5.0,
        allowed_module_names: Optional[List[str]] = None,
        allowed_builtins: Optional[Dict[str, Any]] = None,
        code_save_dir: Optional[str] = None,
    ):
        if timeout <= 0:
            raise ValueError("Timeout must be positive")
        self.timeout = timeout
        self.code_save_dir = code_save_dir

        self.allowed_module_names = (
            allowed_module_names if allowed_module_names is not None else ["math"]
        )

        # --- Build restricted globals for the SUBPROCESS ---
        # Build safe builtins
        current_safe_builtins = self.DEFAULT_SAFE_BUILTINS.copy()
        if allowed_builtins is not None:
            current_safe_builtins.update(allowed_builtins)

        # Build allowed modules dict (importing them in the *main* process first)
        # We pass this dict to the subprocess (pickling might be sensitive)
        allowed_modules_dict = {}
        for module_name in self.allowed_module_names:
            try:
                module = importlib.import_module(module_name)
                allowed_modules_dict[module_name] = module
            except ImportError:
                logging.error(
                    f"Config Error: Allowed module '{module_name}' not found."
                )
                # Decide: raise error or continue without it? Continuing for now.
                # raise ImportError(f"Allowed module '{module_name}' not found.")
            except Exception as e:
                logging.error(
                    f"Config Error: Error importing allowed module '{module_name}': {e}"
                )

        # Globals dictionary to be passed to the subprocess
        self.restricted_globals = {"__builtins__": current_safe_builtins}
        self.restricted_globals.update(allowed_modules_dict)
        # Lock down builtins further
        self.restricted_globals["__builtins__"].update(
            {"globals": None, "locals": None}
        )

        logging.info(
            f"CodeEvaluator initialized: timeout={self.timeout}s, "
            f"allowed_modules={list(self.allowed_module_names)}"  # Log names, not objects
        )
        # (Code saving directory setup as before)
        if self.code_save_dir:
            logging.info(f"Executed code will be saved to: {self.code_save_dir}")
            try:
                os.makedirs(self.code_save_dir, exist_ok=True)
            except OSError as e:
                logging.error(
                    f"Could not create code save directory '{self.code_save_dir}': {e}"
                )
                self.code_save_dir = None

    def _extract_python_code(self, text: str) -> Optional[str]:
        """Extracts the first Python code block."""
        if not isinstance(text, str):
            return None
        pattern = r"```(?:python|py)\s*\n?(.*?)```"
        matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
        if matches:
            extracted_code = matches[0].strip()
            # ***** FIX: Dedent the code here *****
            try:
                extracted_code = textwrap.dedent(extracted_code)
                logging.debug(
                    f"Extracted and dedented code (length {len(extracted_code)} chars)"
                )
                return extracted_code
            except Exception as e_dedent:
                logging.warning(f"Failed to dedent extracted code: {e_dedent}")
                return extracted_code.lstrip()  # Fallback to simple left strip
        return None

    def _check_imports(self, code: str) -> Tuple[bool, Optional[str]]:
        """Parses the code using AST to check for disallowed imports."""
        try:
            # Dedent before parsing to avoid indentation errors in AST
            # code_to_parse = textwrap.dedent(code) # Dedent happens in extract now
            tree = ast.parse(code)  # Parse the potentially dedented code
            disallowed_imports = []
            allowed_set = set(self.allowed_module_names)

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        # Check the actual imported name (before potential 'as')
                        module_name = alias.name.split(".")[
                            0
                        ]  # Get base module (e.g., 'os' from 'os.path')
                        if module_name not in allowed_set:
                            disallowed_imports.append(module_name)
                elif isinstance(node, ast.ImportFrom):
                    # Allow 'from allowed_module import ...'
                    if node.module and node.module.split(".")[0] not in allowed_set:
                        disallowed_imports.append(node.module.split(".")[0])

            if disallowed_imports:
                unique_disallowed = sorted(list(set(disallowed_imports)))
                error_msg = f"Disallowed imports found: {', '.join(unique_disallowed)}"
                logging.warning(error_msg)
                return False, error_msg
            else:
                logging.debug("Import check passed.")
                return True, None

        except (
            IndentationError
        ) as e_indent:  # Catch IndentationError specifically if dedent failed
            error_msg = f"AST Parsing Indentation Error: {e_indent}"
            logging.error(error_msg)  # Log as error as it prevents checks
            return False, error_msg
        except SyntaxError as e_syntax:  # Catch SyntaxError during parsing
            error_msg = f"AST Parsing Syntax Error: {e_syntax}"
            logging.warning(error_msg)  # Warn, but lint check will catch it formally
            # Allow lint check to proceed, but flag import check as potentially incomplete
            return (
                False,
                error_msg,
            )  # Or return True but add a warning flag? Let's fail import check.
        except Exception as e:
            error_msg = (
                f"AST parsing error during import check: {type(e).__name__}: {e}"
            )
            logging.error(error_msg, exc_info=True)
            return False, error_msg

    def _lint_code(self, code: str) -> Tuple[bool, Optional[str]]:
        """Performs a basic syntax check using compile()."""
        try:
            # code_to_compile = textwrap.dedent(code) # Dedent happens in extract now
            compile(code, "<string>", "exec")
            logging.debug("Syntax check passed.")
            return True, None
        except IndentationError as e_indent:  # Catch IndentationError specifically
            error_msg = f"Syntax (Indentation) Error: {e_indent}"
            logging.warning(error_msg)
            return False, error_msg
        except SyntaxError as e:
            error_msg = f"Syntax Error: {e.msg} (line {e.lineno}, offset {e.offset})"
            logging.warning(f"Syntax check failed: {error_msg}")
            return False, error_msg
        except Exception as e_compile:
            error_msg = f"Compilation Error: {type(e_compile).__name__}: {e_compile}"
            logging.error(error_msg, exc_info=True)
            return False, error_msg

    def _save_code_with_metadata(self, code: str, metrics: Dict[str, Any]):
        """Saves the executed code with metadata as comments."""
        # (Implementation remains the same as previous response)
        if not self.code_save_dir:
            return
        try:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"code_{timestamp}.py"
            filepath = os.path.join(self.code_save_dir, filename)
            metadata_lines = [
                f"# Evaluation Timestamp: {timestamp}",
                f"# Extraction Success: {metrics.get('extraction_success', 'N/A')}",
                f"# Import Check Success: {metrics.get('import_check_success', 'N/A')}",
                f"# Linting Success: {metrics.get('linting_success', 'N/A')}",
                f"# Execution Success: {metrics.get('execution_success', 'N/A')}",
                f"# Timed Out: {metrics.get('timed_out', 'N/A')}",
                f"# Execution Time (s): {metrics.get('execution_time', 0.0):.4f}",
                f"# Error Type: {metrics.get('execution_error_type', 'None')}",
                f"# Error Message: {metrics.get('execution_error_message', 'None')}",
                "#" + "-" * 20,
                "",
            ]
            content_to_save = "\n".join(metadata_lines) + code
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content_to_save)
            logging.info(f"Saved executed code to: {filepath}")
        except IOError as e:
            logging.error(f"Failed to save code to '{self.code_save_dir}': {e}")
        except Exception as e_save:
            logging.error(f"Unexpected error saving code: {e_save}", exc_info=True)

    def evaluate_code_string(self, input_string: str) -> Tuple[Dict[str, Any], str]:
        """
        Extracts, checks imports, lints, and executes Python code in a separate process.
        """
        metrics = {
            "extraction_success": False,
            "import_check_success": None,
            "import_check_output": None,
            "linting_success": None,
            "linting_output": None,
            "execution_success": None,
            "execution_output_captured": False,
            "execution_time": 0.0,
            "timed_out": False,
            "execution_error_type": None,
            "execution_error_message": None,
            "error_message": None,
            "total_time": 0.0,
        }
        captured_output = ""
        eval_start_time = time.monotonic()
        process = None  # Define process var outside try
        result_queue = None  # Define queue var outside try

        try:
            # 1. Extract Code (with dedent)
            code_to_run = self._extract_python_code(input_string)
            if not code_to_run:
                metrics["error_message"] = "No Python code block found."
                metrics["total_time"] = time.monotonic() - eval_start_time
                return metrics, captured_output  # No code to save
            metrics["extraction_success"] = True

            # 2. Check Imports using AST
            imports_ok, import_msg = self._check_imports(code_to_run)
            metrics["import_check_success"] = imports_ok
            metrics["import_check_output"] = import_msg
            if not imports_ok:
                metrics["linting_success"] = False  # Fail linting if imports fail
                metrics["execution_success"] = False
                metrics["error_message"] = f"Import Check Failed: {import_msg}"
                metrics["total_time"] = time.monotonic() - eval_start_time
                self._save_code_with_metadata(code_to_run, metrics)  # Save failed code
                return metrics, captured_output

            # 3. Lint Code (Syntax Check)
            lint_ok, lint_msg = self._lint_code(code_to_run)
            metrics["linting_success"] = lint_ok
            metrics["linting_output"] = lint_msg
            if not lint_ok:
                metrics["execution_success"] = False
                metrics["error_message"] = f"Syntax Error: {lint_msg}"
                metrics["total_time"] = time.monotonic() - eval_start_time
                self._save_code_with_metadata(code_to_run, metrics)  # Save failed code
                return metrics, captured_output

            # 4. Execute Code in Separate Process
            result_queue = multiprocessing.Queue(maxsize=1)
            process = multiprocessing.Process(
                target=_execute_code_subprocess_target,  # Use top-level function
                args=(code_to_run, result_queue, self.restricted_globals, self.timeout),
                daemon=True,
            )

            logging.debug("Starting code execution process...")
            process.start()

            # 5. Wait for Result or Timeout
            try:
                # Wait for the process queue to have a result
                (
                    exec_success,
                    err_type,
                    err_repr,
                    captured_output,
                    exec_time,
                ) = result_queue.get(
                    timeout=self.timeout  # Use the configured timeout
                )
                metrics["execution_success"] = exec_success
                metrics["execution_time"] = exec_time
                metrics["timed_out"] = False
                metrics["execution_output_captured"] = bool(captured_output)
                if not exec_success:
                    metrics["error_message"] = (
                        f"{err_type}: {err_repr}"
                        if err_type
                        else "Unknown execution error"
                    )
                    metrics["execution_error_type"] = err_type
                    metrics["execution_error_message"] = (
                        err_repr[:500] if err_repr else None
                    )

                # Join the process cleanly since it finished
                process.join(timeout=1.0)  # Short timeout for join
                if process.is_alive():
                    logging.warning(
                        "Process still alive after finishing and join timeout, killing."
                    )
                    try:
                        process.kill()
                        process.join(0.1)
                    except:
                        pass  # Ignore kill errors if already dead

            except queue.Empty:  # Timeout happened
                logging.warning(
                    f"Code execution timed out after {self.timeout:.1f}s. Terminating process."
                )
                metrics["execution_success"] = False
                metrics["timed_out"] = True
                metrics[
                    "error_message"
                ] = f"Execution timed out after {self.timeout:.1f}s"
                metrics["execution_error_type"] = "TimeoutError"
                metrics["execution_error_message"] = metrics["error_message"]
                metrics["execution_time"] = self.timeout
                captured_output = "--- Execution Timed Out ---"

                # Terminate the hung process
                if process and process.is_alive():
                    logging.debug("Attempting to terminate timed-out process...")
                    try:
                        process.terminate()
                        process.join(timeout=0.5)
                        if process.is_alive():
                            logging.warning("Process termination failed, killing.")
                            process.kill()
                            process.join(timeout=0.1)
                    except Exception as term_err:
                        logging.error(f"Error terminating/killing process: {term_err}")

            except Exception as e_q:  # Error getting from queue
                logging.error(
                    f"Error retrieving result from code execution queue: {e_q}",
                    exc_info=True,
                )
                metrics["execution_success"] = False
                metrics["error_message"] = f"Queue retrieval error: {e_q}"
                metrics["execution_error_type"] = type(e_q).__name__
                metrics["execution_error_message"] = str(e_q)
                # Try to cleanup process if it exists
                if process and process.is_alive():
                    try:
                        process.terminate()
                        process.join(0.5)
                    except:
                        pass

        except Exception as e_outer:
            # Catch errors during setup (e.g., process creation)
            logging.error(
                f"Outer error during code evaluation setup: {e_outer}", exc_info=True
            )
            metrics["error_message"] = f"Evaluation setup error: {e_outer}"
            metrics["execution_success"] = False

        finally:
            # Final cleanup for queue and process object
            if result_queue:
                try:
                    # Ensure queue is empty
                    while not result_queue.empty():
                        try:
                            result_queue.get_nowait()
                        except queue.Empty:
                            break
                    result_queue.close()
                    result_queue.join_thread()
                except Exception as e_q_clean:
                    logging.debug(f"Error cleaning up queue: {e_q_clean}")
            if process:
                if process.is_alive():
                    logging.warning(
                        "Process still alive at end of evaluation, attempting kill."
                    )
                    try:
                        process.kill()
                        process.join(0.1)
                    except:
                        pass
                process.close()  # Release process object resources

        metrics["total_time"] = time.monotonic() - eval_start_time
        logging.debug(
            f"Code evaluation completed in {metrics['total_time']:.4f}s. Success: {metrics['execution_success']}"
        )

        # 6. Save Code with Metadata (Optional)
        if code_to_run:  # Only save if code was actually extracted
            self._save_code_with_metadata(code_to_run, metrics)

        return metrics, captured_output


# --- Example Usage (ensure this block is guarded) ---
if __name__ == "__main__":
    # Example: Evaluate code and save successful attempts
    evaluator_with_save = CodeEvaluator(
        timeout=3.0,
        allowed_module_names=["math", "random"],  # Allow math and random
        code_save_dir="./executed_code_logs",  # Specify directory
    )

    test_strings = {
        "Success": """
            Some text.
            ```python
            import math
            import random # Allowed in this specific evaluator instance

            print("Calculating...")
            radius = random.uniform(1, 10)
            area = math.pi * radius**2
            print(f"Radius: {radius:.2f}, Area: {area:.2f}")
            ```
            More text.
        """,
        "SyntaxError": "```python\nprint('Hello world'\n```",
        "DisallowedImport": "```python\nimport os\nprint(os.getcwd())\n```",
        "RuntimeError": "```python\nprint(1 / 0)\n```",
        "Timeout": "```python\nimport time\nprint('Looping...')\nwhile True: time.sleep(0.1)\n```",
        "NoCode": "Just plain text here.",
        "EmptyCode": "```python\n# Empty block\n```",
        "IndentationErrorInAST": """
            ```python
            def func():
             print("Indented incorrectly") # This causes AST parse error
            ```
        """,
        "AllowedFromImport": """
            ```python
            from math import sqrt, pi # Allowed because 'math' is allowed
            print(f"Sqrt(2): {sqrt(2)}")
            print(f"Pi: {pi}")
            ```
        """,
    }

    for name, text in test_strings.items():
        print(f"\n--- Test Case: {name} ---")
        try:
            metrics, output = evaluator_with_save.evaluate_code_string(text)
            print("Metrics:", metrics)
            print("Output:\n", output if output.strip() else "[No Output Captured]")
        except Exception as e_main:
            print(
                f"ERROR DURING EVALUATION CALL: {e_main}"
            )  # Catch errors in the call itself
        print("-" * 20)

    # Example: Disallow math module
    evaluator_no_math = CodeEvaluator(
        timeout=2.0, allowed_module_names=[], code_save_dir="./executed_code_logs"
    )  # Empty list disallows all optional imports
    print("\n--- Test Case: Disallowed Math ---")
    metrics_no_math, output_no_math = evaluator_no_math.evaluate_code_string(
        "```python\nimport math\nprint(math.pi)\n```"
    )
    print("Metrics:", metrics_no_math)
    print("Output:\n", output_no_math)
    print("-" * 20)
