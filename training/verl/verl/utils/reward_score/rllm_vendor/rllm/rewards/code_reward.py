"""
This module contains the RewardCode class, which evaluates code datasets answers
and assigns rewards based on their correctness on unit tests.
"""

import ast
import json
import multiprocessing
import re
import time
from multiprocessing import Manager
from queue import Empty
from typing import Any

from rllm.rewards.code_utils.firejail_exec import code_exec_firejail as lc_code_exec
from rllm.rewards.code_utils.humanevalplus import get_num_test_cases

# from rllm.rewards.code_utils.swebench import swebench_check_correctness
from rllm.rewards.code_utils.humanevalplus import run_test as humanevalplus_run_test
from rllm.rewards.code_utils.kodcode import code_exec as kod_code_exec

# from rllm.rewards.code_utils.code_contests import run_test as code_contests_run_test
from rllm.rewards.code_utils.livecodebench import run_test as lcb_run_test
from rllm.rewards.code_utils.taco import run_test as taco_run_test
from rllm.rewards.reward_types import RewardConfig, RewardOutput, RewardType
from rllm.tools.code_tools.code_tool import CodeTool
from rllm.tools.code_tools.together_tool import TogetherCodeTool


def extract_code_from_model(model_response: str):
    """
    Extracts the code from a Markdown-style code block in an LLM output.

    Parameters:
        model_response (str): The text output from the LLM.

    Returns:
        str: The extracted code, or an empty string if no code block is found.
    """
    code_blocks = re.findall(r"```(?:\w+)?\n(.*?)```", model_response, re.DOTALL)
    if not code_blocks:
        return None
    return code_blocks[-1].strip()


def clean_code_main_block(code: str) -> str:
    """
    Removes `if __name__ == "__main__"` blocks from Python code.

    Args:
        code (str): The input Python code.

    Returns:
        str: Cleaned code without the main execution block.
    """
    code_lines = code.split("\n")
    filtered_lines = []
    skip_block = False

    for line in code_lines:
        if line.strip().startswith('if __name__ == "__main__"') or line.strip().startswith("if __name__ == '__main__'"):
            skip_block = True
            continue
        if skip_block:
            # Check if we're out of the block (less indentation)
            if line.strip() and not line.startswith(" ") and not line.startswith("\t"):
                skip_block = False
            else:
                continue
        filtered_lines.append(line)

    return "\n".join(filtered_lines)


def check_correctness(tests: list[dict[str, str]] | dict[str, list[str]], code: str, test_fn, timeout_per_test: int = 12, max_tests: int = 15) -> tuple[bool, dict[str, Any]]:
    """
    Check if generated code passes all test cases within a timeout period.

    Args:
        tests: Test cases in either list of dictionaries or dictionary of lists format
        code: Generated code to test
        test_fn: Function to run tests
        timeout: Maximum execution time in seconds before killing process

    Returns:
        tuple: (bool, dict) where:
            - bool: True if all tests pass, False otherwise
            - dict: Detailed test results with test cases and pass/fail status
    """
    manager = Manager()
    test_results = manager.list()

    def evaluate_code(tests, generation, debug, test_results, test_fn):
        """Helper function to run tests in separate process."""
        try:
            test_results.append(test_fn(tests, test=generation, debug=debug, timeout=timeout_per_test))
        except Exception as e:
            print(f"Error in evaluate_code: {e}")

    original_tests = tests
    if isinstance(tests, list):
        list_tests = tests
        total_tests = len(list_tests)
        if total_tests > max_tests:
            # Sort indices by test input length and take the max_tests longest ones
            selected_indices = sorted(range(total_tests), key=lambda i: len(list_tests[i]["input"]), reverse=True)[:max_tests]
            tests = [list_tests[i] for i in selected_indices]
        num_tests = len(tests)
    else:
        dict_tests = tests
        total_tests = len(dict_tests["inputs"])
        if total_tests > max_tests:
            # Select the tests with the longest input length.
            selected_indices = sorted(range(total_tests), key=lambda i: len(dict_tests["inputs"][i]), reverse=True)[:max_tests]
            # Create a new dict with only the selected test cases
            selected_tests: dict[str, list[str]] = {"inputs": [dict_tests["inputs"][i] for i in selected_indices], "outputs": [dict_tests["outputs"][i] for i in selected_indices]}
            tests = selected_tests
        num_tests = len(tests["inputs"])

    process = multiprocessing.Process(target=evaluate_code, args=(tests, code, False, test_results, test_fn))
    process.start()
    process.join()

    if process.is_alive():
        process.kill()
    test_results_list = list(test_results)

    detailed_results: dict[str, Any] = {"all_passed": False, "test_results": [], "total_tests": num_tests, "passed_tests": 0}

    if len(test_results_list) == 0:
        return False, detailed_results

    test_results_data = test_results_list[0]
    passed_results = [r == True for r in test_results_data]

    # Create detailed test results
    test_results_list_typed: list[dict[str, Any]] = detailed_results["test_results"]
    if isinstance(original_tests, list):
        assert isinstance(tests, list)
        for i, (test, result) in enumerate(zip(tests, passed_results, strict=False)):
            test_results_list_typed.append({"input": test.get("input", ""), "expected": test.get("output", ""), "passed": result})
    else:
        assert isinstance(tests, dict)
        for i, (inp, out, result) in enumerate(zip(tests["inputs"], tests["outputs"], passed_results, strict=False)):
            test_results_list_typed.append({"input": inp, "expected": out, "passed": result})

    detailed_results["passed_tests"] = sum(passed_results)
    detailed_results["all_passed"] = all(passed_results)

    return all(passed_results), detailed_results


def postprocess_lcb_sample(sample):
    sample_inputs = [sample["input"] for sample in sample]
    sample_outputs = [sample["output"] for sample in sample]

    sample_dict = {
        "inputs": sample_inputs,
        "outputs": sample_outputs,
    }

    if sample[0].get("testtype") == "functional":
        metadata = sample[0].get("metadata", {})
        fn_name = metadata.get("func_name") or sample[0].get("fn_name")
        if fn_name is not None:
            # Fill in the blank for call-based problems. Some DeepCoder rows are
            # tagged functional but only provide stdin/stdout tests.
            sample_dict["fn_name"] = fn_name

    sample = {
        "input_output": json.dumps(sample_dict),
    }
    return sample


def _select_longest_input_tests(sample: list[dict[str, Any]], max_tests: int | None) -> list[dict[str, Any]]:
    if max_tests is None or max_tests <= 0 or len(sample) <= max_tests:
        return list(sample)
    return sorted(sample, key=lambda test: len(str(test.get("input", ""))), reverse=True)[:max_tests]


# https://huggingface.co/datasets/PrimeIntellect/verifiable-coding-problems
def primeintellect_check_correctness(tests, code, use_tci=False):
    if isinstance(tests, str):
        try:
            tests = ast.literal_eval(tests)
            assert isinstance(tests, dict)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing string: {e}")
            return False, {"all_passed": False, "error": str(e)}

    assert len(tests) >= 1, "PrimeIntellect needs at least one test case"
    # Convert the tests to the format expected by the taco_run_test function
    inputs = [t["input"] for t in tests]
    outputs = [t["output"] for t in tests]
    fn_name = tests[0].get("fn_name", None)
    tests_formatted = {
        "inputs": inputs,
        "outputs": outputs,
    }
    if fn_name:
        tests_formatted["fn_name"] = fn_name

    if use_tci:
        codetool = TogetherCodeTool()
        return codetool_check_correctness(tests_formatted, code, codetool, is_taco_format=True)

    return check_correctness(tests_formatted, code, taco_run_test)


def _temp_run(sample, generation, debug, result, metadata_list, timeout):
    res, metadata = lcb_run_test(sample, test=generation, debug=debug, timeout=timeout)
    result.append(res)
    metadata_list.append(metadata)


def _temp_run_single(sample, generation, debug, timeout, output_queue):
    try:
        res, metadata = lcb_run_test(sample, test=generation, debug=debug, timeout=timeout)
        if res is None:
            res = [-4]
            metadata = {"error_code": -4, "error_message": "No result returned"}
    except BaseException as exc:
        res = [-4]
        metadata = {"error_code": -4, "error_message": f"Error during testing: {exc!r}"}
    output_queue.put((res, metadata))


def _run_lcb_tests_parallel(samples: list[dict[str, Any]], generation, timeout: int, debug: bool, testcase_max_workers: int) -> tuple[list[Any], list[dict[str, Any]]]:
    testcase_max_workers = max(1, int(testcase_max_workers))
    results: list[Any] = [-4 for _ in samples]
    metadata_by_test: list[dict[str, Any]] = [{"error_code": -4, "error_message": "No result returned"} for _ in samples]

    for chunk_start in range(0, len(samples), testcase_max_workers):
        chunk = list(enumerate(samples[chunk_start : chunk_start + testcase_max_workers], start=chunk_start))
        processes = []
        for idx, test_case in chunk:
            output_queue = multiprocessing.Queue(maxsize=1)
            p = multiprocessing.Process(
                target=_temp_run_single,
                args=(postprocess_lcb_sample([test_case]), generation, debug, timeout, output_queue),
            )
            processes.append((idx, p, output_queue))
            p.start()

        deadline = time.monotonic() + timeout + 6
        for _, p, _ in processes:
            p.join(timeout=max(0.0, deadline - time.monotonic()))

        for idx, p, output_queue in processes:
            if p.is_alive():
                p.kill()
                p.join()
                results[idx] = -3
                metadata_by_test[idx] = {"error_code": -3, "error_message": "Time Limit Exceeded"}
            else:
                try:
                    res, metadata = output_queue.get(timeout=1)
                except Empty:
                    res, metadata = [-4], {"error_code": -4, "error_message": "No result returned"}
                results[idx] = res[0] if isinstance(res, list) and res else res
                metadata_by_test[idx] = metadata or {}
            output_queue.close()

    return results, metadata_by_test


def lcb_check_correctness_v2(sample, generation, timeout=6, debug=False, max_tests=15, testcase_max_workers=1):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    assert len(sample) >= 1, "Sample must contain at least one test case"
    original_num_tests = len(sample)
    sample = _select_longest_input_tests(sample, max_tests)
    selected_num_tests = len(sample)
    testcase_max_workers = max(1, int(testcase_max_workers))

    if testcase_max_workers > 1 and selected_num_tests > 1:
        result = _run_lcb_tests_parallel(sample, generation, timeout, debug, testcase_max_workers)
        result_values, metadata_by_test = result
        processed_sample = postprocess_lcb_sample(sample)
        in_outs = json.loads(processed_sample["input_output"])
        detailed_results = {
            "all_passed": all(r == True for r in result_values),
            "test_results": [],
            "total_tests": len(result_values),
            "passed_tests": sum(1 for r in result_values if r == True),
            "original_tests": original_num_tests,
            "selected_tests": selected_num_tests,
            "max_tests": max_tests or 0,
            "timeout": timeout,
            "testcase_max_workers": testcase_max_workers,
        }
        detailed_results["test_results"] = [
            {
                "input": inp,
                "expected": out,
                "passed": res == True,
                "error": metadata.get("error", None),
                "error_message": metadata.get("error_message", None),
                "output": metadata.get("output", None),
            }
            for inp, out, res, metadata in zip(in_outs["inputs"], in_outs["outputs"], result_values, metadata_by_test, strict=False)
        ]
        return detailed_results["all_passed"], detailed_results

    sample = postprocess_lcb_sample(sample)

    manager = multiprocessing.Manager()
    result = manager.list()
    metadata_list = manager.list()

    p = multiprocessing.Process(
        target=_temp_run,
        args=(sample, generation, debug, result, metadata_list, timeout),
    )
    p.start()
    p.join(timeout=(timeout + 1) * len(json.loads(sample["input_output"])["inputs"]) + 5)

    detailed_results = {"all_passed": False, "test_results": [], "total_tests": 0, "passed_tests": 0}

    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result.extend([[-1 for i in range(len(in_outs["inputs"]))]])
        detailed_results["total_tests"] = len(in_outs["inputs"])
        detailed_results["original_tests"] = original_num_tests
        detailed_results["selected_tests"] = selected_num_tests
        detailed_results["max_tests"] = max_tests or 0
        detailed_results["test_results"] = [{"input": inp, "expected": out, "passed": False, "error": "global timeout"} for inp, out in zip(in_outs["inputs"], in_outs["outputs"], strict=False)]
        if debug:
            print("global timeout")
        return False, detailed_results

    if not result:
        return False, detailed_results

    # Create detailed test results
    in_outs = json.loads(sample["input_output"])
    detailed_results["total_tests"] = len(result[0])
    detailed_results["original_tests"] = original_num_tests
    detailed_results["selected_tests"] = selected_num_tests
    detailed_results["max_tests"] = max_tests or 0
    detailed_results["timeout"] = timeout
    detailed_results["testcase_max_workers"] = testcase_max_workers
    detailed_results["test_results"] = [{"input": inp, "expected": out, "passed": res == True, "error": metadata_list[0].get("error", None), "error_message": metadata_list[0].get("error_message", None), "output": metadata_list[0].get("output", None)} for inp, out, res in zip(in_outs["inputs"], in_outs["outputs"], result[0], strict=False)]
    detailed_results["passed_tests"] = sum(1 for r in result[0] if r == True)
    detailed_results["all_passed"] = all(r == True for r in result[0])

    return all(x == True for x in result[0]), detailed_results


def leetcode_check_correctness(tests: dict[str, str], code: str) -> tuple[bool, dict[str, Any]]:
    """
    Check if generated code passes all LeetCode test cases.

    Args:
         tests: Dict of test cases with "functional" key containing test code
         code: Generated code to test
         timeout: Maximum execution time in seconds before killing process
         runtime_debug: Whether to print debug info during test execution

    Returns:
         tuple: (bool, dict) where:
           - bool: True if all tests pass, False otherwise
           - dict: Detailed test results
    """
    succ, output = lc_code_exec(code + "\n" + tests["functional"])
    detailed_results = {"all_passed": succ, "output": output, "test_results": [{"passed": succ, "output": output}]}

    if not succ:
        print(f"Error in code execution: {output}")
    return succ, detailed_results


def kodcode_check_correctness(test: str, code: str, timeout_per_test: int = 5) -> tuple[bool, dict[str, Any]]:
    """
    Check if generated code passes all Kodcode test cases.

    Args:
        test: String of the test file content
        code: Generated code to test
        timeout: Maximum execution time in seconds before killing process
        runtime_debug: Whether to print debug info during test execution

    Returns:
        tuple: (bool, dict) where:
            - bool: True if all tests pass, False otherwise
            - dict: Detailed test results
    """
    # Count the number of test functions in the test file
    num_tests = test.count("def test")

    # Remove 'if __name__ == "__main__":' block if present
    code = clean_code_main_block(code)

    succ, output = kod_code_exec(code, test, timeout_per_test * num_tests)
    detailed_results = {"all_passed": succ, "output": output, "total_tests": num_tests, "test_results": [{"passed": succ, "output": output}]}

    if not succ:
        print(f"Error in code execution: {output}")
    return succ, detailed_results


def humanevalplus_check_correctness(test: str, code: str, timeout_per_test: int = 1) -> tuple[bool, dict[str, Any]]:
    """
    Check if generated code passes all HumanEvalPlus test cases.

    Args:
        test: String of the test file content
        code: Generated code to test
        timeout: Maximum execution time in seconds before killing process
        runtime_debug: Whether to print debug info during test execution

    Returns:
        tuple: (bool, dict) where:
            - bool: True if all tests pass, False otherwise
            - dict: Detailed test results
    """
    code = clean_code_main_block(code)

    num_test_cases = get_num_test_cases(test)
    succ, output = humanevalplus_run_test(code, test, timeout_per_test * num_test_cases)

    detailed_results = {"all_passed": succ, "output": output, "total_tests": num_test_cases, "test_results": [{"passed": succ, "output": output}]}

    if not succ:
        print(f"Error in code execution: {output}")
    return succ, detailed_results


def taco_to_lcb_format(tests):
    """
    Given a dictionary with keys "inputs" and "outputs", returns a list of test cases.
    Each test case is a dictionary with keys "input" and "output". If the lists are unequal,
    missing entries are filled by reusing the first element of the shorter list.

    Args:
        data (dict): A dictionary with keys "inputs" and "outputs", each mapped to a list of strings.

    Returns:
        list of dict: A list where each element is a dict with keys "input" and "output".
    """
    inputs = tests.get("inputs", [])
    outputs = tests.get("outputs", [])

    # Determine the number of test cases to create.
    n = max(len(inputs), len(outputs))

    test_cases = []
    for i in range(n):
        # Use the first element as a fallback if the list is shorter than n.
        inp = inputs[i] if i < len(inputs) else (inputs[0] if inputs else "")
        out = outputs[i] if i < len(outputs) else (outputs[0] if outputs else "")
        out = out[0] if isinstance(out, list) else out
        test_case: dict[str, Any] = {"input": inp, "output": out, "metadata": {}}
        if "fn_name" in tests:
            test_case["testtype"] = "functional"
            test_case["metadata"]["func_name"] = tests["fn_name"]
        test_cases.append(test_case)

    return test_cases


def codetool_check_correctness(tests: Any, code: str, codetool: CodeTool, is_taco_format=True, timeout=30) -> tuple[bool, dict[str, Any]]:
    from rllm.tools.utils import call_based_test_code_wrapper, stdin_test_code_wrapper

    fn_name = None
    call_based = False

    if isinstance(tests, dict) and "fn_name" in tests:
        call_based = True
        fn_name = tests.get("fn_name", None)

    new_tests = taco_to_lcb_format(tests) if is_taco_format and not fn_name else tests

    if call_based:
        test_wrapped_code = call_based_test_code_wrapper(code, new_tests)
    else:
        test_wrapped_code = stdin_test_code_wrapper(code, new_tests)

    tool_response = codetool(code=test_wrapped_code, timeout=timeout)

    detailed_results = {"all_passed": not tool_response.error, "output": tool_response.output, "error": tool_response.error, "test_results": []}

    # Try to extract individual test results if possible
    if isinstance(new_tests, list):
        detailed_results["total_tests"] = len(new_tests)
        detailed_results["test_results"] = [{"input": test.get("input", ""), "expected": test.get("output", ""), "passed": not tool_response.error} for test in new_tests]

    if tool_response.error:
        print(f"Error in code execution: {tool_response.error}")
        return False, detailed_results
    return True, detailed_results


class RewardCodeFn:
    """
    Reward function for evaluating code dataset answers.

    This class implements the RewardFunction protocol to process the input and determine
    the reward based on the correctness of the unit tests provided
    """

    def __init__(self, config: RewardConfig):
        self.config = config

    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        """
        Calculate the reward for a code task based on the agent's action.

        Args:
            task_info: Dictionary containing problem, data_source, problem_type, and ground_truth
            action: The agent's response/solution (code)

        Returns:
            RewardOutput: The calculated reward with correctness information
        """
        # total_start_time = time.time()

        model_response = action
        dataset_name = task_info.get("data_source", "")
        tests = task_info.get("ground_truth", None)

        if tests is None:
            print("No tests found in task_info")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False, metadata={"error": "No tests found in task_info"})

        model_code = extract_code_from_model(model_response)
        if model_code is None:
            # print("No code found in model response")
            return RewardOutput(reward=self.config.format_error_reward, is_correct=False, metadata={"error": "No code found in model response"})

        if self.config.use_together_code_interpreter:
            codetool = TogetherCodeTool()

        # Tests: List[Dictionary] - Codeforces, LiveCodeBench
        # Tests: Dictionary[Lists] - CodeContests, Taco/Apps
        is_correct = False
        test_details: dict[str, Any] = {}

        if dataset_name in ["taco", "apps", "code_contests"]:
            if self.config.use_together_code_interpreter:
                is_correct, test_details = codetool_check_correctness(tests, model_code, codetool, is_taco_format=True)
            else:
                tests = taco_to_lcb_format(tests)
                is_correct, test_details = lcb_check_correctness_v2(
                    tests,
                    model_code,
                    timeout=self.config.code_timeout,
                    debug=False,
                    max_tests=self.config.code_max_tests,
                    testcase_max_workers=self.config.code_testcase_max_workers,
                )
                # test_fn = taco_run_test
                # is_correct, test_details = check_correctness(tests, model_code, test_fn)
        elif dataset_name == "leetcode":
            is_correct, test_details = leetcode_check_correctness(tests, model_code)
        elif dataset_name in ["livecodebench", "codeforces", "primeintellect"]:
            is_correct, test_details = lcb_check_correctness_v2(
                tests,
                model_code,
                timeout=self.config.code_timeout,
                debug=False,
                max_tests=self.config.code_max_tests,
                testcase_max_workers=self.config.code_testcase_max_workers,
            )
        elif dataset_name == "kodcode":
            is_correct, test_details = kodcode_check_correctness(tests, model_code)
        elif dataset_name == "humanevalplus":
            is_correct, test_details = humanevalplus_check_correctness(tests, model_code)
        else:
            raise NotImplementedError(f"Dataset {dataset_name} not implemented")

        # total_time = time.time() - total_start_time
        # print(f"Total reward function execution time: {total_time:.2f} seconds")

        if is_correct:
            return RewardOutput(reward=self.config.correct_reward, is_correct=True, metadata=test_details)
        else:
            return RewardOutput(reward=self.config.incorrect_reward, is_correct=False, metadata=test_details)


def rllm_reward_fn_code(data_source: str, llm_solution: str, ground_truth: dict, **kwargs):
    """Evaluate code solutions against ground truth answers

        This function creates a reward function to evaluate code solutions by pass the test_case from groun_truth. It can optionally use a language model
        for more sophisticated answer validation.

        Args:
            data_source: The source/dataset the problem comes from
            llm_solution: The solution string provided by the language model to evaluate
            ground_truth: some tests for this llm_solution
            enable_llm: Whether to enable language model validation for complex cases (default: False)

        Returns:
            tuple: (bool, dict) where:
                - bool: True if the solution passes all the test_case, False otherwise
                - dict: Detailed test results with test cases and pass/fail status

        Example:
                model_response = '''
    import sys
    from itertools import permutations
    def main():
        n,m=map(int, input().split())
        a=sum(list(map(int, input().split())))
        if a+(n-1)*10<=m:
            print(5)
        else:
            print(5)
    if __name__ == "__main__":
        main()
    '''

        print(f"test the code_forces")
        # tests = [ { "input": "3 30\n2 2 1", "output": "5" }, { "input": "3 10\n3 2 1", "output": "5" } ]
        metadata = {
             "tests": tests,
        }
        True, {"all_passed": True, "test_results": [...]}
    """
    default_config = RewardConfig()
    reward_config = RewardConfig(
        code_max_tests=int(kwargs.get("max_tests") or default_config.code_max_tests),
        code_timeout=int(kwargs.get("timeout") or default_config.code_timeout),
        code_testcase_max_workers=int(kwargs.get("testcase_max_workers") or default_config.code_testcase_max_workers),
    )
    reward_fn = RewardCodeFn(reward_config)

    # Convert to new format
    task_info = {"problem": None, "problem_type": RewardType.CODE, "data_source": data_source, "ground_truth": ground_truth}

    reward_response = reward_fn(task_info, llm_solution)
    return reward_response
