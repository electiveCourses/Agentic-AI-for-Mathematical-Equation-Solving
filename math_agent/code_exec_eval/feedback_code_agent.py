import platform
import re
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_from_disk
from smolagents import (
    CodeAgent,
    InferenceClientModel,
    TransformersModel,
)

from math_agent.utils.commons import run_full_evaluation, save_evaluation_results


class MathCodeAgentWithEvaluation:
    """
    An enhanced agent that solves mathematical problems by generating and executing Python code
    with self-evaluation and correction capabilities.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        use_local: bool = True,
        use_mlx: Optional[bool] = None,
        max_retries: int = 3,
    ):
        """
        Initialize the Math Code Agent with evaluation capabilities.

        Args:
            model_name: The name of the model to use for code generation
            use_local: Whether to use local model inference (True) or remote inference (False)
            use_mlx: Whether to use MLX for Apple Silicon (None for auto-detect, True to force, False to disable)
            max_retries: Maximum number of retry attempts for code generation
        """
        # Set default model based on platform and available resources
        if model_name is None:
            model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"

        self.max_retries = max_retries

        # Auto-detect MLX usage for Apple Silicon
        if use_mlx is None:
            use_mlx = platform.system() == "Darwin" and platform.processor() == "arm"

        # Initialize the model based on use_local flag
        if use_local:
            if use_mlx:
                try:
                    from smolagents import MLXModel

                    model = MLXModel(
                        model_id=model_name,
                        max_tokens=10000,
                    )
                    print("Using MLX for Apple Silicon")
                except ImportError:
                    print("MLX not available, falling back to TransformersModel")
                    model = TransformersModel(
                        model_id=model_name,
                        temperature=0.1,
                        torch_dtype="auto",
                        device_map="auto",
                    )
            else:
                model = TransformersModel(
                    model_id=model_name,
                    temperature=0.1,
                    torch_dtype="auto",
                    device_map="auto",
                )
                print(f"Using TransformersModel with {model_name}")
        else:
            model = InferenceClientModel(model_name)
            print(f"Using remote inference with {model_name}")

        # Initialize the smolagents CodeAgent
        self.agent = CodeAgent(
            tools=[],
            model=model,
            verbosity_level=0,
            max_steps=5,
            additional_authorized_imports=[
                "numpy",
                "sympy",
                "scipy",
                "math",
                "statistics",
                "itertools",
                "collections",
                "random",
                "re",
                "datetime",
                "time",
                "unicodedata",
                "stat",
                "queue",
                "decimal",
                "fractions",
                "operator",
                "functools",
            ],
        )

        # System prompts
        self.math_system_prompt = """You are a mathematical problem solver. When given a math problem, you must:

1. Write Python code to solve the problem
2. Use the final_answer() function to return your result
3. Always end your code with final_answer("your_answer_here")

IMPORTANT: You must ALWAYS use this exact format:
```py
# Your calculation code here
final_answer("your_result")
```

Your answer can be in various formats:
- Single numbers: "5", "3.14159", "-2"
- Multiple numbers: "2, 3, 5", "0.5, 0.2, 0.1, -5.0"
- Fractions: "1/2", "3/4", "7/8"
- Symbolic expressions: "x^2 + 3*x + 2", "sqrt(2)", "2*pi"
- Lists/sets: "[1, 2, 3]", "{2, 4, 6}"
- Complex expressions: "2 + 3*I", "sqrt(5) + 2"
- Option for multiple choise answers (b)

Available libraries:
- numpy: For numerical computations and arrays
- sympy: For symbolic mathematics and equation solving
- scipy: For scientific computing
- All standard library modules (math, statistics, fractions, decimal, etc.)

CRITICAL:
- Always end with final_answer("your_result")
- Convert your result to string format: str(result)
- For multiple values, join them with commas: ', '.join(map(str, values))
- Keep the exact format expected by the problem (fractions, decimals, symbolic, etc.)"""

        self.evaluation_prompt = """You are a code evaluator for mathematical problem solving.
Analyze the following code and its execution result to determine if it correctly solves the given problem.

PRIMARY EVALUATION GOAL:
The main reason for evaluation is to check if the FORMAT of the output corresponds to what is mentioned in the task.
Focus primarily on whether the result format matches the expected output format specified in the problem.

EVALUATION GUIDELINES - BE VERY LENIENT:
- If the code produces any reasonable mathematical result, it should generally PASS
- If the code uses proper mathematical libraries (sympy, numpy, etc.) and has sound logic, it should PASS
- If the final_answer() function is called with a result, it should PASS unless there are critical errors
- Minor formatting differences, variable naming, or approach variations are acceptable
- Focus on whether the mathematical approach makes sense, not perfection
- MOST IMPORTANT: Check if the output format matches what the problem is asking for

Consider these criteria in order of importance:
1. FORMAT MATCH: Does the result format correspond to what the task specified? (MOST IMPORTANT)
2. COMPLETENESS: Does the code attempt to address the problem?
3. CORRECTNESS: Does the code produce a reasonable mathematical result?
4. LOGIC: Is the mathematical approach generally sound?
5. ERRORS: Are there any critical runtime errors?

DEFAULT TO PASS unless there are obvious critical errors or the output format completely doesn't match the task requirements.

EXAMPLES OF FIXES SUGGESTIONS:
1. answer is in decimal format (for example 0.1), but expected as a fraction from the task (1/10)
2. answer is in numeric format, but in the task it is multiple choise with given options -> return one of the options 

Provide feedback in this format:
EVALUATION: [PASS/FAIL]
CONFIDENCE: [0-100]
ISSUES: [List specific problems if any]
SUGGESTIONS: [Specific improvements for the code]
"""

    def evaluate_code_and_result(
        self,
        problem: str,
        code: str,
        result: str,
    ) -> Dict[str, Any]:
        """
        Evaluate the generated code and its execution result using self-evaluation.
        The expected_answer parameter is kept for compatibility but not used in evaluation.

        Args:
            problem: The original math problem
            code: The generated Python code
            result: The execution result

        Returns:
            Dictionary containing evaluation results
        """
        evaluation: Dict[str, Any] = {
            "passed": False,
            "confidence": 0,
            "issues": [],
            "suggestions": [],
            "needs_retry": False,
        }

        # Check for critical runtime errors (be more lenient)
        critical_errors = ["SyntaxError", "NameError", "TypeError", "ZeroDivisionError", "ImportError"]
        if any(error in result for error in critical_errors):
            evaluation["issues"].append("Critical runtime error detected")
            evaluation["suggestions"].append("Fix the critical runtime error in the code")
            evaluation["needs_retry"] = True
            return evaluation
        
        # For minor errors or warnings, just note them but don't fail
        if "Error" in result or "error" in result.lower():
            evaluation["issues"].append("Minor error or warning detected")
            evaluation["suggestions"].append("Consider addressing any minor issues")
            # Don't automatically retry for minor errors

        # Check for missing final_answer call (more lenient)
        if "final_answer(" not in code and "final_answer" not in code.lower():
            evaluation["issues"].append("Missing final_answer() function call")
            evaluation["suggestions"].append(
                "Add final_answer() at the end of your code"
            )
            evaluation["needs_retry"] = True
            return evaluation
        
        # If final_answer is mentioned but not called properly, just warn
        if "final_answer" in code.lower() and "final_answer(" not in code:
            evaluation["issues"].append("final_answer mentioned but not called properly")
            evaluation["suggestions"].append("Use proper final_answer() function call syntax")
            # Don't automatically retry for this

        # Check if result is empty or too short
        if not result or len(result.strip()) < 1:
            evaluation["issues"].append("Empty or insufficient result")
            evaluation["suggestions"].append(
                "Ensure your code produces a meaningful output"
            )
            evaluation["needs_retry"] = True
            return evaluation

        # For very short results, be more lenient
        if len(result.strip()) < 3 and not result.strip().isdigit():
            evaluation["issues"].append("Very short result - may need more detail")
            evaluation["suggestions"].append(
                "Consider providing more complete output"
            )
            # Don't automatically retry for short results
            evaluation["needs_retry"] = False

        # LLM-based self-evaluation (independent of expected answer)
        llm_eval = self._llm_evaluate_code(problem, code, result)
        
        # Merge the evaluation results properly
        evaluation["passed"] = llm_eval["passed"]
        evaluation["confidence"] = llm_eval["confidence"]
        evaluation["needs_retry"] = llm_eval["needs_retry"]
        
        # Merge issues and suggestions lists
        if "issues" in llm_eval and isinstance(llm_eval["issues"], list):
            issues_list = evaluation["issues"]
            if isinstance(issues_list, list):
                issues_list.extend(llm_eval["issues"])
        if "suggestions" in llm_eval and isinstance(llm_eval["suggestions"], list):
            suggestions_list = evaluation["suggestions"]
            if isinstance(suggestions_list, list):
                suggestions_list.extend(llm_eval["suggestions"])

        return evaluation

    def _llm_evaluate_code(
        self, problem: str, code: str, result: str
    ) -> Dict[str, Any]:
        """Use LLM to evaluate code quality and correctness."""
        eval_prompt = f"""{self.evaluation_prompt}

PROBLEM: {problem}

GENERATED CODE:
```python
{code}
```

EXECUTION RESULT: {result}

Please evaluate this solution:"""

        try:
            # Use the underlying model directly for evaluation (no code execution)
            # Format messages correctly for smolagents model

            # specific mssages format for MLX model
            if not (isinstance(self.agent.model, TransformersModel) or isinstance(self.agent.model, InferenceClientModel)):
                messages = [    
                    {"role": "user", "content": [{"type": "text", "text": eval_prompt}]}
                ]
            else:
                # костыль для TransformersModel, который не поддерживает нормальный формат сообщений
                from pydantic import BaseModel
                class Message(BaseModel):
                    role: str
                    content: list[dict[str, str]]
                messages = [Message(role="user", content=[{"type": "text", "text": eval_prompt}])]

            evaluation_response = self.agent.model(messages)
            # Parse LLM evaluation
            eval_text = str(evaluation_response)

            passed = "EVALUATION: PASS" in eval_text

            # Extract confidence
            confidence_match = re.search(r"CONFIDENCE:\s*(\d+)", eval_text)
            confidence = int(confidence_match.group(1)) if confidence_match else 50
            
            # Override passed if confidence > 60
            if confidence >= 60:
                passed = True

            # Extract issues
            issues_match = re.search(
                r"ISSUES:\s*(.*?)(?=SUGGESTIONS:|$)", eval_text, re.DOTALL
            )
            issues = []
            if issues_match:
                issues_text = issues_match.group(1).strip()
                # Split by lines and clean up numbered lists
                for line in issues_text.split("\n"):
                    line = line.strip()
                    if line:
                        # Remove leading numbers and dots (e.g., "1. ", "2. ", etc.)
                        cleaned_line = re.sub(r"^\d+\.\s*", "", line)
                        if cleaned_line:
                            issues.append(cleaned_line)

            # Extract suggestions
            suggestions_match = re.search(
                r"SUGGESTIONS:\s*(.*?)$", eval_text, re.DOTALL
            )
            suggestions = []
            if suggestions_match:
                suggestions_text = suggestions_match.group(1).strip()
                # Split by lines and clean up numbered lists
                for line in suggestions_text.split("\n"):
                    line = line.strip()
                    if line:
                        # Remove leading numbers and dots (e.g., "1. ", "2. ", etc.)
                        cleaned_line = re.sub(r"^\d+\.\s*", "", line)
                        if cleaned_line:
                            suggestions.append(cleaned_line)

            return {
                "passed": passed,
                "confidence": confidence,
                "issues": issues,
                "suggestions": suggestions,
                "needs_retry": not passed or confidence < 60,
            }

        except Exception as e:
            return {
                "passed": False,
                "confidence": 0,
                "issues": [f"Evaluation error: {str(e)}"],
                "suggestions": ["Try a different approach"],
                "needs_retry": True,
            }

    def _normalize_answer(self, answer: str) -> str:
        """Normalize answer for comparison."""
        if not answer:
            return ""
        # Remove whitespace and convert to string
        answer = str(answer).strip()
        # Remove common formatting
        answer = answer.replace(" ", "").replace(",", "")
        return answer.lower()


    def solve_with_evaluation(
        self, problem: str, verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Solve a math problem with evaluation and retry mechanism.

        Args:
            problem: The math problem to solve
            verbose: Whether to print detailed information

        Returns:
            Dictionary containing solution details and evaluation results
        """
        attempt_history: List[Dict[str, Any]] = []
        temp_solutions: List[Dict[str, Any]] = []  # Track valid parsed outputs

        for attempt in range(self.max_retries):
            if verbose:
                print(f"\n--- Attempt {attempt + 1} ---")

            # Generate code and get result
            code, result = self._generate_and_execute(problem, attempt_history)

            if verbose:
                print(f"Generated code:\n{code}")
                print(f"Result: {result}")

                        
            temp_solution = {
                    "attempt": attempt + 1,
                    "code": code,
                    "result": result,
                    "timestamp": attempt,
                }
            temp_solutions.append(temp_solution)
            if verbose:
                print(f"Valid temp solution captured: {result}")

            # Evaluate the result
            evaluation = self.evaluate_code_and_result(problem, code, result)

            attempt_info = {
                "attempt": attempt + 1,
                "code": code,
                "result": result,
                "evaluation": evaluation,
            }
            attempt_history.append(attempt_info)

            if verbose:
                print(f"Evaluation: {'PASS' if evaluation['passed'] else 'FAIL'}")
                print(f"Confidence: {evaluation['confidence']}")
                if evaluation["issues"]:
                    print(f"Issues: {', '.join(evaluation['issues'])}")

            # If evaluation passes, return success
            if evaluation["passed"] and not evaluation["needs_retry"]:
                return {
                    "success": True,
                    "final_answer": result,
                    "attempts": attempt + 1,
                    "history": attempt_history,
                    "code": code,
                    "temp_solutions": temp_solutions,
                }

            # If this was the last attempt, break
            if attempt == self.max_retries - 1:
                break

        # All attempts failed evaluation, but check for temp solutions
        if temp_solutions:
            # Return the last valid temp solution
            last_temp_solution = temp_solutions[-1]
            if verbose:
                print(f"\nNo attempts passed evaluation, returning last temp solution from attempt {last_temp_solution['attempt']}")
            
            return {
                "success": True, # Evaluation failed but we have a temp solution
                "final_answer": last_temp_solution["result"],
                "attempts": self.max_retries,
                "history": attempt_history,
                "code": last_temp_solution["code"],
                "temp_solutions": temp_solutions,
                "fallback_used": True,
                "fallback_reason": "No attempts passed evaluation, using last valid temp solution",
            }
        else:
            # No valid temp solutions found at all
            return {
                "success": False,
                "final_answer": None,
                "attempts": self.max_retries,
                "history": attempt_history,
                "error": "All retry attempts failed and no valid temp solutions found",
                "temp_solutions": temp_solutions,
            }

    def _generate_and_execute(
        self, problem: str, previous_attempts: List[Dict]
    ) -> Tuple[str, str]:
        """Generate code and execute it, incorporating feedback from previous attempts."""

        # Build feedback from previous attempts
        feedback = ""
        if previous_attempts:
            feedback = "\n\nPREVIOUS ATTEMPTS AND FEEDBACK:\n"
            for i, attempt in enumerate(previous_attempts):
                feedback += f"Attempt {i + 1}:\n"
                feedback += f"Code: {attempt['code'][:200]}...\n"
                feedback += f"Result: {attempt['result']}\n"
                
                # Format issues as numbered list
                if attempt['evaluation']['issues']:
                    feedback += "Issues:\n"
                    for j, issue in enumerate(attempt['evaluation']['issues'], 1):
                        feedback += f"  {j}. {issue}\n"
                
                # Format suggestions as numbered list
                if attempt['evaluation']['suggestions']:
                    feedback += "Suggestions:\n"
                    for j, suggestion in enumerate(attempt['evaluation']['suggestions'], 1):
                        feedback += f"  {j}. {suggestion}\n"
                
                feedback += "\n"
            feedback += "Please fix these issues in your new attempt.\n"

        prompt = f"""{self.math_system_prompt}

Problem: {problem}

{feedback}

Please write Python code to solve this problem and execute it using the execute_python_code tool.
Make sure to address any previous issues and print the final answer in the exact format expected."""

        try:
            response = self.agent.run(prompt)
            
            # Extract code and result from smolagents memory structure
            code = ""
            result = ""
            
            # Get the latest step from memory
            steps = self.agent.memory.get_full_steps()
            if steps:
                latest_step = steps[-1]
                
                # Extract code from tool_calls
                if 'tool_calls' in latest_step and latest_step['tool_calls']:
                    tool_call = latest_step['tool_calls'][0]
                    if 'function' in tool_call and 'arguments' in tool_call['function']:
                        code = tool_call['function']['arguments']
                
                # Extract result from action_output - handle complex numbers and various formats
                if 'action_output' in latest_step:
                    action_output = latest_step['action_output']
                    
                    # Handle different output formats
                    if isinstance(action_output, dict) and 'output' in action_output:
                        result = str(action_output['output']).strip()
                    elif isinstance(action_output, str):
                        result = action_output.strip()
                    else:
                        result = str(action_output).strip()
            
            # Enhanced fallback: try multiple extraction methods
            if not code or not result:
                response_str = str(response)
                
                # Try to extract code from different patterns
                if not code:
                    code_patterns = [
                        r"```py\n(.*?)\n```",
                        r"```python\n(.*?)\n```",
                        r"<execute_python_code>(.*?)</execute_python_code>",
                    ]
                    
                    for pattern in code_patterns:
                        code_match = re.search(pattern, response_str, re.DOTALL)
                        if code_match:
                            code = code_match.group(1).strip()
                            break
                
                # Try to extract result from response
                if not result:
                    if isinstance(response, dict):
                        if "output" in response:
                            result = str(response["output"]).strip()
                        elif "result" in response:
                            result = str(response["result"]).strip()
                        else:
                            result = str(response).strip()
                    else:
                        result = str(response).strip()
                        
                    # Clean up the result to handle complex numbers and special formats
                    result = self._clean_result_output(result)

            return code, result
        except Exception as e:
            return "", f"Error: {str(e)}"

    def _clean_result_output(self, result: str) -> str:
        """Clean and format the result output to handle various formats."""
        if not result:
            return ""
        
        # Remove common prefixes and suffixes
        result = result.strip()
        
        # Remove "Output:" prefix if present
        if result.startswith("Output:"):
            result = result[7:].strip()
        
        # Remove "Result:" prefix if present
        if result.startswith("Result:"):
            result = result[7:].strip()
        
        # Handle complex numbers and special mathematical notation
        # Keep the result as is for most cases - be lenient
        return result

    def solve(self, problem: str) -> str:
        """Backward compatibility method."""
        result = self.solve_with_evaluation(problem)
        final_answer = result.get("final_answer", "Error: Failed to solve")
        return str(final_answer) if final_answer is not None else "Error: Failed to solve"


def verify_answer(answer: Optional[str], expected_answer: str) -> bool:
    """Verify if the answer is correct."""
    def try_convert_to_number(value: str) -> Any:
        """Try to convert a string to a number (int or float)."""
        try:
            # Try integer first
            if '.' not in value:
                return int(value)
            else:
                return float(value)
        except ValueError:
            return value
    
    # Handle None answer
    if answer is None:
        return False
    
    # Convert both to numbers if possible
    answer_num = try_convert_to_number(str(answer).strip())
    expected_num = try_convert_to_number(str(expected_answer).strip())
    
    # If both are numbers, compare numerically
    if isinstance(answer_num, (int, float)) and isinstance(expected_num, (int, float)):
        return abs(float(answer_num) - float(expected_num)) < 1e-6
    
    # If both are strings, compare as strings
    elif isinstance(answer_num, str) and isinstance(expected_num, str):
        return answer_num == expected_num
    
    # Mixed types - convert both to strings and compare
    else:
        return str(answer_num) == str(expected_num)

def solve_math_problem_with_evaluation(
    problem: str,
    expected_answer: Optional[str] = None,
    agent: Optional[MathCodeAgentWithEvaluation] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Solve a math problem using the enhanced agent with evaluation.
    Compatible with commons.py evaluation functions.

    Args:
        problem (str): The math problem to solve
        expected_answer (str, optional): The expected answer for evaluation
        agent: The MathCodeAgentWithEvaluation instance
        verbose (bool): Whether to print detailed information

    Returns:
        dict: Contains the parsed result and evaluation metrics
    """
    try:
        # Check if agent is provided
        if agent is None:
            raise ValueError("Agent instance is required")
        
        # Solve the problem with evaluation
        solution_result = agent.solve_with_evaluation(problem, verbose=verbose)

        success = solution_result["success"]
        answer = solution_result.get("final_answer")

        result = {
            "answer": answer,
            "success": success,
            "raw_output": answer,
            "attempts": solution_result.get("attempts", 1),
            "evaluation_history": solution_result.get("history", []),
        }

        # Evaluate if expected answer is provided
        if expected_answer is not None:
            result["correct"] = verify_answer(answer, expected_answer)
            result["expected_answer"] = expected_answer

        return result

    except Exception as e:
        return {
            "answer": None,
            "success": False,
            "error": str(e),
            "raw_output": f"Error: {str(e)}",
            "correct": False,
            "expected_answer": expected_answer,
            "attempts": 1,
        }


def main() -> None:
    """Demonstration of the enhanced agent with evaluation."""
    print("Initializing Enhanced Math Code Agent with Evaluation...")

    # Initialize the enhanced agent
    agent = MathCodeAgentWithEvaluation(use_local=True, max_retries=3, use_mlx=False)

    # Evaluate on a dataset using comprehensive evaluation
    print("\n\n=== Comprehensive Dataset Evaluation ===")

    # Load the processed dataset
    dataset_path = "data/processed/math_qa_dataset"

    if os.path.exists(dataset_path):
        try:
            print("Loading processed dataset...")
            dataset = load_from_disk(dataset_path)

            # Run comprehensive evaluation
            solve_function = solve_math_problem_with_evaluation
            solve_function_args = {
                "agent": agent,
            }
            full_results = run_full_evaluation(
                dataset, solve_function, solve_function_args, verbose=False, problems_per_category=1000
            )

            # Save results
            save_evaluation_results(
                full_results, "agent_comprehensive_evaluation_results.json"
            )

            print("\n=== Evaluation Complete ===")
            print("Full results saved to agent_comprehensive_evaluation_results.json")

        except Exception as e:
            print(f"Comprehensive evaluation error: {e}")
            print("Dataset evaluation failed.")
    else:
        print("Dataset not found. Please ensure the dataset is available at data/processed/math_qa_dataset")

    print("\n=== Enhanced Agent Setup Complete ===")
    print("The enhanced agent is now using local model inference with evaluation!")


if __name__ == "__main__":
    main()
