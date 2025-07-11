import platform
import re
from typing import Any, Dict, List, Optional, Tuple

from smolagents import (
    CodeAgent,
    InferenceClientModel,
    TransformersModel,
)


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
            verbosity_level=1,
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

Consider these criteria:
1. CORRECTNESS: Does the code produce the correct mathematical result?
2. COMPLETENESS: Does the code address all parts of the problem?
3. LOGIC: Is the mathematical approach sound?
4. FORMAT: Is the result in the expected format?
5. ERRORS: Are there any runtime errors or logical mistakes?

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
        expected_answer: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the generated code and its execution result.

        Args:
            problem: The original math problem
            code: The generated Python code
            result: The execution result
            expected_answer: The expected answer if known

        Returns:
            Dictionary containing evaluation results
        """
        evaluation = {
            "passed": False,
            "confidence": 0,
            "issues": [],
            "suggestions": [],
            "needs_retry": False,
        }

        # Check for runtime errors
        if "Error" in result or "error" in result.lower():
            evaluation["issues"].append("Runtime error detected")
            evaluation["suggestions"].append("Fix the runtime error in the code")
            evaluation["needs_retry"] = True
            return evaluation

        # Check for missing final_answer call
        if "final_answer(" not in code:
            evaluation["issues"].append("Missing final_answer() function call")
            evaluation["suggestions"].append(
                "Add final_answer() at the end of your code"
            )
            evaluation["needs_retry"] = True
            return evaluation

        # Check if result is empty or too short
        if not result or len(result.strip()) < 1:
            evaluation["issues"].append("Empty or insufficient result")
            evaluation["suggestions"].append(
                "Ensure your code produces a meaningful output"
            )
            evaluation["needs_retry"] = True
            return evaluation

        # Check against expected answer if provided
        if expected_answer:
            if self._normalize_answer(result) == self._normalize_answer(
                expected_answer
            ):
                evaluation["passed"] = True
                evaluation["confidence"] = 95
            else:
                evaluation["issues"].append(
                    f"Result '{result}' doesn't match expected '{expected_answer}'"
                )
                evaluation["suggestions"].append(
                    "Review your mathematical approach and calculations"
                )
                evaluation["needs_retry"] = True
                return evaluation

        # Use LLM for detailed evaluation if no expected answer
        if not expected_answer:
            llm_eval = self._llm_evaluate_code(problem, code, result)
            evaluation.update(llm_eval)

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
            evaluation_response = self.agent.model.chat(
                [{"role": "user", "content": eval_prompt}]
            )

            # Parse LLM evaluation
            eval_text = str(evaluation_response)

            passed = "EVALUATION: PASS" in eval_text

            # Extract confidence
            confidence_match = re.search(r"CONFIDENCE:\s*(\d+)", eval_text)
            confidence = int(confidence_match.group(1)) if confidence_match else 50

            # Extract issues
            issues_match = re.search(
                r"ISSUES:\s*(.*?)(?=SUGGESTIONS:|$)", eval_text, re.DOTALL
            )
            issues = (
                [i.strip() for i in issues_match.group(1).split("\n") if i.strip()]
                if issues_match
                else []
            )

            # Extract suggestions
            suggestions_match = re.search(
                r"SUGGESTIONS:\s*(.*?)$", eval_text, re.DOTALL
            )
            suggestions = (
                [s.strip() for s in suggestions_match.group(1).split("\n") if s.strip()]
                if suggestions_match
                else []
            )

            return {
                "passed": passed,
                "confidence": confidence,
                "issues": issues,
                "suggestions": suggestions,
                "needs_retry": not passed or confidence < 70,
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
        self, problem: str, expected_answer: Optional[str] = None, verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Solve a math problem with evaluation and retry mechanism.

        Args:
            problem: The math problem to solve
            expected_answer: Expected answer for validation
            verbose: Whether to print detailed information

        Returns:
            Dictionary containing solution details and evaluation results
        """
        attempt_history: List[Dict[str, Any]] = []

        for attempt in range(self.max_retries):
            if verbose:
                print(f"\n--- Attempt {attempt + 1} ---")

            # Generate code and get result
            code, result = self._generate_and_execute(problem, attempt_history)

            if verbose:
                print(f"Generated code:\n{code}")
                print(f"Result: {result}")

            # Evaluate the result
            evaluation = self.evaluate_code_and_result(
                problem, code, result, expected_answer
            )

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
                }

            # If this was the last attempt, break
            if attempt == self.max_retries - 1:
                break

        # All attempts failed
        return {
            "success": False,
            "final_answer": None,
            "attempts": self.max_retries,
            "history": attempt_history,
            "error": "All retry attempts failed",
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
                feedback += f"Issues: {', '.join(attempt['evaluation']['issues'])}\n"
                feedback += f"Suggestions: {', '.join(attempt['evaluation']['suggestions'])}\n\n"
            feedback += "Please fix these issues in your new attempt.\n"

        prompt = f"""{self.math_system_prompt}

Problem: {problem}

{feedback}

Please write Python code to solve this problem and execute it using the execute_python_code tool.
Make sure to address any previous issues and print the final answer in the exact format expected."""

        try:
            response = self.agent.run(prompt)

            # Extract code
            code_match = re.search(r"```python\n(.*?)\n```", str(response), re.DOTALL)
            if code_match:
                code = code_match.group(1)
            else:
                # If no code block found, try to extract from response
                code = str(response)

            # Extract result
            if isinstance(response, dict) and "output" in response:
                result = str(response["output"]).strip()
            else:
                result = str(response).strip()

            return code, result
        except Exception as e:
            return "", f"Error: {str(e)}"

    def solve(self, problem: str) -> str:
        """Backward compatibility method."""
        result = self.solve_with_evaluation(problem)
        return result.get("final_answer", "Error: Failed to solve")


# Enhanced solve function for commons.py compatibility
def solve_math_problem_with_evaluation(
    problem: str,
    expected_answer: Optional[str] = None,
    agent: Optional[MathCodeAgentWithEvaluation] = None,
) -> Dict[str, Any]:
    """
    Solve a math problem using the enhanced agent with evaluation.
    Compatible with commons.py evaluation functions.

    Args:
        problem (str): The math problem to solve
        expected_answer (str, optional): The expected answer for evaluation
        agent: The MathCodeAgentWithEvaluation instance

    Returns:
        dict: Contains the parsed result and evaluation metrics
    """
    try:
        # Solve the problem with evaluation
        solution_result = agent.solve_with_evaluation(
            problem, expected_answer, verbose=False
        )

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
            result["correct"] = (answer == expected_answer) if success else False
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
    agent = MathCodeAgentWithEvaluation(use_local=True, max_retries=3)

    # Test problems
    test_problems = [
        ("Solve -20*b + 128*b + 648 = 0 for b.", "-6"),
        ("What is the square root of 144?", "12"),
        ("Find the prime factors of 84.", "[2, 2, 3, 7]"),
    ]

    for problem, expected in test_problems:
        print(f"\n{'=' * 50}")
        print(f"Problem: {problem}")
        print(f"Expected: {expected}")
        print(f"{'=' * 50}")

        result = agent.solve_with_evaluation(problem, expected, verbose=True)

        print(f"\nFinal Result: {'SUCCESS' if result['success'] else 'FAILED'}")
        print(f"Answer: {result.get('final_answer', 'None')}")
        print(f"Attempts: {result['attempts']}")


if __name__ == "__main__":
    main()
