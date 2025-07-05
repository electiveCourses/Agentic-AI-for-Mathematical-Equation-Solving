import os
import re
import json
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

from smolagents import CodeAgent, TransformersModel
from huggingface_hub import login
import platform


class MathCodeAgent:
    """
    An agent that solves mathematical problems by generating and executing Python code.
    """
    
    def __init__(self, model_name: Optional[str] = None, use_local: bool = True, use_mlx: Optional[bool] = None):
        """
        Initialize the Math Code Agent.
        
        Args:
            model_name: The name of the model to use for code generation
            use_local: Whether to use local model inference (True) or remote inference (False)
            use_mlx: Whether to use MLX for Apple Silicon (None for auto-detect, True to force, False to disable)
        """
        # Set default model based on platform and available resources
        if model_name is None:
            # Use smaller models for local inference by default
            model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
        
        # Auto-detect MLX usage for Apple Silicon
        if use_mlx is None:
            use_mlx = platform.system() == "Darwin" and platform.processor() == "arm"
        
        # Initialize the model based on use_local flag
        if use_local:
            if use_mlx:
                try:
                    # Use MLX for Apple Silicon (faster and more memory efficient)
                    from smolagents import MLXModel
                    model = MLXModel(
                        model_id=model_name,
                        max_tokens=10000,
                    )
                    print("Using MLX for Apple Silicon")
                except ImportError:
                    print("MLX not available, falling back to TransformersModel")
                    # Fall back to TransformersModel
                    model = TransformersModel(
                        model_id=model_name,
                        temperature=0.1,
                        torch_dtype="auto",
                        device_map="auto"
                    )
            else:
                # Use standard TransformersModel for local inference
                model = TransformersModel(
                    model_id=model_name,
                    temperature=0.1,
                    torch_dtype="auto",
                    device_map="auto"
                )
                print(f"Using TransformersModel with {model_name}")
        else:
            # Use remote inference (fallback)
            from smolagents import InferenceClientModel
            model = InferenceClientModel(model_name)
            print(f"Using remote inference with {model_name}")
        
        # Initialize the smolagents CodeAgent
        self.agent = CodeAgent(
            tools=[],  # CodeAgent has built-in code execution, no custom tools needed
            model=model,
            verbosity_level=1,
            additional_authorized_imports=[
                # External math packages
                "numpy",
                "sympy",
                "scipy",
                # Standard library modules
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
            ]
        )
        
        # Add a custom system prompt for math problems
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

    def solve(self, problem: str) -> str:
        """
        Solve a mathematical problem by generating and executing Python code.
        
        Args:
            problem: The mathematical problem to solve
            
        Returns:
            The solution to the problem
        """
        # Create a detailed prompt for the agent
        prompt = f"""{self.math_system_prompt}

Problem: {problem}

Please write Python code to solve this problem and execute it using the execute_python_code tool. 
Make sure to print the final answer in the exact format expected."""
        
        try:
            # Run the agent
            result = self.agent.run(prompt)
            
            # Extract the answer from the agent's response
            if isinstance(result, dict) and 'output' in result:
                return str(result['output']).strip()
            else:
                return str(result).strip()
        except Exception as e:
            return f"Error solving problem: {str(e)}"
    
    def solve_with_code(self, problem: str) -> Tuple[str, str]:
        """
        Solve a mathematical problem and return both the code and the result.
        
        Args:
            problem: The mathematical problem to solve
            
        Returns:
            A tuple of (generated_code, result)
        """
        # For debugging and transparency, we'll capture the generated code
        prompt = f"""{self.math_system_prompt}

Problem: {problem}

Please write Python code to solve this problem. First, show the complete Python code, then execute it."""
        
        try:
            # Run the agent
            response = self.agent.run(prompt)
            
            # Try to extract code from the response
            code_match = re.search(r'```python\n(.*?)\n```', str(response), re.DOTALL)
            if code_match:
                code = code_match.group(1)
            else:
                code = "# Code extraction failed"
            
            # Get the result
            if isinstance(response, dict) and 'output' in response:
                result = str(response['output']).strip()
            else:
                result = str(response).strip()
            
            return code, result
        except Exception as e:
            return "", f"Error: {str(e)}"


def parse_dataset_file(file_path: str) -> List[Tuple[str, str]]:
    """
    Parse a dataset file containing math problems and their answers.
    
    Args:
        file_path: Path to the dataset file
        
    Returns:
        List of (problem, answer) tuples
    """
    problems = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Process lines in pairs (problem, answer)
    i = 0
    while i < len(lines):
        if i + 1 < len(lines):
            problem = lines[i].strip()
            answer = lines[i + 1].strip()
            if problem and answer:  # Skip empty lines
                problems.append((problem, answer))
            i += 2
        else:
            break
    
    return problems


def evaluate_agent(agent: MathCodeAgent, dataset_path: str, num_problems: int = 10) -> Dict[str, Any]:
    """
    Evaluate the agent on a dataset of math problems.
    
    Args:
        agent: The MathCodeAgent instance
        dataset_path: Path to the dataset file
        num_problems: Number of problems to evaluate (default: 10)
        
    Returns:
        Dictionary containing evaluation results
    """
    problems = parse_dataset_file(dataset_path)
    
    # Limit to specified number of problems
    problems = problems[:num_problems]
    
    results: Dict[str, Any] = {
        'total': len(problems),
        'correct': 0,
        'incorrect': 0,
        'errors': 0,
        'details': []
    }
    
    for i, (problem, expected_answer) in enumerate(problems):
        print(f"\nProblem {i+1}/{len(problems)}: {problem}")
        
        try:
            # Solve the problem
            answer = agent.solve(problem)
            
            # Clean up the answer
            answer = answer.strip()
            
            # Check if the answer is correct
            # Handle different answer formats
            if answer == expected_answer:
                results['correct'] += 1
                status = 'correct'
            elif normalize_answer(answer) == normalize_answer(expected_answer):
                results['correct'] += 1
                status = 'correct'
            else:
                results['incorrect'] += 1
                status = 'incorrect'
            
            print(f"Expected: {expected_answer}")
            print(f"Got: {answer}")
            print(f"Status: {status}")
            
            results['details'].append({
                'problem': problem,
                'expected': expected_answer,
                'got': answer,
                'status': status
            })
            
        except Exception as e:
            print(f"Error: {str(e)}")
            results['errors'] += 1
            results['details'].append({
                'problem': problem,
                'expected': expected_answer,
                'got': f"Error: {str(e)}",
                'status': 'error'
            })
    
    # Calculate accuracy
    results['accuracy'] = results['correct'] / results['total'] if results['total'] > 0 else 0
    
    return results


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer for comparison.
    
    Args:
        answer: The answer string to normalize
        
    Returns:
        Normalized answer string
    """
    # Remove extra whitespace
    answer = answer.strip()
    
    # Handle common variations
    # Remove trailing zeros after decimal point
    if '.' in answer:
        try:
            # Try to convert to float and back to remove trailing zeros
            num = float(answer)
            if num == int(num):
                answer = str(int(num))
            else:
                answer = f"{num:g}"  # Use general format to remove trailing zeros
        except:
            pass
    
    # Handle list/tuple formatting for sorting problems
    answer = answer.replace('(', '').replace(')', '')
    answer = answer.replace('[', '').replace(']', '')
    
    return answer


def main():
    """
    Main function to demonstrate the Math Code Agent.
    """
    # Initialize the agent with local inference
    print("Initializing Math Code Agent with local inference...")
    
    # For local inference, you may still need to login to Hugging Face to download models
    # login()
    
    # Create agent with local inference (default)
    # For smaller/faster models, you can use:
    # agent = MathCodeAgent(model_name="HuggingFaceTB/SmolLM-1.7B-Instruct")
    # agent = MathCodeAgent(model_name="microsoft/DialoGPT-medium")
    
    # Default uses a good balance between size and performance
    agent = MathCodeAgent(use_local=True)
    
    # Test on some example problems
    test_problems = [
        "1.480219 - 0.2",
        "What is 0.06 less than -0.2?",
        "Solve -20*b + 128*b + 648 = 0 for b.",
        "Sort 0.2, 1/10, 1/2, -5 in decreasing order.",
        "What is 62.131 + -4?"
    ]
    
    print("\n=== Testing Individual Problems ===")
    for problem in test_problems:
        print(f"\nProblem: {problem}")
        try:
            answer = agent.solve(problem)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Error: {e}")
    
    # Evaluate on a dataset
    print("\n\n=== Evaluating on Dataset ===")
    dataset_path = "/Users/sergeevnikita/Agentic-AI-for-Mathematical-Equation-Solving/data/math_qa/train-easy/arithmetic__add_or_sub.txt"
    
    if os.path.exists(dataset_path):
        try:
            results = evaluate_agent(agent, dataset_path, num_problems=3)  # Reduced for local testing
            
            print("\n=== Evaluation Results ===")
            print(f"Total problems: {results['total']}")
            print(f"Correct: {results['correct']}")
            print(f"Incorrect: {results['incorrect']}")
            print(f"Errors: {results['errors']}")
            print(f"Accuracy: {results['accuracy']:.2%}")
            
            # Save results
            with open('evaluation_results.json', 'w') as f:
                json.dump(results, f, indent=2)
            print("\nResults saved to evaluation_results.json")
        except Exception as e:
            print(f"Evaluation error: {e}")
    else:
        print("Dataset not found. Testing with individual problems only.")
        
    print("\n=== Local Inference Setup Complete ===")
    print("The agent is now using local model inference!")
    print("Available options:")
    print("- Default: Qwen/Qwen2.5-Coder-7B-Instruct")
    print("- For Apple Silicon: Automatic MLX detection")
    print("- For smaller models: HuggingFaceTB/SmolLM-1.7B-Instruct")
    print("- For remote fallback: MathCodeAgent(use_local=False)")


if __name__ == "__main__":
    main() 