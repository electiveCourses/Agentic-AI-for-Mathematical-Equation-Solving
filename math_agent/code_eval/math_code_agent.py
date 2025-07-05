import os
import re
import json
from typing import Dict, List, Tuple, Optional, Any
from smolagents import CodeAgent, TransformersModel
import platform
from smolagents import InferenceClientModel
from datasets import load_from_disk, Dataset
from tqdm import tqdm
from collections import defaultdict

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
            model = InferenceClientModel(model_name)
            print(f"Using remote inference with {model_name}")
        
        # Initialize the smolagents CodeAgent
        self.agent = CodeAgent(
            tools=[],
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
        prompt = f"""{self.math_system_prompt}

Problem: {problem}

Please write Python code to solve this problem and execute it using the execute_python_code tool. 
Make sure to print the final answer in the exact format expected."""
        
        try:
            result = self.agent.run(prompt)
            if isinstance(result, dict) and 'output' in result:
                return str(result['output']).strip()
            else:
                return str(result).strip()
        except Exception as e:
            return f"Error solving problem: {str(e)}"
    
    def solve_with_code(self, problem: str) -> Tuple[str, str]:
        prompt = f"""{self.math_system_prompt}
Problem: {problem}

Please write Python code to solve this problem. First, show the complete Python code, then execute it."""
        try:
            response = self.agent.run(prompt)
            code_match = re.search(r'```python\n(.*?)\n```', str(response), re.DOTALL)
            if code_match:
                code = code_match.group(1)
            else:
                code = "# Code extraction failed"
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


def evaluate_solution(predicted_answer, ground_truth):
    """
    Evaluate if the predicted answer matches the ground truth.
    
    Args:
        predicted_answer (str): The extracted answer from agent
        ground_truth (str): The correct answer
        
    Returns:
        bool: True if answers match, False otherwise
    """
    if not predicted_answer or not ground_truth:
        return False
    
    # Clean both answers
    def clean_answer(ans):
        if ans is None:
            return None
        # Remove whitespace and convert to string
        ans = str(ans).strip()
        # Remove common formatting
        ans = ans.replace(" ", "").replace(",", "")
        return ans.lower()
    
    pred_clean = clean_answer(predicted_answer)
    truth_clean = clean_answer(ground_truth)
    
    if pred_clean == truth_clean:
        return True
    
    # Try to convert to numbers for comparison
    try:
        pred_num = float(pred_clean)
        truth_num = float(truth_clean)
        # Check if numbers are close (handle floating point precision)
        return abs(pred_num - truth_num) < 1e-6
    except (ValueError, TypeError):
        # If conversion fails, do string comparison
        return pred_clean == truth_clean


def solve_math_problem(problem, agent, expected_answer=None):
    """
    Solve a math problem using the agent and parse the result.
    
    Args:
        problem (str): The math problem to solve
        agent: The MathCodeAgent instance
        expected_answer (str, optional): The expected answer for evaluation
        
    Returns:
        dict: Contains the parsed result and evaluation metrics
    """
    try:
        # Solve the problem
        answer = agent.solve(problem)
        
        # Parse the output - check if we got a meaningful answer
        success = answer is not None and answer.strip() != "" and not answer.startswith("Error")
        
        result = {
            'answer': answer.strip() if answer else None,
            'success': success,
            'raw_output': answer
        }
        
        # Evaluate if expected answer is provided
        if expected_answer is not None:
            result['correct'] = evaluate_solution(result['answer'], expected_answer) if success else False
            result['expected_answer'] = expected_answer
        
        return result
        
    except Exception as e:
        return {
            'answer': None,
            'success': False,
            'error': str(e),
            'raw_output': f"Error: {str(e)}",
            'correct': False,
            'expected_answer': expected_answer
        }


def evaluate_agent_on_dataset(dataset, agent, max_problems=None, categories=None, verbose=True):
    """
    Evaluate the agent on a HuggingFace dataset with category-wise statistics.
    
    Args:
        dataset: HuggingFace dataset with 'question', 'answer', and 'category' fields
        agent: The MathCodeAgent instance
        max_problems (int, optional): Maximum number of problems to evaluate
        categories (list, optional): Specific categories to evaluate (if None, evaluate all)
        verbose (bool): Whether to print detailed progress
        
    Returns:
        dict: Evaluation results with overall and category-wise statistics
    """
    # Filter by categories if specified
    if categories:
        dataset = dataset.filter(lambda x: x['category'] in categories)
    
    # Limit the number of problems if specified
    if max_problems:
        dataset = dataset.select(range(min(max_problems, len(dataset))))
    
    # Initialize tracking variables
    results = []
    category_stats = defaultdict(lambda: {
        'total': 0, 'correct': 0, 'parsed': 0, 'errors': 0
    })
    
    total_problems = len(dataset)
    print(f"Evaluating agent on {total_problems} problems...")
    
    # Process each problem
    if not verbose:  # use tqdm only if not printing detailed output
        tqdm_bar = tqdm(enumerate(dataset), total=total_problems, desc="Evaluating")
    else:
        tqdm_bar = enumerate(dataset)
        
    for i, example in tqdm_bar:
        problem = example['question']
        expected_answer = example['answer']
        category = example['category']
        
        if not verbose:
            tqdm_bar.set_description(f"Problem {i+1}/{total_problems} [{category}]")
        else:
            print(f"\nProblem {i+1}/{total_problems} [{category}]: {problem[:50]}...")
        
        # Update category total count
        category_stats[category]['total'] += 1
        
        try:
            result = solve_math_problem(problem, agent, expected_answer)
            
            if result['success']:
                category_stats[category]['parsed'] += 1
                if result['correct']:
                    category_stats[category]['correct'] += 1
                    if verbose:
                        print(f"✅ CORRECT: {result['answer']}")
                else:
                    if verbose:
                        print(f"❌ WRONG: Got {result['answer']}, Expected {expected_answer}")
            else:
                if verbose:
                    print("⚠️ PARSING FAILED")
            
            results.append({
                'problem': problem,
                'expected_answer': expected_answer,
                'predicted_answer': result['answer'],
                'correct': result.get('correct', False),
                'parsed_successfully': result['success'],
                'category': category,
                'raw_output': result['raw_output']
            })
            
        except Exception as e:
            category_stats[category]['errors'] += 1
            if verbose:
                print(f"❌ ERROR: {str(e)}")
            results.append({
                'problem': problem,
                'expected_answer': expected_answer,
                'predicted_answer': None,
                'correct': False,
                'parsed_successfully': False,
                'category': category,
                'error': str(e)
            })
    
    # Calculate overall metrics
    total_correct = sum(stats['correct'] for stats in category_stats.values())
    total_parsed = sum(stats['parsed'] for stats in category_stats.values())
    total_errors = sum(stats['errors'] for stats in category_stats.values())
    
    overall_parsing_accuracy = total_parsed / total_problems if total_problems > 0 else 0
    overall_solving_accuracy = total_correct / total_problems if total_problems > 0 else 0
    overall_conditional_accuracy = total_correct / total_parsed if total_parsed > 0 else 0
    
    # Calculate category-wise metrics
    category_metrics = {}
    for category, stats in category_stats.items():
        parsing_acc = stats['parsed'] / stats['total'] if stats['total'] > 0 else 0
        solving_acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        conditional_acc = stats['correct'] / stats['parsed'] if stats['parsed'] > 0 else 0
        
        category_metrics[category] = {
            'total_problems': stats['total'],
            'correct_answers': stats['correct'],
            'parsed_successfully': stats['parsed'],
            'errors': stats['errors'],
            'parsing_accuracy': parsing_acc,
            'solving_accuracy': solving_acc,
            'conditional_accuracy': conditional_acc
        }
    
    # Create summary
    summary = {
        'overall_metrics': {
            'total_problems': total_problems,
            'parsed_successfully': total_parsed,
            'correct_answers': total_correct,
            'errors': total_errors,
            'parsing_accuracy': overall_parsing_accuracy,
            'solving_accuracy': overall_solving_accuracy,
            'conditional_accuracy': overall_conditional_accuracy
        },
        'category_metrics': category_metrics,
        'detailed_results': results
    }
    
    # Print summary
    if verbose:
        print(f"\n{'='*80}")
        print("EVALUATION SUMMARY")
        print(f"{'='*80}")
        print("Overall Results:")
        print(f"  Total Problems: {total_problems}")
        print(f"  Successfully Parsed: {total_parsed} ({overall_parsing_accuracy:.2%})")
        print(f"  Correct Answers: {total_correct} ({overall_solving_accuracy:.2%})")
        print(f"  Accuracy (given successful parsing): {overall_conditional_accuracy:.2%}")
        print(f"  Errors: {total_errors}")
        
        print(f"\n{'='*80}")
        print("CATEGORY-WISE RESULTS")
        print(f"{'='*80}")
        
        # Create a nice table for category results
        category_data = []
        for category, metrics in category_metrics.items():
            category_data.append([
                category,
                metrics['total_problems'],
                metrics['correct_answers'],
                f"{metrics['solving_accuracy']:.2%}",
                f"{metrics['parsing_accuracy']:.2%}",
                f"{metrics['conditional_accuracy']:.2%}"
            ])
        
        # Sort by category name
        category_data.sort(key=lambda x: x[0])
        
        # Print table header
        print(f"{'Category':<15} {'Total':<7} {'Correct':<7} {'Solve%':<7} {'Parse%':<7} {'Cond%':<7}")
        print("-" * 80)
        
        # Print category results
        for row in category_data:
            print(f"{row[0]:<15} {row[1]:<7} {row[2]:<7} {row[3]:<7} {row[4]:<7} {row[5]:<7}")
    
    return summary


def save_evaluation_results(summary, filename="evaluation_results.json"):
    """Save evaluation results to a JSON file."""
    # Create a serializable version (remove non-serializable parts)
    serializable_summary = {
        'overall_metrics': summary['overall_metrics'],
        'category_metrics': summary['category_metrics'],
        'detailed_results': [{
            'problem': r['problem'],
            'expected_answer': r['expected_answer'],
            'predicted_answer': r['predicted_answer'],
            'correct': r['correct'],
            'parsed_successfully': r['parsed_successfully'],
            'category': r['category']
        } for r in summary['detailed_results']]
    }
    
    with open(filename, 'w') as f:
        json.dump(serializable_summary, f, indent=2)
    
    print(f"Results saved to {filename}")


def run_full_evaluation(dataset, agent, problems_per_category=10, verbose=False):
    """
    Run a comprehensive evaluation with balanced sampling from each category.
    
    Args:
        dataset: The HuggingFace dataset
        agent: The MathCodeAgent instance
        problems_per_category: Number of problems to sample from each category
        verbose (bool): Whether to print detailed progress
        
    Returns:
        dict: Comprehensive evaluation results
    """
    print(f"Running comprehensive evaluation with {problems_per_category} problems per category...")
    
    # Get all categories
    categories = list(set(dataset['category']))
    print(f"Categories found: {categories}")
    
    # Sample problems from each category
    sampled_examples = []
    for category in categories:
        category_data = dataset.filter(lambda x: x['category'] == category)
        n_samples = min(problems_per_category, len(category_data))
        category_samples = category_data.select(range(n_samples))
        sampled_examples.extend(category_samples)
    
    # Create balanced dataset
    balanced_dataset = Dataset.from_list(sampled_examples)
    
    print(f"Created balanced dataset with {len(balanced_dataset)} problems")
    
    # Run evaluation
    results = evaluate_agent_on_dataset(balanced_dataset, agent, verbose=verbose)
    
    # Print detailed analysis
    print("\n" + "="*80)
    print("COMPREHENSIVE ANALYSIS")
    print("="*80)
    
    # Sort categories by performance
    category_performance = []
    for category, metrics in results['category_metrics'].items():
        category_performance.append((category, metrics['solving_accuracy']))
    
    category_performance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nCATEGORY PERFORMANCE RANKING:")
    print("-" * 50)
    for i, (category, accuracy) in enumerate(category_performance, 1):
        print(f"{i:2d}. {category:<15} {accuracy:.2%}")
    
    # Find best and worst categories
    best_category = category_performance[0][0]
    worst_category = category_performance[-1][0]
    
    print(f"\nBest performing category: {best_category} ({category_performance[0][1]:.2%})")
    print(f"Worst performing category: {worst_category} ({category_performance[-1][1]:.2%})")
    
    # Identify parsing vs solving issues
    parsing_issues = []
    solving_issues = []
    
    for category, metrics in results['category_metrics'].items():
        if metrics['parsing_accuracy'] < 0.8:  # Less than 80% parsing success
            parsing_issues.append(category)
        elif metrics['conditional_accuracy'] < 0.6:  # Less than 60% accuracy given successful parsing
            solving_issues.append(category)
    
    if parsing_issues:
        print(f"\nCategories with parsing issues: {parsing_issues}")
    if solving_issues:
        print(f"Categories with solving issues: {solving_issues}")
    
    return results


def main():
    """
    Main function to demonstrate the Math Code Agent.
    """
    # Initialize the agent with local inference
    print("Initializing Math Code Agent with local inference...")
    
    # Default uses a good balance between size and performance
    agent = MathCodeAgent(use_local=True)
    
    # Evaluate on a dataset using comprehensive evaluation
    print("\n\n=== Comprehensive Dataset Evaluation ===")
    
    # Load the processed dataset
    dataset_path = "../../data/processed/math_qa_dataset"
    
    if os.path.exists(dataset_path):
        try:
            print("Loading processed dataset...")
            dataset = load_from_disk(dataset_path)
            
            # Run comprehensive evaluation
            full_results = run_full_evaluation(dataset, agent, problems_per_category=5, verbose=True)
            
            # Save results
            save_evaluation_results(full_results, "comprehensive_evaluation_results.json")
            
            print("\n=== Evaluation Complete ===")
            print("Full results saved to comprehensive_evaluation_results.json")
            
        except Exception as e:
            print(f"Comprehensive evaluation error: {e}")
            print("Falling back to simple evaluation...")
            
            # Fallback to simple evaluation on text files
            simple_dataset_path = "../../data/math_qa/train-easy/arithmetic__add_or_sub.txt"
            if os.path.exists(simple_dataset_path):
                try:
                    results = evaluate_agent(agent, simple_dataset_path, num_problems=3)
                    
                    print("\n=== Simple Evaluation Results ===")
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
                    print(f"Simple evaluation error: {e}")
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