from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_from_disk
import re
from tqdm import tqdm
from collections import defaultdict


prompt = """
Evaluate any defined expressions or constants.
Substitute known values into subsequent equations where applicable.
Solve any resulting systems of equations to find the values of unknowns.
If a variable is defined in terms of another (e.g., x=ay+bx=ay+b), solve for the required variable.
Present all steps clearly and logically, showing how each result is derived from the previous one.
Ensure all algebraic manipulations are valid. Simplify expressions where appropriate.
Provide the final answer, along with intermediate steps as needed for clarity. At the end, give just a numerical or symbolic answer.
At the end, clearly separate the final numerical answer using the format:
FinalAnswer: <value>
"""

def parse_llm_output(output_text):
    """
    Parse LLM output to extract the final answer.
    
    Args:
        output_text (str): The raw output from the LLM
        
    Returns:
        dict: Contains 'answer', 'reasoning', and 'success' fields
    """
    try:
        # Remove special tokens and clean the text
        cleaned_text = output_text.replace("<bos>", "").replace("<end_of_turn>", "").strip()
        
        # Split by user/assistant turns to get only the assistant's response
        if "<start_of_turn>" in cleaned_text:
            parts = cleaned_text.split("<start_of_turn>")
            if len(parts) > 1:
                assistant_response = parts[-1].strip()
            else:
                assistant_response = cleaned_text
        else:
            assistant_response = cleaned_text
        
        # Extract the final answer using regex
        final_answer_patterns = [
            r"FinalAnswer:\s*([^\n<]+)",  # Standard format
            r"Final Answer:\s*([^\n<]+)",  # Alternative format
            r"Answer:\s*([^\n<]+)",  # Simple format
            r"Therefore,?\s*[a-zA-Z]?\s*=\s*([^\n<]+)",  # Pattern like "Therefore, b = -6"
            r"The answer is\s*([^\n<]+)",  # Natural language format
        ]
        
        extracted_answer = None
        for pattern in final_answer_patterns:
            match = re.search(pattern, assistant_response, re.IGNORECASE)
            if match:
                extracted_answer = match.group(1).strip()
                break
        
        # If no pattern matched, try to find the last number or expression
        if not extracted_answer:
            # Look for patterns like "b = -6" or "x = 5" at the end
            number_pattern = r"[a-zA-Z]?\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
            matches = re.findall(number_pattern, assistant_response)
            if matches:
                extracted_answer = matches[-1]
        
        # Clean up the extracted answer
        if extracted_answer:
            extracted_answer = extracted_answer.strip(".,!?;")
            # Remove any trailing punctuation or markdown
            extracted_answer = re.sub(r'[*_`]+', '', extracted_answer)
            
        return {
            'answer': extracted_answer,
            'reasoning': assistant_response,
            'success': extracted_answer is not None,
            'raw_output': output_text
        }
        
    except Exception as e:
        return {
            'answer': None,
            'reasoning': output_text,
            'success': False,
            'error': str(e),
            'raw_output': output_text
        }

def evaluate_solution(predicted_answer, ground_truth):
    """
    Evaluate if the predicted answer matches the ground truth.
    
    Args:
        predicted_answer (str): The extracted answer from LLM
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
    

def solve_math_problem(problem, model, tokenizer, device, expected_answer=None):
    """
    Solve a math problem using the LLM and parse the result.
    
    Args:
        problem (str): The math problem to solve
        model: The loaded language model
        tokenizer: The tokenizer for the model
        device (str): The device to run the model on
        expected_answer (str, optional): The expected answer for evaluation
        
    Returns:
        dict: Contains the parsed result and evaluation metrics
    """
    # Prepare the input
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": problem}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Generate response
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_new_tokens=1024, temperature=0.2, top_p=0.9, do_sample=True)
    raw_output = tokenizer.decode(outputs[0])
    
    # Parse the output
    parsed_result = parse_llm_output(raw_output)
    
    # Evaluate if expected answer is provided
    if expected_answer is not None:
        parsed_result['correct'] = evaluate_solution(parsed_result['answer'], expected_answer)
        parsed_result['expected_answer'] = expected_answer
    
    return parsed_result

def evaluate_model_on_dataset(dataset, model, tokenizer, device, max_problems=None, categories=None, verbose=True):
    """
    Evaluate the model on a HuggingFace dataset with category-wise statistics.
    
    Args:
        dataset: HuggingFace dataset with 'question', 'answer', and 'category' fields
        model: The loaded language model
        tokenizer: The tokenizer for the model
        device (str): The device to run the model on
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
    print(f"Evaluating model on {total_problems} problems...")
    
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
            result = solve_math_problem(problem, model, tokenizer, device, expected_answer)
            
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
                'reasoning': result['reasoning']
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
    import json
    
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


# Extended evaluation and analysis
def run_full_evaluation(dataset, model, tokenizer, device, problems_per_category=10, verbose=False):
    """
    Run a comprehensive evaluation with balanced sampling from each category.
    
    Args:
        dataset: The HuggingFace dataset
        model: The loaded language model
        tokenizer: The tokenizer for the model
        device (str): The device to run the model on
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
    from datasets import Dataset
    balanced_dataset = Dataset.from_list(sampled_examples)
    
    print(f"Created balanced dataset with {len(balanced_dataset)} problems")
    
    # Run evaluation
    results = evaluate_model_on_dataset(balanced_dataset, model, tokenizer, device, verbose=verbose)
    
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
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print('USING DEVICE: ', device)

    checkpoint = "google/gemma-3-1b-it"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

    print("Starting comprehensive evaluation...")

    dataset = load_from_disk("../../data/processed/math_qa_dataset")
    full_results = run_full_evaluation(dataset, model, tokenizer, device, problems_per_category=5, verbose=True)
    save_evaluation_results(full_results, "comprehensive_evaluation_results.json")
    print(full_results)


if __name__ == "__main__":
    main()