import json
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional

from datasets import Dataset
from tqdm import tqdm


def evaluate_agent_on_dataset(
    dataset: Dataset,
    solve_function: Callable,
    solve_function_args: Dict[str, Any],
    max_problems: Optional[int] = None,
    categories: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
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
        dataset = dataset.filter(lambda x: x["category"] in categories)

    # Limit the number of problems if specified
    if max_problems:
        dataset = dataset.select(range(min(max_problems, len(dataset))))

    # Initialize tracking variables
    results = []
    category_stats: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "parsed": 0, "errors": 0}
    )

    total_problems = len(dataset)
    print(f"Evaluating agent on {total_problems} problems...")

    # Process each problem
    if not verbose:  # use tqdm only if not printing detailed output
        tqdm_bar = tqdm(enumerate(dataset), total=total_problems, desc="Evaluating")
    else:
        tqdm_bar = enumerate(dataset)

    for i, example in tqdm_bar:
        problem = example["question"]
        expected_answer = example["answer"]
        category = example["category"]

        if not verbose:
            tqdm_bar.set_description(f"Problem {i + 1}/{total_problems} [{category}]")
        else:
            print(f"\nProblem {i + 1}/{total_problems} [{category}]: {problem}")

        # Update category total count
        category_stats[category]["total"] += 1

        try:
            result = solve_function(problem, expected_answer, **solve_function_args)

            if result["success"]:
                category_stats[category]["parsed"] += 1
                if result["correct"]:
                    category_stats[category]["correct"] += 1
                    if verbose:
                        print(f"✅ CORRECT: {result['answer']}")
                else:
                    if verbose:
                        print(
                            f"❌ WRONG: Got {result['answer']}, Expected {expected_answer}"
                        )
            else:
                if verbose:
                    print("⚠️ PARSING FAILED")

            results.append(
                {
                    "problem": problem,
                    "expected_answer": expected_answer,
                    "predicted_answer": result["answer"],
                    "correct": result.get("correct", False),
                    "parsed_successfully": result["success"],
                    "category": category,
                    "raw_output": result["raw_output"],
                }
            )

        except Exception as e:
            category_stats[category]["errors"] += 1
            if verbose:
                print(f"❌ ERROR: {str(e)}")
            results.append(
                {
                    "problem": problem,
                    "expected_answer": expected_answer,
                    "predicted_answer": None,
                    "correct": False,
                    "parsed_successfully": False,
                    "category": category,
                    "error": str(e),
                }
            )

    # Calculate overall metrics
    total_correct = sum(stats["correct"] for stats in category_stats.values())
    total_parsed = sum(stats["parsed"] for stats in category_stats.values())
    total_errors = sum(stats["errors"] for stats in category_stats.values())

    overall_parsing_accuracy = (
        total_parsed / total_problems if total_problems > 0 else 0
    )
    overall_solving_accuracy = (
        total_correct / total_problems if total_problems > 0 else 0
    )
    overall_conditional_accuracy = (
        total_correct / total_parsed if total_parsed > 0 else 0
    )

    # Calculate category-wise metrics
    category_metrics = {}
    for category, stats in category_stats.items():
        parsing_acc = stats["parsed"] / stats["total"] if stats["total"] > 0 else 0
        solving_acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0
        conditional_acc = (
            stats["correct"] / stats["parsed"] if stats["parsed"] > 0 else 0
        )

        category_metrics[category] = {
            "total_problems": stats["total"],
            "correct_answers": stats["correct"],
            "parsed_successfully": stats["parsed"],
            "errors": stats["errors"],
            "parsing_accuracy": parsing_acc,
            "solving_accuracy": solving_acc,
            "conditional_accuracy": conditional_acc,
        }

    # Create summary
    summary = {
        "overall_metrics": {
            "total_problems": total_problems,
            "parsed_successfully": total_parsed,
            "correct_answers": total_correct,
            "errors": total_errors,
            "parsing_accuracy": overall_parsing_accuracy,
            "solving_accuracy": overall_solving_accuracy,
            "conditional_accuracy": overall_conditional_accuracy,
        },
        "category_metrics": category_metrics,
        "detailed_results": results,
    }

    # Print summary
    if verbose:
        print(f"\n{'=' * 80}")
        print("EVALUATION SUMMARY")
        print(f"{'=' * 80}")
        print("Overall Results:")
        print(f"  Total Problems: {total_problems}")
        print(f"  Successfully Parsed: {total_parsed} ({overall_parsing_accuracy:.2%})")
        print(f"  Correct Answers: {total_correct} ({overall_solving_accuracy:.2%})")
        print(
            f"  Accuracy (given successful parsing): {overall_conditional_accuracy:.2%}"
        )
        print(f"  Errors: {total_errors}")

        print(f"\n{'=' * 80}")
        print("CATEGORY-WISE RESULTS")
        print(f"{'=' * 80}")

        # Create a nice table for category results
        category_data = []
        for category, metrics in category_metrics.items():
            category_data.append(
                [
                    category,
                    metrics["total_problems"],
                    metrics["correct_answers"],
                    f"{metrics['solving_accuracy']:.2%}",
                    f"{metrics['parsing_accuracy']:.2%}",
                    f"{metrics['conditional_accuracy']:.2%}",
                ]
            )

        # Sort by category name
        category_data.sort(key=lambda x: x[0])

        # Print table header
        print(
            f"{'Category':<15} {'Total':<7} {'Correct':<7} {'Solve%':<7} {'Parse%':<7} {'Cond%':<7}"
        )
        print("-" * 80)

        # Print category results
        for row in category_data:
            print(
                f"{row[0]:<15} {row[1]:<7} {row[2]:<7} {row[3]:<7} {row[4]:<7} {row[5]:<7}"
            )

    return summary


def save_evaluation_results(
    summary: Dict[str, Any], filename: str = "evaluation_results.json"
) -> None:
    """Save evaluation results to a JSON file."""
    # Create a serializable version (remove non-serializable parts)
    serializable_summary = {
        "overall_metrics": summary["overall_metrics"],
        "category_metrics": summary["category_metrics"],
        "detailed_results": [
            {
                "problem": r["problem"],
                "expected_answer": r["expected_answer"],
                "predicted_answer": r["predicted_answer"],
                "correct": r["correct"],
                "parsed_successfully": r["parsed_successfully"],
                "category": r["category"],
            }
            for r in summary["detailed_results"]
        ],
    }

    with open(filename, "w") as f:
        json.dump(serializable_summary, f, indent=2)

    print(f"Results saved to {filename}")


def run_full_evaluation(
    dataset: Dataset,
    solve_function: Callable,
    solve_function_args: Dict[str, Any],
    problems_per_category: int = 10,
    verbose: bool = False,
) -> Dict[str, Any]:
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
    print(
        f"Running comprehensive evaluation with {problems_per_category} problems per category..."
    )

    # Get all categories
    categories = list(set(dataset["category"]))
    print(f"Categories found: {categories}")

    # Sample problems from each category
    sampled_examples = []
    for category in categories:
        category_data = dataset.filter(lambda x: x["category"] == category)
        n_samples = min(problems_per_category, len(category_data))
        category_samples = category_data.select(range(n_samples))
        sampled_examples.extend(category_samples)

    # Create balanced dataset
    balanced_dataset = Dataset.from_list(sampled_examples)

    print(f"Created balanced dataset with {len(balanced_dataset)} problems")

    # Run evaluation
    results = evaluate_agent_on_dataset(
        balanced_dataset, solve_function, solve_function_args, verbose=verbose
    )

    # Print detailed analysis
    print("\n" + "=" * 80)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 80)

    # Sort categories by performance
    category_performance = []
    for category, metrics in results["category_metrics"].items():
        category_performance.append((category, metrics["solving_accuracy"]))

    category_performance.sort(key=lambda x: x[1], reverse=True)

    print("\nCATEGORY PERFORMANCE RANKING:")
    print("-" * 50)
    for i, (category, accuracy) in enumerate(category_performance, 1):
        print(f"{i:2d}. {category:<15} {accuracy:.2%}")

    # Find best and worst categories
    best_category = category_performance[0][0]
    worst_category = category_performance[-1][0]

    print(
        f"\nBest performing category: {best_category} ({category_performance[0][1]:.2%})"
    )
    print(
        f"Worst performing category: {worst_category} ({category_performance[-1][1]:.2%})"
    )

    # Identify parsing vs solving issues
    parsing_issues = []
    solving_issues = []

    for category, metrics in results["category_metrics"].items():
        if metrics["parsing_accuracy"] < 0.8:  # Less than 80% parsing success
            parsing_issues.append(category)
        elif (
            metrics["conditional_accuracy"] < 0.6
        ):  # Less than 60% accuracy given successful parsing
            solving_issues.append(category)

    if parsing_issues:
        print(f"\nCategories with parsing issues: {parsing_issues}")
    if solving_issues:
        print(f"Categories with solving issues: {solving_issues}")

    return results
