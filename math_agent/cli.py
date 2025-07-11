"""Command-line interface for the math agent package."""

import argparse
import json
import logging
import os
import sys

from .core import (
    VERSION,
    ConfigurationError,
    DatasetError,
    LogLevel,
    MathAgentConfig,
    MathAgentError,
    ProblemCategory,
    get_config,
    set_config,
    setup_logging,
)


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="math-agent",
        description="Agentic AI for Mathematical Equation Solving",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Solve a single problem
  math-agent solve "What is 2 + 3?"

  # Evaluate on a dataset
  math-agent evaluate --dataset data/test.txt --max-problems 100

  # Configure the agent
  math-agent config --model-name "microsoft/DialoGPT-medium" --temperature 0.2

  # Show configuration
  math-agent config --show

  # Run with debug logging
  math-agent solve "x^2 + 3x + 2 = 0" --log-level DEBUG
        """,
    )

    # Global options
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
        help="Show version information",
    )

    parser.add_argument(
        "--config-file",
        type=str,
        metavar="PATH",
        help="Path to configuration file",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=[level.value for level in LogLevel],
        default="INFO",
        help="Set logging level (default: INFO)",
    )

    parser.add_argument(
        "--log-file",
        type=str,
        metavar="PATH",
        help="Path to log file",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress output (except errors)",
    )

    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        help="Available commands",
        metavar="COMMAND",
    )

    # Solve command
    solve_parser = subparsers.add_parser(
        "solve",
        help="Solve a mathematical problem",
        description="Solve a single mathematical problem using the configured agent",
    )
    solve_parser.add_argument(
        "problem",
        type=str,
        help="Mathematical problem to solve",
    )
    solve_parser.add_argument(
        "--expected-answer",
        type=str,
        help="Expected answer for validation",
    )
    solve_parser.add_argument(
        "--model-name",
        type=str,
        help="Model name to use for solving",
    )
    solve_parser.add_argument(
        "--temperature",
        type=float,
        help="Temperature for text generation",
    )
    solve_parser.add_argument(
        "--max-retries",
        type=int,
        help="Maximum number of retry attempts",
    )
    solve_parser.add_argument(
        "--show-reasoning",
        action="store_true",
        help="Show the reasoning process",
    )

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate agent on a dataset",
        description="Evaluate the agent's performance on a mathematical dataset",
    )
    eval_parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset file or directory",
    )
    eval_parser.add_argument(
        "--max-problems",
        type=int,
        help="Maximum number of problems to evaluate",
    )
    eval_parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        choices=[cat.value for cat in ProblemCategory],
        help="Problem categories to evaluate",
    )
    eval_parser.add_argument(
        "--output-file",
        type=str,
        help="Output file for results",
    )
    eval_parser.add_argument(
        "--model-name",
        type=str,
        help="Model name to use for evaluation",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for evaluation",
    )
    eval_parser.add_argument(
        "--save-detailed",
        action="store_true",
        help="Save detailed results for each problem",
    )

    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Configure the math agent",
        description="Configure the math agent settings",
    )
    config_parser.add_argument(
        "--show",
        action="store_true",
        help="Show current configuration",
    )
    config_parser.add_argument(
        "--save",
        type=str,
        help="Save configuration to file",
    )
    config_parser.add_argument(
        "--load",
        type=str,
        help="Load configuration from file",
    )
    config_parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset to default configuration",
    )
    config_parser.add_argument(
        "--model-name",
        type=str,
        help="Set model name",
    )
    config_parser.add_argument(
        "--temperature",
        type=float,
        help="Set temperature",
    )
    config_parser.add_argument(
        "--max-tokens",
        type=int,
        help="Set maximum tokens",
    )
    config_parser.add_argument(
        "--use-local",
        action="store_true",
        help="Use local model inference",
    )
    config_parser.add_argument(
        "--use-remote",
        action="store_true",
        help="Use remote model inference",
    )

    # Interactive command
    interactive_parser = subparsers.add_parser(
        "interactive",
        help="Start interactive mode",
        description="Start an interactive session for solving problems",
    )
    interactive_parser.add_argument(
        "--model-name",
        type=str,
        help="Model name to use",
    )
    interactive_parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show configuration at startup",
    )

    # Dataset command
    dataset_parser = subparsers.add_parser(
        "dataset",
        help="Dataset management commands",
        description="Commands for managing mathematical datasets",
    )
    dataset_subparsers = dataset_parser.add_subparsers(
        dest="dataset_action",
        help="Dataset actions",
        metavar="ACTION",
    )

    # Dataset info
    dataset_info_parser = dataset_subparsers.add_parser(
        "info",
        help="Show dataset information",
    )
    dataset_info_parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset",
    )

    # Dataset validate
    dataset_validate_parser = dataset_subparsers.add_parser(
        "validate",
        help="Validate dataset format",
    )
    dataset_validate_parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to dataset",
    )

    # Dataset convert
    dataset_convert_parser = dataset_subparsers.add_parser(
        "convert",
        help="Convert dataset format",
    )
    dataset_convert_parser.add_argument(
        "input_path",
        type=str,
        help="Input dataset path",
    )
    dataset_convert_parser.add_argument(
        "output_path",
        type=str,
        help="Output dataset path",
    )
    dataset_convert_parser.add_argument(
        "--format",
        type=str,
        choices=["txt", "json", "jsonl", "csv", "parquet"],
        default="json",
        help="Output format (default: json)",
    )

    return parser


def setup_logging_from_args(args: argparse.Namespace) -> None:
    """Setup logging based on command-line arguments."""
    # Determine log level
    if args.quiet:
        log_level = LogLevel.ERROR.value
    elif args.verbose:
        log_level = LogLevel.DEBUG.value
    else:
        log_level = args.log_level

    # Setup logging
    setup_logging(
        level=log_level,
        log_file=args.log_file,
        console_output=not args.quiet,
    )


def load_config_from_args(args: argparse.Namespace) -> MathAgentConfig:
    """Load configuration from command-line arguments."""
    # Load from file if specified
    if args.config_file:
        try:
            config = MathAgentConfig.from_file(args.config_file)
        except Exception as e:
            raise ConfigurationError(f"Failed to load config file: {e}")
    else:
        config = get_config()

    return config


def solve_command(args: argparse.Namespace) -> None:
    """Handle the solve command."""
    from .code_exec.math_code_agent import MathCodeAgent

    logger = logging.getLogger(__name__)

    # Load configuration
    config = load_config_from_args(args)

    # Override config with command-line arguments
    if args.model_name:
        config.model.name = args.model_name
    if args.temperature:
        config.model.temperature = args.temperature
    if args.max_retries:
        config.evaluation.max_retries = args.max_retries

    # Create agent
    logger.info(f"Initializing agent with model: {config.model.name}")
    agent = MathCodeAgent(
        model_name=config.model.name,
        use_local=config.model.use_local,
        use_mlx=config.model.use_mlx,
    )

    # Solve problem
    logger.info(f"Solving problem: {args.problem}")
    try:
        result = agent.solve(args.problem)

        print(f"Problem: {args.problem}")
        print(f"Answer: {result}")

        if args.expected_answer:
            correct = result.strip() == args.expected_answer.strip()
            print(f"Expected: {args.expected_answer}")
            print(f"Correct: {'✅' if correct else '❌'}")

        if args.show_reasoning:
            # This would require extending the agent to return reasoning
            print("\nReasoning: [Not implemented yet]")

    except Exception as e:
        logger.error(f"Failed to solve problem: {e}")
        print(f"Error: {e}")
        sys.exit(1)


def evaluate_command(args: argparse.Namespace) -> None:
    """Handle the evaluate command."""
    from datasets import load_from_disk

    from .code_exec.math_code_agent import MathCodeAgent
    from .utils.commons import evaluate_agent_on_dataset

    logger = logging.getLogger(__name__)

    # Load configuration
    config = load_config_from_args(args)

    # Override config with command-line arguments
    if args.model_name:
        config.model.name = args.model_name
    if args.max_problems:
        config.evaluation.max_problems = args.max_problems
    if args.batch_size:
        config.data.batch_size = args.batch_size

    # Load dataset
    logger.info(f"Loading dataset from: {args.dataset}")
    try:
        if os.path.isdir(args.dataset):
            dataset = load_from_disk(args.dataset)
        else:
            # Handle text files
            with open(args.dataset, "r") as f:
                lines = f.readlines()

            # Simple format: alternating problem/answer lines
            problems = []
            answers = []
            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    problems.append(lines[i].strip())
                    answers.append(lines[i + 1].strip())

            # Create a simple dataset structure
            dataset = {
                "question": problems,
                "answer": answers,
                "category": ["unknown"] * len(problems),
            }

    except Exception as e:
        raise DatasetError(f"Failed to load dataset: {e}")

    # Create agent
    logger.info(f"Initializing agent with model: {config.model.name}")
    agent = MathCodeAgent(
        model_name=config.model.name,
        use_local=config.model.use_local,
        use_mlx=config.model.use_mlx,
    )

    # Run evaluation
    logger.info("Starting evaluation...")
    try:

        def solve_function(problem: str, expected_answer: str) -> dict:
            result = agent.solve(problem)
            return {
                "answer": result,
                "correct": result.strip() == expected_answer.strip(),
                "success": True,
                "raw_output": result,
            }

        results = evaluate_agent_on_dataset(
            dataset=dataset,
            solve_function=solve_function,
            solve_function_args={},
            max_problems=config.evaluation.max_problems,
            categories=args.categories,
            verbose=config.evaluation.verbose,
        )

        # Save results
        if args.output_file:
            output_path = args.output_file
        else:
            output_path = f"evaluation_results_{agent.__class__.__name__}.json"

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        print(f"Error: {e}")
        sys.exit(1)


def config_command(args: argparse.Namespace) -> None:
    """Handle the config command."""
    logger = logging.getLogger(__name__)

    if args.show:
        config = get_config()
        print(json.dumps(config.to_dict(), indent=2))
        return

    if args.reset:
        from .core import reset_config

        reset_config()
        logger.info("Configuration reset to defaults")
        return

    if args.load:
        try:
            config = MathAgentConfig.from_file(args.load)
            set_config(config)
            logger.info(f"Configuration loaded from: {args.load}")
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)
        return

    # Update configuration
    config = get_config()

    if args.model_name:
        config.model.name = args.model_name
    if args.temperature:
        config.model.temperature = args.temperature
    if args.max_tokens:
        config.model.max_tokens = args.max_tokens
    if args.use_local:
        config.model.use_local = True
    if args.use_remote:
        config.model.use_local = False

    # Validate and set
    try:
        config.validate()
        set_config(config)
        logger.info("Configuration updated successfully")
    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        sys.exit(1)

    # Save if requested
    if args.save:
        try:
            config.save(args.save)
            logger.info(f"Configuration saved to: {args.save}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            sys.exit(1)


def interactive_command(args: argparse.Namespace) -> None:
    """Handle the interactive command."""
    from .code_exec.math_code_agent import MathCodeAgent

    logger = logging.getLogger(__name__)

    # Load configuration
    config = load_config_from_args(args)

    if args.model_name:
        config.model.name = args.model_name

    if args.show_config:
        print("Current configuration:")
        print(json.dumps(config.to_dict(), indent=2))
        print()

    # Create agent
    logger.info(f"Initializing agent with model: {config.model.name}")
    agent = MathCodeAgent(
        model_name=config.model.name,
        use_local=config.model.use_local,
        use_mlx=config.model.use_mlx,
    )

    print("Math Agent Interactive Mode")
    print("Type 'quit' or 'exit' to exit")
    print("Type 'help' for help")
    print()

    while True:
        try:
            problem = input("Problem: ").strip()

            if problem.lower() in ["quit", "exit"]:
                break
            elif problem.lower() == "help":
                print("Available commands:")
                print("  quit/exit - Exit interactive mode")
                print("  help - Show this help")
                print("  <problem> - Solve a mathematical problem")
                print()
                continue
            elif not problem:
                continue

            print("Solving...")
            result = agent.solve(problem)
            print(f"Answer: {result}")
            print()

        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error solving problem: {e}")
            print(f"Error: {e}")
            print()

    print("Goodbye!")


def dataset_command(args: argparse.Namespace) -> None:
    """Handle the dataset command."""
    logger = logging.getLogger(__name__)

    if args.dataset_action == "info":
        # Show dataset information
        try:
            from datasets import load_from_disk

            dataset = load_from_disk(args.dataset_path)
            print(f"Dataset: {args.dataset_path}")
            print(f"Number of examples: {len(dataset)}")
            print(f"Features: {list(dataset.features.keys())}")

            # Show category distribution if available
            if "category" in dataset.features:
                categories = dataset["category"]
                from collections import Counter

                category_counts = Counter(categories)
                print("\nCategory distribution:")
                for category, count in sorted(category_counts.items()):
                    print(f"  {category}: {count}")

        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            sys.exit(1)

    elif args.dataset_action == "validate":
        # Validate dataset format
        print(f"Validating dataset: {args.dataset_path}")
        # This would implement dataset validation logic
        print("Dataset validation not implemented yet")

    elif args.dataset_action == "convert":
        # Convert dataset format
        print(f"Converting {args.input_path} to {args.output_path}")
        # This would implement dataset conversion logic
        print("Dataset conversion not implemented yet")


def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    setup_logging_from_args(args)
    logger = logging.getLogger(__name__)

    # Handle commands
    try:
        if args.command == "solve":
            solve_command(args)
        elif args.command == "evaluate":
            evaluate_command(args)
        elif args.command == "config":
            config_command(args)
        elif args.command == "interactive":
            interactive_command(args)
        elif args.command == "dataset":
            dataset_command(args)
        else:
            parser.print_help()

    except MathAgentError as e:
        logger.error(f"Math Agent Error: {e}")
        if hasattr(e, "error_code") and e.error_code:
            logger.error(f"Error Code: {e.error_code}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
