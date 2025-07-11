import json
import logging
import logging.config
import os
import sys
from typing import Any, Dict, Optional


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging."""

    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }

        # Add exception information if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)

        # Add extra fields if present
        if hasattr(record, "extra_fields"):
            log_entry.update(record.extra_fields)

        return json.dumps(log_entry, ensure_ascii=False)


def setup_logging(
    level: str = "INFO",
    format_type: str = "standard",
    log_file: Optional[str] = None,
    console_output: bool = True,
    json_logging: bool = False,
) -> None:
    """
    Setup logging configuration for the math agent.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        format_type: Format type ('standard', 'detailed', 'simple')
        log_file: Optional log file path
        console_output: Whether to output to console
        json_logging: Whether to use JSON structured logging
    """
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Define formatters
    formatters = {
        "simple": "%(levelname)s: %(message)s",
        "standard": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        "detailed": "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s",
    }

    # Create handlers
    handlers: Dict[str, logging.Handler] = {}

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)

        if json_logging:
            console_handler.setFormatter(JSONFormatter())
        else:
            console_handler.setFormatter(
                logging.Formatter(formatters.get(format_type, formatters["standard"]))
            )

        handlers["console"] = console_handler

    # File handler
    if log_file:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)

        if json_logging:
            file_handler.setFormatter(JSONFormatter())
        else:
            file_handler.setFormatter(logging.Formatter(formatters["detailed"]))

        handlers["file"] = file_handler

    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=list(handlers.values()),
        force=True,
    )

    # Configure specific loggers
    _configure_package_loggers(numeric_level)

    # Log the configuration
    logger = logging.getLogger(__name__)
    logger.info(
        "Logging configured",
        extra={
            "extra_fields": {
                "level": level,
                "format_type": format_type,
                "log_file": log_file,
                "console_output": console_output,
                "json_logging": json_logging,
            }
        },
    )


def _configure_package_loggers(level: int) -> None:
    """Configure loggers for different package components."""
    # Math agent loggers
    loggers = [
        "math_agent",
        "math_agent.baseline",
        "math_agent.code_exec",
        "math_agent.code_exec_eval",
        "math_agent.utils",
        "math_agent.core",
    ]

    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
        logger.propagate = True

    # Set external library log levels
    external_loggers = {
        "transformers": logging.WARNING,
        "torch": logging.WARNING,
        "datasets": logging.WARNING,
        "smolagents": logging.INFO,
        "urllib3": logging.WARNING,
        "requests": logging.WARNING,
    }

    for logger_name, log_level in external_loggers.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def log_function_call(logger: logging.Logger, func_name: str, **kwargs: Any) -> None:
    """Log a function call with parameters."""
    logger.debug(
        f"Calling function: {func_name}",
        extra={
            "extra_fields": {
                "function": func_name,
                "parameters": kwargs,
            }
        },
    )


def log_performance(
    logger: logging.Logger, operation: str, duration: float, **kwargs: Any
) -> None:
    """Log performance metrics."""
    logger.info(
        f"Performance metric: {operation}",
        extra={
            "extra_fields": {
                "operation": operation,
                "duration_seconds": duration,
                "metrics": kwargs,
            }
        },
    )


def log_evaluation_result(
    logger: logging.Logger,
    problem: str,
    expected: str,
    predicted: str,
    correct: bool,
    **kwargs: Any,
) -> None:
    """Log evaluation results."""
    logger.info(
        f"Evaluation result: {'CORRECT' if correct else 'INCORRECT'}",
        extra={
            "extra_fields": {
                "problem": problem,
                "expected_answer": expected,
                "predicted_answer": predicted,
                "correct": correct,
                "metadata": kwargs,
            }
        },
    )


# Configure logging from environment variables
def setup_logging_from_env() -> None:
    """Setup logging from environment variables."""
    level = os.getenv("MATH_AGENT_LOG_LEVEL", "INFO")
    format_type = os.getenv("MATH_AGENT_LOG_FORMAT", "standard")
    log_file = os.getenv("MATH_AGENT_LOG_FILE")
    console_output = os.getenv("MATH_AGENT_LOG_CONSOLE", "true").lower() == "true"
    json_logging = os.getenv("MATH_AGENT_LOG_JSON", "false").lower() == "true"

    setup_logging(
        level=level,
        format_type=format_type,
        log_file=log_file,
        console_output=console_output,
        json_logging=json_logging,
    )


# Default logging configuration
if not logging.getLogger().handlers:
    setup_logging_from_env()
