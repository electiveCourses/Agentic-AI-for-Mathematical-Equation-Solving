"""Custom exceptions for the math agent package."""

from typing import Any, Dict, Optional


class MathAgentError(Exception):
    """Base exception for all math agent errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the exception.

        Args:
            message: Human-readable error message
            error_code: Machine-readable error code
            details: Additional error details
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}


class ModelError(MathAgentError):
    """Exception raised when there's an error with the model."""

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the model error.

        Args:
            message: Human-readable error message
            model_name: Name of the model that caused the error
            error_code: Machine-readable error code
            details: Additional error details
        """
        super().__init__(message, error_code, details)
        self.model_name = model_name


class ParsingError(MathAgentError):
    """Exception raised when there's an error parsing the model output."""

    def __init__(
        self,
        message: str,
        raw_output: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the parsing error.

        Args:
            message: Human-readable error message
            raw_output: The raw output that failed to parse
            error_code: Machine-readable error code
            details: Additional error details
        """
        super().__init__(message, error_code, details)
        self.raw_output = raw_output


class EvaluationError(MathAgentError):
    """Exception raised when there's an error during evaluation."""

    def __init__(
        self,
        message: str,
        problem: Optional[str] = None,
        expected_answer: Optional[str] = None,
        predicted_answer: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the evaluation error.

        Args:
            message: Human-readable error message
            problem: The problem that caused the error
            expected_answer: The expected answer
            predicted_answer: The predicted answer
            error_code: Machine-readable error code
            details: Additional error details
        """
        super().__init__(message, error_code, details)
        self.problem = problem
        self.expected_answer = expected_answer
        self.predicted_answer = predicted_answer


class CodeExecutionError(MathAgentError):
    """Exception raised when there's an error executing generated code."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        execution_output: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the code execution error.

        Args:
            message: Human-readable error message
            code: The code that failed to execute
            execution_output: The output from the failed execution
            error_code: Machine-readable error code
            details: Additional error details
        """
        super().__init__(message, error_code, details)
        self.code = code
        self.execution_output = execution_output


class ConfigurationError(MathAgentError):
    """Exception raised when there's a configuration error."""

    def __init__(
        self,
        message: str,
        config_key: Optional[str] = None,
        config_value: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the configuration error.

        Args:
            message: Human-readable error message
            config_key: The configuration key that caused the error
            config_value: The configuration value that caused the error
            error_code: Machine-readable error code
            details: Additional error details
        """
        super().__init__(message, error_code, details)
        self.config_key = config_key
        self.config_value = config_value


class DatasetError(MathAgentError):
    """Exception raised when there's an error with the dataset."""

    def __init__(
        self,
        message: str,
        dataset_name: Optional[str] = None,
        dataset_path: Optional[str] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the dataset error.

        Args:
            message: Human-readable error message
            dataset_name: Name of the dataset that caused the error
            dataset_path: Path to the dataset that caused the error
            error_code: Machine-readable error code
            details: Additional error details
        """
        super().__init__(message, error_code, details)
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path


class ValidationError(MathAgentError):
    """Exception raised when there's a validation error."""

    def __init__(
        self,
        message: str,
        field_name: Optional[str] = None,
        field_value: Optional[Any] = None,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize the validation error.

        Args:
            message: Human-readable error message
            field_name: The field that failed validation
            field_value: The value that failed validation
            error_code: Machine-readable error code
            details: Additional error details
        """
        super().__init__(message, error_code, details)
        self.field_name = field_name
        self.field_value = field_value


# Exception hierarchy for easy catching
__all__ = [
    "MathAgentError",
    "ModelError",
    "ParsingError",
    "EvaluationError",
    "CodeExecutionError",
    "ConfigurationError",
    "DatasetError",
    "ValidationError",
]
