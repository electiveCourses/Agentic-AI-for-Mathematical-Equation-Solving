"""
Math Agent Package - Agentic AI for Mathematical Equation Solving

A comprehensive Python package for solving mathematical problems using AI agents
with code execution capabilities, built on top of modern LLMs and symbolic computation.
"""

from .core import (
    DEFAULT_MODEL_NAME,
    VERSION,
    EvaluationError,
    LogLevel,
    MathAgentConfig,
    MathAgentError,
    ModelError,
    ParsingError,
    ProblemCategory,
    get_config,
    set_config,
    setup_logging,
)

__version__ = VERSION
__author__ = "Math Agent Team"
__email__ = "your.email@example.com"
__license__ = "MIT"
__description__ = "Agentic AI for Mathematical Equation Solving"
__url__ = (
    "https://github.com/your-username/Agentic-AI-for-Mathematical-Equation-Solving"
)

__all__ = [
    # Core functionality
    "MathAgentConfig",
    "get_config",
    "set_config",
    "setup_logging",
    # Constants
    "VERSION",
    "DEFAULT_MODEL_NAME",
    "LogLevel",
    "ProblemCategory",
    # Exceptions
    "MathAgentError",
    "ModelError",
    "ParsingError",
    "EvaluationError",
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "__description__",
    "__url__",
]
