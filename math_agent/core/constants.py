"""Constants and enums for the math agent package."""

from enum import Enum
from typing import Final, List


class LogLevel(Enum):
    """Logging levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """Logging formats."""

    SIMPLE = "simple"
    STANDARD = "standard"
    DETAILED = "detailed"


class ModelType(Enum):
    """Model types."""

    TRANSFORMERS = "transformers"
    MLX = "mlx"
    INFERENCE_CLIENT = "inference_client"


class ProblemCategory(Enum):
    """Mathematical problem categories."""

    ALGEBRA = "algebra"
    ARITHMETIC = "arithmetic"
    CALCULUS = "calculus"
    COMPARISON = "comparison"
    MEASUREMENT = "measurement"
    NUMBERS = "numbers"
    POLYNOMIALS = "polynomials"
    PROBABILITY = "probability"


class DatasetSplit(Enum):
    """Dataset splits."""

    TRAIN_EASY = "train-easy"
    TRAIN_MEDIUM = "train-medium"
    TRAIN_HARD = "train-hard"
    INTERPOLATE = "interpolate"
    EXTRAPOLATE = "extrapolate"


class EvaluationStatus(Enum):
    """Evaluation status."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(Enum):
    """Agent status."""

    IDLE = "idle"
    LOADING = "loading"
    READY = "ready"
    PROCESSING = "processing"
    ERROR = "error"


# Default values
DEFAULT_MODEL_NAME: Final[str] = "Qwen/Qwen2.5-Coder-7B-Instruct"
DEFAULT_TEMPERATURE: Final[float] = 0.1
DEFAULT_MAX_TOKENS: Final[int] = 2048
DEFAULT_MAX_NEW_TOKENS: Final[int] = 512
DEFAULT_TOP_P: Final[float] = 0.85
DEFAULT_REPETITION_PENALTY: Final[float] = 1.1

# File extensions
SUPPORTED_CONFIG_EXTENSIONS: Final[List[str]] = [".json", ".yaml", ".yml"]
SUPPORTED_DATA_EXTENSIONS: Final[List[str]] = [
    ".txt",
    ".json",
    ".jsonl",
    ".csv",
    ".parquet",
]

# Directories
DEFAULT_LOG_DIR: Final[str] = "logs"
DEFAULT_RESULTS_DIR: Final[str] = "results"
DEFAULT_CACHE_DIR: Final[str] = "data/cache"
DEFAULT_DATASET_PATH: Final[str] = "data/processed/math_qa_dataset"

# Timeouts and limits
DEFAULT_CODE_EXECUTION_TIMEOUT: Final[int] = 30
DEFAULT_MODEL_TIMEOUT: Final[int] = 300
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_BATCH_SIZE: Final[int] = 32
DEFAULT_NUM_WORKERS: Final[int] = 4

# Patterns for answer extraction
ANSWER_PATTERNS: Final[List[str]] = [
    r"FinalAnswer:\s*([^\n<]+)",
    r"Final Answer:\s*([^\n<]+)",
    r"Answer:\s*([^\n<]+)",
    r"Therefore,?\s*[a-zA-Z]?\s*=\s*([^\n<]+)",
    r"The answer is\s*([^\n<]+)",
    r"The probability is\s*([^\n<]+)",
    r"=\s*([^\n<]+?)(?:\s|$)",
]

# System prompts
DEFAULT_MATH_SYSTEM_PROMPT: Final[
    str
] = """You are a mathematical problem solver. When given a math problem, you must:

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

# Allowed imports for code execution
DEFAULT_ALLOWED_IMPORTS: Final[List[str]] = [
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
]

# Environment variable names
ENV_VARS: Final[dict] = {
    "MODEL_NAME": "MATH_AGENT_MODEL_NAME",
    "USE_LOCAL": "MATH_AGENT_USE_LOCAL",
    "TEMPERATURE": "MATH_AGENT_TEMPERATURE",
    "MAX_TOKENS": "MATH_AGENT_MAX_TOKENS",
    "MAX_NEW_TOKENS": "MATH_AGENT_MAX_NEW_TOKENS",
    "TOP_P": "MATH_AGENT_TOP_P",
    "REPETITION_PENALTY": "MATH_AGENT_REPETITION_PENALTY",
    "LOG_LEVEL": "MATH_AGENT_LOG_LEVEL",
    "LOG_FORMAT": "MATH_AGENT_LOG_FORMAT",
    "LOG_FILE": "MATH_AGENT_LOG_FILE",
    "LOG_CONSOLE": "MATH_AGENT_LOG_CONSOLE",
    "LOG_JSON": "MATH_AGENT_LOG_JSON",
    "LOG_DIR": "MATH_AGENT_LOG_DIR",
    "MAX_PROBLEMS": "MATH_AGENT_MAX_PROBLEMS",
    "CATEGORIES": "MATH_AGENT_CATEGORIES",
    "MAX_RETRIES": "MATH_AGENT_MAX_RETRIES",
    "TIMEOUT": "MATH_AGENT_TIMEOUT",
    "VERBOSE": "MATH_AGENT_VERBOSE",
    "SAVE_RESULTS": "MATH_AGENT_SAVE_RESULTS",
    "RESULTS_DIR": "MATH_AGENT_RESULTS_DIR",
    "DATASET_PATH": "MATH_AGENT_DATASET_PATH",
    "CACHE_DIR": "MATH_AGENT_CACHE_DIR",
    "BATCH_SIZE": "MATH_AGENT_BATCH_SIZE",
    "NUM_WORKERS": "MATH_AGENT_NUM_WORKERS",
    "PREPROCESSING_CACHE": "MATH_AGENT_PREPROCESSING_CACHE",
    "CODE_TIMEOUT": "MATH_AGENT_CODE_TIMEOUT",
    "VERBOSITY": "MATH_AGENT_VERBOSITY",
    "PROJECT_ROOT": "MATH_AGENT_PROJECT_ROOT",
    "RANDOM_SEED": "MATH_AGENT_RANDOM_SEED",
    "DEBUG": "MATH_AGENT_DEBUG",
}

# Error codes
ERROR_CODES: Final[dict] = {
    "MODEL_LOAD_FAILED": "E001",
    "MODEL_INFERENCE_FAILED": "E002",
    "PARSING_FAILED": "E003",
    "CODE_EXECUTION_FAILED": "E004",
    "EVALUATION_FAILED": "E005",
    "CONFIGURATION_INVALID": "E006",
    "DATASET_LOAD_FAILED": "E007",
    "VALIDATION_FAILED": "E008",
    "TIMEOUT_EXCEEDED": "E009",
    "INSUFFICIENT_MEMORY": "E010",
    "UNKNOWN_ERROR": "E999",
}

# Success codes
SUCCESS_CODES: Final[dict] = {
    "PROBLEM_SOLVED": "S001",
    "EVALUATION_COMPLETED": "S002",
    "MODEL_LOADED": "S003",
    "CONFIGURATION_LOADED": "S004",
    "DATASET_LOADED": "S005",
}

# File patterns
FILE_PATTERNS: Final[dict] = {
    "RESULT_FILE": "evaluation_results_{timestamp}.json",
    "LOG_FILE": "math_agent_{timestamp}.log",
    "CONFIG_FILE": "math_agent_config.json",
    "CHECKPOINT_FILE": "model_checkpoint_{epoch}.pt",
}

# Unicode symbols for logging
SYMBOLS: Final[dict] = {
    "SUCCESS": "✅",
    "ERROR": "❌",
    "WARNING": "⚠️",
    "INFO": "ℹ️",
    "DEBUG": "🔍",
    "PROCESSING": "⏳",
    "COMPLETED": "✨",
    "FAILED": "💥",
}

# Performance thresholds
PERFORMANCE_THRESHOLDS: Final[dict] = {
    "MIN_ACCURACY": 0.5,
    "TARGET_ACCURACY": 0.8,
    "MAX_INFERENCE_TIME": 30.0,
    "MAX_PARSING_TIME": 5.0,
    "MAX_EVALUATION_TIME": 300.0,
}

# Model configuration presets
MODEL_PRESETS: Final[dict] = {
    "fast": {
        "temperature": 0.1,
        "max_new_tokens": 256,
        "top_p": 0.9,
    },
    "accurate": {
        "temperature": 0.05,
        "max_new_tokens": 1024,
        "top_p": 0.8,
    },
    "creative": {
        "temperature": 0.3,
        "max_new_tokens": 512,
        "top_p": 0.95,
    },
}

# Dataset statistics (for validation)
DATASET_STATS: Final[dict] = {
    "min_problems_per_category": 100,
    "max_problems_per_category": 10000,
    "min_answer_length": 1,
    "max_answer_length": 100,
    "min_question_length": 10,
    "max_question_length": 500,
}

# Version information
VERSION_INFO: Final[dict] = {
    "major": 0,
    "minor": 1,
    "patch": 0,
    "pre_release": None,
    "build": None,
}


# Build version string
def _build_version() -> str:
    """Build version string from version info."""
    version = f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['patch']}"
    if VERSION_INFO["pre_release"]:
        version += f"-{VERSION_INFO['pre_release']}"
    if VERSION_INFO["build"]:
        version += f"+{VERSION_INFO['build']}"
    return version


VERSION: Final[str] = _build_version()
