"""Configuration management for the math agent package."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from .exceptions import ConfigurationError


@dataclass
class ModelConfig:
    """Configuration for model settings."""

    name: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    use_local: bool = True
    use_mlx: Optional[bool] = None
    temperature: float = 0.1
    max_tokens: int = 2048
    max_new_tokens: int = 512
    top_p: float = 0.85
    repetition_penalty: float = 1.1
    device_map: str = "auto"
    torch_dtype: str = "auto"


@dataclass
class LoggingConfig:
    """Configuration for logging settings."""

    level: str = "INFO"
    format_type: str = "standard"
    log_file: Optional[str] = None
    console_output: bool = True
    json_logging: bool = False
    log_dir: str = "logs"


@dataclass
class EvaluationConfig:
    """Configuration for evaluation settings."""

    max_problems: Optional[int] = None
    categories: Optional[List[str]] = None
    max_retries: int = 3
    timeout_seconds: int = 300
    verbose: bool = True
    save_results: bool = True
    results_dir: str = "results"


@dataclass
class DataConfig:
    """Configuration for data settings."""

    dataset_path: str = "data/processed/math_qa_dataset"
    cache_dir: str = "data/cache"
    batch_size: int = 32
    num_workers: int = 4
    preprocessing_cache: bool = True


@dataclass
class AgentConfig:
    """Configuration for agent settings."""

    system_prompt_template: str = ""
    max_code_execution_time: int = 30
    allowed_imports: List[str] = field(
        default_factory=lambda: [
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
    )
    verbosity_level: int = 1


@dataclass
class MathAgentConfig:
    """Main configuration class for the math agent."""

    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)

    # Global settings
    project_root: str = "."
    random_seed: int = 42
    debug: bool = False

    @classmethod
    def from_env(cls) -> "MathAgentConfig":
        """Create configuration from environment variables."""
        config = cls()

        # Model configuration
        config.model.name = os.getenv("MATH_AGENT_MODEL_NAME", config.model.name)
        config.model.use_local = (
            os.getenv("MATH_AGENT_USE_LOCAL", "true").lower() == "true"
        )
        config.model.temperature = float(
            os.getenv("MATH_AGENT_TEMPERATURE", str(config.model.temperature))
        )
        config.model.max_tokens = int(
            os.getenv("MATH_AGENT_MAX_TOKENS", str(config.model.max_tokens))
        )
        config.model.max_new_tokens = int(
            os.getenv("MATH_AGENT_MAX_NEW_TOKENS", str(config.model.max_new_tokens))
        )
        config.model.top_p = float(
            os.getenv("MATH_AGENT_TOP_P", str(config.model.top_p))
        )
        config.model.repetition_penalty = float(
            os.getenv(
                "MATH_AGENT_REPETITION_PENALTY", str(config.model.repetition_penalty)
            )
        )

        # Logging configuration
        config.logging.level = os.getenv("MATH_AGENT_LOG_LEVEL", config.logging.level)
        config.logging.format_type = os.getenv(
            "MATH_AGENT_LOG_FORMAT", config.logging.format_type
        )
        config.logging.log_file = os.getenv(
            "MATH_AGENT_LOG_FILE", config.logging.log_file
        )
        config.logging.console_output = (
            os.getenv("MATH_AGENT_LOG_CONSOLE", "true").lower() == "true"
        )
        config.logging.json_logging = (
            os.getenv("MATH_AGENT_LOG_JSON", "false").lower() == "true"
        )
        config.logging.log_dir = os.getenv("MATH_AGENT_LOG_DIR", config.logging.log_dir)

        # Evaluation configuration
        max_problems_str = os.getenv("MATH_AGENT_MAX_PROBLEMS")
        if max_problems_str:
            config.evaluation.max_problems = int(max_problems_str)

        categories_str = os.getenv("MATH_AGENT_CATEGORIES")
        if categories_str:
            config.evaluation.categories = [
                cat.strip() for cat in categories_str.split(",")
            ]

        config.evaluation.max_retries = int(
            os.getenv("MATH_AGENT_MAX_RETRIES", str(config.evaluation.max_retries))
        )
        config.evaluation.timeout_seconds = int(
            os.getenv("MATH_AGENT_TIMEOUT", str(config.evaluation.timeout_seconds))
        )
        config.evaluation.verbose = (
            os.getenv("MATH_AGENT_VERBOSE", "true").lower() == "true"
        )
        config.evaluation.save_results = (
            os.getenv("MATH_AGENT_SAVE_RESULTS", "true").lower() == "true"
        )
        config.evaluation.results_dir = os.getenv(
            "MATH_AGENT_RESULTS_DIR", config.evaluation.results_dir
        )

        # Data configuration
        config.data.dataset_path = os.getenv(
            "MATH_AGENT_DATASET_PATH", config.data.dataset_path
        )
        config.data.cache_dir = os.getenv("MATH_AGENT_CACHE_DIR", config.data.cache_dir)
        config.data.batch_size = int(
            os.getenv("MATH_AGENT_BATCH_SIZE", str(config.data.batch_size))
        )
        config.data.num_workers = int(
            os.getenv("MATH_AGENT_NUM_WORKERS", str(config.data.num_workers))
        )
        config.data.preprocessing_cache = (
            os.getenv("MATH_AGENT_PREPROCESSING_CACHE", "true").lower() == "true"
        )

        # Agent configuration
        config.agent.max_code_execution_time = int(
            os.getenv(
                "MATH_AGENT_CODE_TIMEOUT", str(config.agent.max_code_execution_time)
            )
        )
        config.agent.verbosity_level = int(
            os.getenv("MATH_AGENT_VERBOSITY", str(config.agent.verbosity_level))
        )

        # Global settings
        config.project_root = os.getenv("MATH_AGENT_PROJECT_ROOT", config.project_root)
        config.random_seed = int(
            os.getenv("MATH_AGENT_RANDOM_SEED", str(config.random_seed))
        )
        config.debug = os.getenv("MATH_AGENT_DEBUG", "false").lower() == "true"

        return config

    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> "MathAgentConfig":
        """Load configuration from a file."""
        config_path = Path(config_path)

        if not config_path.exists():
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        if config_path.suffix == ".json":
            import json

            with open(config_path, "r") as f:
                data = json.load(f)
        elif config_path.suffix in [".yaml", ".yml"]:
            try:
                import yaml

                with open(config_path, "r") as f:
                    data = yaml.safe_load(f)
            except ImportError:
                raise ConfigurationError(
                    "PyYAML is required for YAML configuration files"
                )
        else:
            raise ConfigurationError(
                f"Unsupported configuration file format: {config_path.suffix}"
            )

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MathAgentConfig":
        """Create configuration from a dictionary."""
        config = cls()

        # Update model config
        if "model" in data:
            for key, value in data["model"].items():
                if hasattr(config.model, key):
                    setattr(config.model, key, value)

        # Update logging config
        if "logging" in data:
            for key, value in data["logging"].items():
                if hasattr(config.logging, key):
                    setattr(config.logging, key, value)

        # Update evaluation config
        if "evaluation" in data:
            for key, value in data["evaluation"].items():
                if hasattr(config.evaluation, key):
                    setattr(config.evaluation, key, value)

        # Update data config
        if "data" in data:
            for key, value in data["data"].items():
                if hasattr(config.data, key):
                    setattr(config.data, key, value)

        # Update agent config
        if "agent" in data:
            for key, value in data["agent"].items():
                if hasattr(config.agent, key):
                    setattr(config.agent, key, value)

        # Update global settings
        for key in ["project_root", "random_seed", "debug"]:
            if key in data:
                setattr(config, key, data[key])

        return config

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "model": {
                "name": self.model.name,
                "use_local": self.model.use_local,
                "use_mlx": self.model.use_mlx,
                "temperature": self.model.temperature,
                "max_tokens": self.model.max_tokens,
                "max_new_tokens": self.model.max_new_tokens,
                "top_p": self.model.top_p,
                "repetition_penalty": self.model.repetition_penalty,
                "device_map": self.model.device_map,
                "torch_dtype": self.model.torch_dtype,
            },
            "logging": {
                "level": self.logging.level,
                "format_type": self.logging.format_type,
                "log_file": self.logging.log_file,
                "console_output": self.logging.console_output,
                "json_logging": self.logging.json_logging,
                "log_dir": self.logging.log_dir,
            },
            "evaluation": {
                "max_problems": self.evaluation.max_problems,
                "categories": self.evaluation.categories,
                "max_retries": self.evaluation.max_retries,
                "timeout_seconds": self.evaluation.timeout_seconds,
                "verbose": self.evaluation.verbose,
                "save_results": self.evaluation.save_results,
                "results_dir": self.evaluation.results_dir,
            },
            "data": {
                "dataset_path": self.data.dataset_path,
                "cache_dir": self.data.cache_dir,
                "batch_size": self.data.batch_size,
                "num_workers": self.data.num_workers,
                "preprocessing_cache": self.data.preprocessing_cache,
            },
            "agent": {
                "system_prompt_template": self.agent.system_prompt_template,
                "max_code_execution_time": self.agent.max_code_execution_time,
                "allowed_imports": self.agent.allowed_imports,
                "verbosity_level": self.agent.verbosity_level,
            },
            "project_root": self.project_root,
            "random_seed": self.random_seed,
            "debug": self.debug,
        }

    def save(self, config_path: Union[str, Path]) -> None:
        """Save configuration to a file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.to_dict()

        if config_path.suffix == ".json":
            import json

            with open(config_path, "w") as f:
                json.dump(data, f, indent=2)
        elif config_path.suffix in [".yaml", ".yml"]:
            try:
                import yaml

                with open(config_path, "w") as f:
                    yaml.dump(data, f, default_flow_style=False, indent=2)
            except ImportError:
                raise ConfigurationError(
                    "PyYAML is required for YAML configuration files"
                )
        else:
            raise ConfigurationError(
                f"Unsupported configuration file format: {config_path.suffix}"
            )

    def validate(self) -> None:
        """Validate the configuration."""
        errors = []

        # Validate model configuration
        if self.model.temperature < 0 or self.model.temperature > 2:
            errors.append("Model temperature must be between 0 and 2")

        if self.model.max_tokens <= 0:
            errors.append("Model max_tokens must be positive")

        if self.model.max_new_tokens <= 0:
            errors.append("Model max_new_tokens must be positive")

        if self.model.top_p < 0 or self.model.top_p > 1:
            errors.append("Model top_p must be between 0 and 1")

        # Validate logging configuration
        if self.logging.level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            errors.append(f"Invalid logging level: {self.logging.level}")

        if self.logging.format_type not in ["simple", "standard", "detailed"]:
            errors.append(f"Invalid logging format type: {self.logging.format_type}")

        # Validate evaluation configuration
        if (
            self.evaluation.max_problems is not None
            and self.evaluation.max_problems <= 0
        ):
            errors.append("Evaluation max_problems must be positive")

        if self.evaluation.max_retries < 0:
            errors.append("Evaluation max_retries must be non-negative")

        if self.evaluation.timeout_seconds <= 0:
            errors.append("Evaluation timeout_seconds must be positive")

        # Validate data configuration
        if self.data.batch_size <= 0:
            errors.append("Data batch_size must be positive")

        if self.data.num_workers < 0:
            errors.append("Data num_workers must be non-negative")

        # Validate agent configuration
        if self.agent.max_code_execution_time <= 0:
            errors.append("Agent max_code_execution_time must be positive")

        if self.agent.verbosity_level < 0:
            errors.append("Agent verbosity_level must be non-negative")

        # Validate global settings
        if self.random_seed < 0:
            errors.append("Random seed must be non-negative")

        if errors:
            raise ConfigurationError(
                f"Configuration validation failed: {'; '.join(errors)}"
            )

    def setup_directories(self) -> None:
        """Create necessary directories."""
        directories = [
            self.logging.log_dir,
            self.evaluation.results_dir,
            self.data.cache_dir,
        ]

        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)


# Global configuration instance
_config: Optional[MathAgentConfig] = None


def get_config() -> MathAgentConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = MathAgentConfig.from_env()
        _config.validate()
        _config.setup_directories()
    return _config


def set_config(config: MathAgentConfig) -> None:
    """Set the global configuration instance."""
    global _config
    config.validate()
    config.setup_directories()
    _config = config


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config
    _config = None
