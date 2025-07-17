# Agentic-AI-for-Mathematical-Equation-Solving
A self-correcting AI agent that solves mathematical problems using LLMs, symbolic reasoning, and code execution. The system is designed for robust, extensible, and interpretable mathematical problem solving across a wide range of topics.

---

## Overview
This project develops an autonomous AI agent capable of solving mathematical problems ranging from basic arithmetic to advanced calculus. The system combines:

- **Large Language Models (LLMs)** for problem understanding and code generation
- **Symbolic Computation** (SymPy for exact solutions)
- **Agentic Pipelines** (planning, tool use, and self-correction)
- **Safe Code Execution** in a sandboxed environment
- **Evaluation and Feedback** for self-correction and improvement

### Agentic Pipeline Stages
1. **Problem Interpretation**: Dual-channel input (text/math), domain classification
2. **Solution Planning**: Prompt engineering, tool selection, problem decomposition
3. **Execution**: Code generation and execution, resource management
4. **Verification**: Output normalization, symbolic/numeric equivalence, error analysis

### Main Features
- **Multiple Problem Types**: Arithmetic, algebra, calculus, sorting, comparison, number theory, polynomials, probability, and more
- **Flexible Input**: Accepts both natural language and mathematical notation
- **Extensible**: Easy to add new problem types, tools, and evaluation strategies
- **Evaluation Framework**: Built-in tools for testing on custom and standard datasets
- **Device Adaptation**: Supports CPU, CUDA, and Apple Silicon (MPS/MLX)

### Supported Problem Types
- Arithmetic operations (add, subtract, multiply, divide, mixed)
- Algebra (linear, polynomial, variable isolation)
- Sorting and comparison (sort, kth largest, pairwise, etc.)
- Number theory (GCD, LCM, primes, factorization, base conversion)
- Calculus (differentiation, basic integration)
- Polynomials (evaluation, expansion, composition)
- Probability and measurement

---

## Dataset
We utilize the Google DeepMind Mathematics Dataset as the foundation for training and evaluation:
https://github.com/deepmind/mathematics_dataset

### The `data/math_qa` Folder
This folder contains a curated and preprocessed version of the DeepMind dataset, organized for agentic evaluation and benchmarking. It is structured as follows:

- **interpolate/**: Standard problems for interpolation (in-distribution). Each file corresponds to a specific mathematical topic (e.g., `arithmetic__add_or_sub.txt`, `algebra__linear_1d.txt`).
- **extrapolate/**: Out-of-distribution or more challenging problems for extrapolation/generalization. File naming mirrors `interpolate/` but with more complex or larger-scale problems (e.g., `arithmetic__add_or_sub_big.txt`).
- **train-easy/**, **train-medium/**, **train-hard/**: Training splits of increasing difficulty, each containing the same set of topics as `interpolate/`, but with problems of varying complexity.
- **interpolate.zip**, **extrapolate.zip**, **train-easy.zip**, **train-medium.zip**, **train-hard.zip**: Compressed versions of the above folders for convenience.

#### File Structure and Format
- Each `.txt` file in these folders contains a sequence of problems and answers, alternating line by line:
  ```
  <problem_1>
  <answer_1>
  <problem_2>
  <answer_2>
  ...
  ```
- Problems are presented in natural language or mathematical notation. Answers are typically a number, expression, or list, depending on the topic.
- File names follow the pattern `<category>__<topic>[__modifier].txt`, where:
  - `category` is the broad math area (e.g., `arithmetic`, `algebra`, `comparison`, `numbers`, `polynomials`, `probability`, `calculus`, `measurement`)
  - `topic` specifies the type of problem (e.g., `add_or_sub`, `linear_1d`, `gcd`, `sort`)
  - `modifier` (optional) indicates special variants (e.g., `composed`, `big`, `longer`)

#### Example (from `interpolate/algebra__linear_1d.txt`):
```
Solve 0 = -135*v - 457*v - 83*v + 86*v + 11780 for v.
20
Solve -5756 + 4209 = -422*j + 5891 + 6488 for j.
33
... (and so on)
```

This structure is consistent across all subfolders, enabling systematic evaluation and benchmarking of the agent's mathematical reasoning capabilities.

---

## Project Structure

```
Agentic-AI-for-Mathematical-Equation-Solving/
│
├── README.md
├── requirements.txt
├── pyproject.toml
├── setup.py
├── Makefile
├── notebooks/
│   ├── evaluation.ipynb  # Visualized evaluation results
├── data/
│   ├── math_qa/                # Main dataset (see Dataset section)
│   │   ├── interpolate/        # In-distribution problems by topic
│   │   ├── extrapolate/        # Out-of-distribution/generalization problems
│   │   ├── train-easy/         # Easy training split
│   │   ├── train-medium/       # Medium training split
│   │   ├── train-hard/         # Hard training split
│   └── processed/
│       ├── math_qa_dataset/    # Arrow format dataset
│       └── math_qa_dataset.parquet
├── baselines/
│   ├── baseline_qwen.py
├── eval_agent/
│   ├── eval_math_agent.py # Evaluation with gemini for code exec 
│   ├── fb_gemini_eval.py # Evaluation with gemini for code exec + fb
│   ├── get_error_statistic.py # Script to get error statistic
├── results/
│   ├── agent_comprehensive_evaluation_results.json  # Results for code exec + fb
│   ├── fb_gemini_eval.json # Results with gemini for code exec + fb
│   ├── code_eval/
│   │   ├── agent_comprehensive_evaluation_results.json
│   │   ├── train_interpolate_results.json  # Results for code exec
│   │   ├── train_extrapolate_results.json  # Results for code exec
│   │   ├── train_easy_results.json    # Results for code exec
│   │   ├── train_medium_results.json   # Results for code exec
│   │   ├── train_hard_results.json    # Results for code exec
│   │   ├── math_eval_easy_results.json   # Results with gemini for code exec 
│   │   ├── math_eval_medium_results.json # Results with gemini for code exec 
│   │   ├── math_eval_hard_results.json  # Results with gemini for code exec 
│   │   ├── math_eval_extrapolate_results.json # Results with gemini for code exec 
│   ├── baseline/
│   │   ├── baseline_results.csv - results for baseline
├── math_agent/
│   ├── __init__.py
│   ├── cli.py
│   ├── baseline/
│   ├── code_eval/
│   │   ├── math_tools.py
│   │   ├── solve_with_tools.py
│   ├── code_exec/
│   │   ├── math_code_agent.py
│   │   ├── README.md
│   ├── code_exec_eval/
│   │   ├── feedback_code_agent.py
│   ├── core/
│   │   ├── config.py
│   │   ├── constants.py
│   │   ├── exceptions.py
│   │   └── logging_config.py
│   ├── utils/
│   │   ├── commons.py
├── mathematics_dataset/        # (DeepMind dataset, may be empty or submodule)
├── docs/
│   ├── steps_idea.md
│   └── AAI_proposal.pdf                  
```

---
