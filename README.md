# Agentic-AI-for-Mathematical-Equation-Solving
A self-correcting AI agent that solves mathematical problems using LLMs and symbolic reasoning.

---

### Overview
This project develops an autonomous AI agent capable of solving mathematical problems ranging from basic arithmetic to advanced calculus. The system combines:

* LLMs for problem understanding

* Symbolic Computation (SymPy for exact solutions)

* Agentic Pipelines (planning, tool use, and self-correction).

---

### Project Structure

```
Agentic-AI-for-Mathematical-Equation-Solving/
│
├── README.md                  # This file
├── data/
│   └── processed/             # Processed datasets 
├── docs/                      # Documentation and proposals
│   ├── steps_idea.md
│   └── AAI_proposal.pdf
├── math_agent/                # Main agent code
│   ├── agent/                 # Agent logic (planner, executor)
│   ├── core/                  # Core modules (parser, solver)
│   ├── data/                  # Data-related scripts/utilities
│   ├── scripts/               # Data processing scripts
│   └── tests/                 # Unit tests
├── mathematics_dataset/       # DeepMind mathematics dataset (submodule or copy)
│   ├── mathematics_dataset/   # Dataset generation code
│   │   ├── generate.py        # Prints questions/answers to stdout
│   │   ├── generate_to_file.py# Writes generated data to files
│   │   ├── modules/           # Math modules (arithmetic, algebra, etc.)
│   │   └── ...
│   ├── setup.py
│   └── README.md
└── ...
```

---

We utilize the Google DeepMind Mathematics Dataset

```
https://github.com/deepmind/mathematics_dataset

```
