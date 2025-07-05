"""
Setup script for the Agentic AI Mathematical Equation Solving project
"""

from setuptools import setup, find_packages

# Read requirements from file if it exists
try:
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
except FileNotFoundError:
    requirements = [
        'datasets>=2.0.0',
        'tqdm>=4.65.0',
        'smolagents>=0.1.0',
        'transformers>=4.30.0',
        'torch>=2.0.0',
        'numpy>=1.21.0',
        'sympy>=1.12.0',
        'scipy>=1.9.0',
    ]

setup(
    name="math-agent",
    version="0.1.0",
    description="Agentic AI for Mathematical Equation Solving",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=requirements,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    entry_points={
        'console_scripts': [
            'math-agent=math_agent.code_eval.math_code_agent:main',
        ],
    },
) 