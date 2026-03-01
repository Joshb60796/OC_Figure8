# AGENTS.md

This file contains guidelines and commands for agentic coding agents operating in this repository.

## Build/Lint/Test Commands

### Running Tests
- **No specific test commands found** - This repository doesn't appear to have formal test infrastructure
- To run code checks: `python -m pyflakes core.py gui_interactive.py`
- To run type checks: `python -m mypy core.py` (if mypy is installed)
- For linting: `python -m pycodestyle core.py gui_interactive.py`

### Build Commands
- This appears to be a Python library with no build process required
- To run the main application: `python gui_interactive.py`

### Formatting
- Uses PEP 8 style guidelines
- Python files formatted with standard Python conventions

## Code Style Guidelines

### Imports
- Standard library imports first (import os, sys)
- Third-party imports second (import numpy as np, matplotlib.pyplot as plt)
- Local imports last (from .core import Figure8Curve)

### Naming Conventions
- Class names: PascalCase (e.g., `Figure8Curve`, `C2ClothoidFigure8`)
- Functions and variables: snake_case (e.g., `get_dense_points`, `total_length`)
- Constants: UPPER_CASE (not present in this codebase)

### Type Hints
- All functions use proper type hints for parameters and return values
- Uses typing imports: `Tuple`, `Dict`, `Optional`, `Union`
- Function signatures include type annotations

### Error Handling
- No explicit error handling patterns found in the codebase
- Uses standard Python error propagation

### Documentation
- Docstrings present for classes and major functions
- Docstrings follow Google or NumPy style conventions
- Comments explain complex calculations and algorithms

### Formatting
- 4-space indentation
- Maximum line length of 79 characters (PEP 8)
- Proper spacing around operators and after commas
- Blank lines between functions and classes

### Special Rules
- This codebase uses NumPy for mathematical operations
- Uses matplotlib for visualization
- Follows mathematical documentation style for complex algorithms

### Cursor/Copilot Instructions
- No specific Cursor or Copilot rules detected in the repository
- Follows standard Python development practices

### Additional Guidelines
- Function parameters have default values where appropriate
- Return values are clearly documented in docstrings
- Complex mathematical computations are well-commented