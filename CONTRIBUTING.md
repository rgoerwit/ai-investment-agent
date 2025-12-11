# Contributing to Investment Agent

Thank you for your interest in contributing! This document provides guidelines for contributing to the multi-agent investment analysis system.

## Quick Start

1. **Fork the repository** and clone your fork
2. **Install dependencies**: `poetry install --with dev`
3. **Set up environment**: Copy `.env.example` to `.env` and add your API keys
4. **Run tests**: `poetry run pytest tests/ -v`
5. **Make your changes** with tests
6. **Submit a pull request**

## Development Setup

### Prerequisites

- Python 3.12 or higher
- Poetry for dependency management
- Git for version control

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/investment-agent-public.git
cd investment-agent-public

# Install all dependencies including dev tools
poetry install --with dev

# Configure API keys
cp .env.example .env
# Edit .env and add at minimum:
#   GOOGLE_API_KEY (required - get from Google AI Studio)
#   FINNHUB_API_KEY (required - free tier available)
#   TAVILY_API_KEY (required - free tier available)

# Verify installation
poetry run pytest tests/ -v
```

## Code Quality Standards

All contributions must meet these standards:

### Linting and Formatting

```bash
# Check code style (must pass before PR)
make check-all

# Or run individual checks:
make format-check  # Black formatting
make lint          # Ruff linting
make typecheck     # MyPy type checking
```

### Auto-fix Issues

```bash
make format    # Auto-format with Black
make lint-fix  # Auto-fix linting issues
```

### Testing Requirements

- All new features must include tests
- Test coverage must remain above 80%
- Tests must pass locally before submitting PR

```bash
# Run all tests
poetry run pytest tests/ -v

# Run with coverage report
poetry run pytest --cov=src --cov-report=term-missing tests/

# Run specific test categories
poetry run pytest -m "not slow" tests/        # Skip slow tests
poetry run pytest tests/test_your_feature.py  # Single file
```

### Type Hints

- All public functions must have type hints
- Use `typing` module for complex types
- MyPy must pass with no errors

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature-add-esg-analyst` or `feature/add-esg-analyst` - New features
- `fix-portfolio-manager-output` or `fix/portfolio-manager-output` - Bug fixes
- `docs-api-reference` or `docs/api-reference` - Documentation updates
- `refactor-memory-isolation` or `refactor/memory-isolation` - Code refactoring
- `test-rate-limiting` or `test/rate-limiting` - Test additions

### Commit Messages

Write clear, concise commit messages:

```text
Add ESG analyst for environmental/social/governance scoring

- Created get_esg_metrics() tool with MSCI API integration
- Added esg_analyst.json prompt with scoring criteria
- Registered agent in graph.py parallel data gathering
- Added tests for ESG metric parsing and validation
```

### Code Style

- Follow existing code patterns in the repository
- Use descriptive variable names (`debt_to_equity_ratio` not `de`)
- Add docstrings to all public functions
- Keep functions focused (single responsibility)
- Prefer explicit over implicit (readability > cleverness)

## Pull Request Process

### Before Submitting

1. **Run all quality checks**: `make check-all`
2. **Run tests**: `poetry run pytest tests/ -v`
3. **Update documentation** if you changed functionality
4. **Add tests** for new features
5. **Update CHANGELOG.md** under `[Unreleased]` section

### PR Description Template

```markdown
## Summary
Brief description of what this PR does.

## Changes
- Bullet list of specific changes
- Include file paths for major changes

## Testing
How to test this change:
1. Step-by-step instructions
2. Expected output/behavior

## Related Issues
Fixes #123 (if applicable)
```

### Review Process

- PRs are typically reviewed within 1-2 weeks
- Security-related PRs are prioritized
- Maintainers may request changes before merging
- All CI checks must pass (tests, linting, security scans)
- At least one approval required from maintainers

## Areas for Contribution

We welcome contributions in these areas:

### 🔴 High Priority

1. **Data Sources** - Integrate additional providers (Alpha Vantage, Polygon.io, IEX Cloud)
2. **Documentation** - Tutorials, examples, architecture guides
3. **Testing** - Increase coverage, add edge cases
4. **Bug Fixes** - See [Issues](https://github.com/rgoerwit/investment-agent-public/issues) for open bugs

### 🟡 Medium Priority

1. **New Agents** - ESG analyst, options analyst, macro analyst
2. **Sentiment Analysis** - Twitter/X API, Reddit scraping, news NLP
3. **UI** - Streamlit/Gradio frontend for non-technical users
4. **Performance** - Caching, parallel processing, batch optimizations

### 🟢 Nice to Have

1. **Backtesting** - Historical performance simulation
2. **Execution** - Interactive Brokers API integration
3. **Alerting** - Email/Slack notifications for thesis changes
4. **Mobile** - React Native or Progressive Web App

## Testing Guidelines

### Writing Tests

Place tests in `tests/` directory with pattern `test_*.py`:

```python
# tests/test_new_feature.py
import pytest
from src.your_module import your_function

def test_your_function_basic_case():
    """Test basic functionality."""
    result = your_function("input")
    assert result == "expected_output"

def test_your_function_edge_case():
    """Test edge case handling."""
    with pytest.raises(ValueError):
        your_function(None)

@pytest.mark.slow
def test_your_function_integration():
    """Test with real API (slow test)."""
    # Integration test with external dependencies
    pass
```

### Test Markers

Use pytest markers for test categorization:

- `@pytest.mark.slow` - Tests taking >5 seconds
- `@pytest.mark.integration` - Tests requiring network/APIs
- `@pytest.mark.memory` - Tests requiring ChromaDB

## Documentation

### Code Documentation

- Add docstrings to all public functions/classes
- Use Google-style docstrings
- Include parameter types, return types, and examples

```python
def calculate_position_size(
    conviction: float,
    financial_health: float,
    growth_score: float
) -> float:
    """
    Calculate position size based on thesis scores.

    Args:
        conviction: Portfolio Manager conviction level (0-100)
        financial_health: Financial health score (0-100)
        growth_score: Growth transition score (0-100)

    Returns:
        Position size as percentage of portfolio (0-5.0)

    Example:
        >>> calculate_position_size(85, 75, 60)
        3.5
    """
    # Implementation
```

### README and Guides

- Update README.md if you change user-facing functionality
- Add examples to `examples/` directory
- Update CLAUDE.md if you change architecture

## Community Guidelines

### Code of Conduct

- Be respectful and inclusive
- Provide constructive feedback
- Focus on what is best for the community
- Show empathy towards other contributors

### Getting Help

- **Questions**: Open a [Discussion](https://github.com/rgoerwit/investment-agent-public/discussions)
- **Bugs**: Open an [Issue](https://github.com/rgoerwit/investment-agent-public/issues)
- **Security**: See [SECURITY.md](SECURITY.md) for vulnerability reporting

## License

By contributing, you agree that your contributions will be licensed under the MIT License. See [LICENSE](LICENSE) file for details.

---

## Thank You

Your contributions help democratize sophisticated financial analysis. Every bug fix, feature addition, and documentation improvement makes this tool more accessible to retail investors worldwide.

**Questions?** Feel free to open a discussion or reach out via [Issues](https://github.com/rgoerwit/investment-agent-public/issues).
