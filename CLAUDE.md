# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Environment Setup
- **Install dependencies**: `poetry install`
- **Activate virtual environment**: `source .venv/bin/activate` (if using poetry with in-project venv)
- **Install pre-commit hooks**: `poetry run pre-commit install`

### Testing
- **Run unit tests only**: `pytest code_tests/unit_tests/`
- **Run low-cost/live API tests**: `pytest code_tests/low_cost_or_live_api_tests/`
- **Run expensive tests** (avoid unless necessary): `pytest code_tests/expensive_tests__run_individually/`
- **Run specific test**: `pytest code_tests/unit_tests/test_specific_file.py::test_function_name`
- **Run tests with parallel execution**: `pytest -nauto` (already configured in pytest.ini)

### Development Server
- **Run Streamlit frontend**: `streamlit run front_end/Home.py`
- **Run benchmark displayer**: `streamlit run forecasting_tools/benchmarking/benchmark_displayer.py`

### Scripts
- **Run benchmarker**: `python scripts/run_benchmarker.py`
- **Run question snapshots**: `python scripts/run_question_snapshots.py`
- **Run higher model evaluation**: `python scripts/run_higher_model_evaluation.py`

## Code Architecture

### Core Components

**Forecast Bots** (`forecasting_tools/forecast_bots/`):
- `ForecastBot` - Abstract base class for all forecasting bots
- `TemplateBot` - Inherits from `Q2TemplateBot2025`, main bot template for customization
- `MainBot` - Production-ready bot with advanced features
- Official quarterly bots: `Q1TemplateBot2025`, `Q2TemplateBot2025`, `Q3TemplateBot2024`, `Q4TemplateBot2024`

**AI Models** (`forecasting_tools/ai_models/`):
- `GeneralLlm` - Main wrapper around litellm for any AI model with retry logic, cost tracking, and structured outputs
- `SmartSearcher` - AI-powered internet search using Exa.ai, similar to Perplexity but more configurable
- `ExaSearcher` - Direct interface to Exa.ai search API
- Model interfaces provide common patterns (TokensIncurCost, RetryableModel, OutputsText, etc.)

**Research Tools** (`forecasting_tools/agents_and_tools/`):
- `KeyFactorsResearcher` - Identifies and scores key factors for forecasting questions
- `BaseRateResearcher` - Calculates historical base rates for events
- `NicheListResearcher` - Analyzes specific lists of events with fact-checking
- `Estimator` - Fermi estimation for numerical predictions

**Data Models** (`forecasting_tools/data_models/`):
- `MetaculusQuestion` - Base question class with subclasses for Binary, MultipleChoice, Numeric, Date
- `ForecastReport` - Contains question, prediction, reasoning, and metadata
- Question-specific reports: `BinaryReport`, `MultipleChoiceReport`, `NumericReport`

**Metaculus Integration** (`forecasting_tools/forecast_helpers/`):
- `MetaculusApi` - API wrapper for questions, tournaments, predictions, and comments
- `ApiFilter` - Filtering class for querying questions
- Tournament IDs available as class constants (e.g., `CURRENT_QUARTERLY_CUP_ID`)

**Benchmarking** (`forecasting_tools/benchmarking/`):
- `Benchmarker` - Runs comparative evaluations between bots
- `BenchmarkForBot` - Contains benchmark results and statistics
- `PromptOptimizer` - Iteratively improves bot prompts

### Bot Customization Pattern

The standard approach is to inherit from `TemplateBot` and override specific methods:
- `run_research()` - Custom research approach
- `_run_forecast_on_binary()` - Binary question forecasting
- `_run_forecast_on_multiple_choice()` - Multiple choice forecasting
- `_run_forecast_on_numeric()` - Numeric forecasting
- `_initialize_notepad()` - Maintain state between forecasts

### Cost Management

Use `MonetaryCostManager` as a context manager to track and limit API costs:
```python
with MonetaryCostManager(max_cost=5.00) as cost_manager:
    # API calls here
    current_cost = cost_manager.current_usage
```

## Key Environment Variables

Required for basic functionality:
- `OPENAI_API_KEY` - For OpenAI models
- `EXA_API_KEY` - For web search functionality
- `METACULUS_TOKEN` - For posting predictions/comments to Metaculus

Optional but commonly used:
- `ANTHROPIC_API_KEY` - For Claude models
- `PERPLEXITY_API_KEY` - Alternative to Exa for search
- `FILE_WRITING_ALLOWED` - Set to "TRUE" to enable log file writing

## Project Structure Notes

- Main entry point: `forecasting_tools/__init__.py` with comprehensive exports
- Tests are organized by cost: unit_tests (free), low_cost_or_live_api, expensive
- Frontend uses Streamlit with pages in `front_end/app_pages/`
- Logs are written to `logs/` directory with rotating files
- Scripts for common workflows are in `scripts/`
- Dev container configuration in `.devcontainer/` for consistent environments

## Model Integration

The project uses litellm for unified model access. Models can be specified as:
- Direct: `"gpt-4o"`, `"claude-3-5-sonnet-20241022"`
- With provider: `"openai/gpt-4o"`, `"anthropic/claude-3-5-sonnet"`
- Via Metaculus proxy: `"metaculus/claude-3-5-sonnet-20241022"`

All models support structured outputs via `invoke_and_return_verified_type()` with Pydantic validation.
