# This file is run before any tests are run in order to configure tests

from typing import Generator

import dotenv
import pytest

from forecasting_tools.ai_models.agent_wrappers import general_trace_or_span
from forecasting_tools.util.custom_logger import CustomLogger


@pytest.fixture(scope="session", autouse=True)
def setup_logging() -> None:
    dotenv.load_dotenv()
    CustomLogger.setup_logging()


@pytest.fixture(scope="session", autouse=True)
def wrap_all_tests_in_trace() -> Generator[None, None, None]:
    with general_trace_or_span("Running tests"):
        yield  # Run all tests
