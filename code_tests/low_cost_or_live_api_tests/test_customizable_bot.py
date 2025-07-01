import pytest

from code_tests.unit_tests.test_forecasting.forecasting_test_manager import (
    ForecastingTestManager,
)
from forecasting_tools.ai_models.agent_wrappers import agent_tool
from forecasting_tools.benchmarking.customizable_bot import CustomizableBot
from forecasting_tools.benchmarking.prompt_data_models import PromptIdea
from forecasting_tools.benchmarking.question_research_snapshot import (
    ResearchType,
)


async def test_customizable_bot_respects_max_tool_calls_limit() -> None:
    call_count = 0

    @agent_tool
    def research_internet(query: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"Failed to research. Please call this function again. Input was: {query}. Call count: {call_count}"

    bot = CustomizableBot(
        reasoning_prompt="Give me a probability of {question_text} happening.",
        research_prompt="Research the internet for {question_text}.",
        research_tools=[research_internet],
        research_snapshots=[],
        research_type=ResearchType.ASK_NEWS_SUMMARIES,
        llms={"default": "gpt-4.1-mini", "researcher": "gpt-4.1-mini"},
        originating_idea=PromptIdea(
            short_name="Test idea",
            idea="Test idea process",
        ),
        max_tool_calls_per_research=2,
    )

    with pytest.raises(RuntimeError):
        fake_question_1 = ForecastingTestManager.get_fake_binary_question(
            question_text="Q1"
        )
        await bot.run_research(fake_question_1)

    assert call_count == 2
