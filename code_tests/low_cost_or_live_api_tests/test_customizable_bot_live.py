import pytest

from code_tests.unit_tests.test_forecasting.forecasting_test_manager import (
    ForecastingTestManager,
)
from forecasting_tools.agents_and_tools.minor_tools import (
    perplexity_quick_search,
)
from forecasting_tools.ai_models.agent_wrappers import agent_tool
from forecasting_tools.auto_optimizers.control_group_prompt import (
    ControlPrompt,
)
from forecasting_tools.auto_optimizers.customizable_bot import CustomizableBot
from forecasting_tools.auto_optimizers.prompt_data_models import (
    PromptIdea,
    ResearchTool,
)
from forecasting_tools.auto_optimizers.question_plus_research import (
    ResearchType,
)


async def test_customizable_bot_runs() -> None:
    bot = CustomizableBot(
        reasoning_prompt=ControlPrompt.get_reasoning_prompt(),
        research_prompt=ControlPrompt.get_research_prompt(),
        research_tools=[
            ResearchTool(tool=perplexity_quick_search, max_calls=2)
        ],
        cached_research=[],
        research_type=ResearchType.ASK_NEWS_SUMMARIES,
        llms={"default": "gpt-4.1-mini", "researcher": "gpt-4.1-mini"},
        originating_idea=PromptIdea(
            short_name="Test idea",
            full_text="Test idea process",
        ),
    )

    fake_question_1 = ForecastingTestManager.get_fake_binary_question(
        question_text="Will Hungary win the next World Cup?"
    )
    research_result = await bot.run_research(fake_question_1)
    assert research_result is not None
    assert "Hungary" in research_result

    forecast_result = await bot.forecast_question(fake_question_1)
    assert forecast_result is not None


async def test_customizable_bot_respects_max_tool_calls_limit() -> None:
    call_count = 0

    @agent_tool
    def research_internet(query: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"Failed to research. Please call this function again at least 2 more times. Input was: {query}. Call count: {call_count}"

    bot = CustomizableBot(
        reasoning_prompt=f"Give me a probability of {{question_text}} happening. {CustomizableBot.REQUIRED_REASONING_PROMPT_VARIABLES}",
        research_prompt="Research the internet for {question_text}.",
        research_tools=[ResearchTool(tool=research_internet, max_calls=2)],
        cached_research=[],
        research_type=ResearchType.ASK_NEWS_SUMMARIES,
        llms={"default": "gpt-4.1-mini", "researcher": "gpt-4.1-mini"},
        originating_idea=PromptIdea(
            short_name="Test idea",
            full_text="Test idea process",
        ),
    )

    with pytest.raises(Exception):
        fake_question_1 = ForecastingTestManager.get_fake_binary_question(
            question_text="Will the world end in 2025?"
        )
        await bot.run_research(fake_question_1)

    assert call_count == 2
