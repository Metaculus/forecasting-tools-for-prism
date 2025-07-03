from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from code_tests.unit_tests.test_forecasting.forecasting_test_manager import (
    ForecastingTestManager,
)
from forecasting_tools.ai_models.agent_wrappers import agent_tool
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.benchmarking.customizable_bot import (
    CustomizableBot,
    ResearchTool,
)
from forecasting_tools.benchmarking.prompt_data_models import PromptIdea
from forecasting_tools.benchmarking.question_plus_research import (
    QuestionPlusResearch,
    ResearchItem,
    ResearchType,
)

fake_question_1 = ForecastingTestManager.get_fake_binary_question(
    question_text="Q1"
)
fake_question_2 = ForecastingTestManager.get_fake_binary_question(
    question_text="Q2"
)


@pytest.fixture
def mock_llm() -> MagicMock:
    llm = MagicMock(spec=GeneralLlm)
    llm.invoke = AsyncMock(return_value="Probability: 50%")
    return llm


@agent_tool
def _fake_tool(input: str) -> str:
    """
    Mock tool that returns a research result
    """
    return f"Mock tool output. Input was: {input}"


mock_tool = ResearchTool(tool=_fake_tool, max_calls=2)


@pytest.fixture
def research_snapshots() -> list[QuestionPlusResearch]:
    return [
        QuestionPlusResearch(
            question=fake_question_1,
            research_items=[
                ResearchItem(
                    research="q1_news", type=ResearchType.ASK_NEWS_SUMMARIES
                )
            ],
        ),
        QuestionPlusResearch(
            question=fake_question_2,
            research_items=[
                ResearchItem(
                    research="q2_news", type=ResearchType.ASK_NEWS_SUMMARIES
                )
            ],
        ),
    ]


@pytest.fixture
def customizable_bot(
    mock_llm: MagicMock,
    research_snapshots: list[QuestionPlusResearch],
) -> CustomizableBot:
    return CustomizableBot(
        reasoning_prompt="Test prompt with {question_text} and {research}",
        research_prompt="Research prompt for {question_text}",
        research_tools=[mock_tool],
        cached_research=research_snapshots,
        research_type=ResearchType.ASK_NEWS_SUMMARIES,
        llms={"default": mock_llm, "researcher": "gpt-4o-mini"},
        originating_idea=PromptIdea(
            short_name="Test idea",
            full_text="Test idea process",
        ),
    )


async def test_customizable_bot_run_research_success(
    customizable_bot: CustomizableBot,
) -> None:
    question = fake_question_1
    research = await customizable_bot.run_research(question)
    assert research == "q1_news"


async def test_customizable_bot_run_research_question_not_found(
    customizable_bot: CustomizableBot,
) -> None:
    question = ForecastingTestManager.get_fake_binary_question(
        question_text="Q3 - not in snapshots"
    )

    with patch.object(
        customizable_bot,
        customizable_bot._run_research_with_tools.__name__,
        new_callable=AsyncMock,
    ) as mock_run_research:
        mock_run_research.return_value = "Research result from tools"

        result = await customizable_bot.run_research(question)

        assert result == "Research result from tools"
        mock_run_research.assert_called_once_with(question)


async def test_customizable_bot_run_research_duplicate_questions_in_snapshots(
    mock_llm: MagicMock,
) -> None:
    q1 = ForecastingTestManager.get_fake_binary_question(question_text="Q1")
    snapshots = [
        QuestionPlusResearch(
            question=q1,
            research_items=[
                ResearchItem(
                    research="q1_news1", type=ResearchType.ASK_NEWS_SUMMARIES
                )
            ],
        ),
        QuestionPlusResearch(
            question=q1,  # Duplicate question
            research_items=[
                ResearchItem(
                    research="q1_news2", type=ResearchType.ASK_NEWS_SUMMARIES
                )
            ],
        ),
    ]
    with pytest.raises(ValueError):
        CustomizableBot(
            reasoning_prompt="Test prompt",
            research_prompt="Research prompt",
            research_tools=[],
            cached_research=snapshots,
            research_type=ResearchType.ASK_NEWS_SUMMARIES,
            llms={"default": mock_llm},
            originating_idea=PromptIdea(
                short_name="Test idea",
                full_text="Test idea process",
            ),
        )


async def test_customizable_bot_raises_error_when_no_researcher_llm_configured(
    mock_llm: MagicMock,
    research_snapshots: list[QuestionPlusResearch],
) -> None:
    bot = CustomizableBot(
        reasoning_prompt="Test prompt",
        research_prompt="Research prompt",
        research_tools=[mock_tool],
        cached_research=research_snapshots,
        research_type=ResearchType.ASK_NEWS_SUMMARIES,
        llms={"default": mock_llm},
        originating_idea=PromptIdea(
            short_name="Test idea",
            full_text="Test idea process",
        ),
    )

    question = ForecastingTestManager.get_fake_binary_question(
        question_text="Q3 - not in snapshots"
    )

    with pytest.raises(ValueError, match="LLM is undefined"):
        await bot.run_research(question)
