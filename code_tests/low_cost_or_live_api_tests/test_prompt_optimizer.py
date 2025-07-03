import logging

from code_tests.unit_tests.test_forecasting.forecasting_test_manager import (
    ForecastingTestManager,
)
from forecasting_tools.agents_and_tools.misc_tools import query_asknews
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.benchmarking.bot_optimizer import BotOptimizer
from forecasting_tools.benchmarking.prompt_data_models import ResearchTool

logger = logging.getLogger(__name__)


async def test_prompt_optimizer() -> None:
    question = ForecastingTestManager.get_fake_binary_question()
    with MonetaryCostManager(1) as cost_manager:
        optimized_result = await BotOptimizer.optimize_a_combined_research_and_reasoning_prompt(
            questions=[question],
            research_tools=[ResearchTool(tool=query_asknews, max_calls=1)],
            research_agent_llm=GeneralLlm(model="gpt-4.1-nano"),
            reasoning_llm=GeneralLlm(model="openrouter/openai/gpt-4.1-nano"),
            questions_batch_size=1,
            num_iterations_per_run=2,
            ideation_llm="o4-mini",
            remove_background_info=True,
        )
        logger.info(f"Cost: {cost_manager.current_usage}")
    assert optimized_result is not None
    assert optimized_result.best_prompt.prompt.text is not None
