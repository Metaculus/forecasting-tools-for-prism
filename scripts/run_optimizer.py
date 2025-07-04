import asyncio
import logging

from forecasting_tools.agents_and_tools.minor_tools import (
    perplexity_quick_search_low_context,
    query_asknews,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.auto_optimizers.bot_optimizer import BotOptimizer
from forecasting_tools.auto_optimizers.prompt_data_models import ResearchTool
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.util.custom_logger import CustomLogger

logger = logging.getLogger(__name__)


async def run_optimizer() -> None:
    # ----- Settings for the optimizer -----
    metaculus_question_path = "questions.json"
    questions = MetaculusQuestion.load_json_from_file_path(
        metaculus_question_path
    )
    research_tools = [
        ResearchTool(
            tool=perplexity_quick_search_low_context,
            max_calls=1,
        ),
        ResearchTool(
            tool=query_asknews,
            max_calls=1,
        ),
    ]
    ideation_llm = "openrouter/google/gemini-2.5-pro-preview"
    research_coordination_llm = GeneralLlm(
        model="openrouter/openai/gpt-4.1-mini", temperature=0.3
    )
    reasoning_llm = GeneralLlm(
        model="openrouter/openai/gpt-4.1-mini", temperature=0.3
    )
    questions_batch_size = 112
    num_iterations_per_run = 3
    remove_background_info = True

    # ------ Run the optimizer -----
    await BotOptimizer.optimize_a_combined_research_and_reasoning_prompt(
        questions=questions,
        research_tools=research_tools,
        research_agent_llm=research_coordination_llm,
        reasoning_llm=reasoning_llm,
        questions_batch_size=questions_batch_size,
        num_iterations_per_run=num_iterations_per_run,
        ideation_llm=ideation_llm,
        remove_background_info=remove_background_info,
    )


if __name__ == "__main__":
    CustomLogger.setup_logging()
    asyncio.run(run_optimizer())
