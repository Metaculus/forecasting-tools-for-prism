from code_tests.unit_tests.test_forecasting.forecasting_test_manager import (
    ForecastingTestManager,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.benchmarking.bot_evaluator import BotEvaluator
from forecasting_tools.benchmarking.bot_optimizer import BotOptimizer
from forecasting_tools.benchmarking.question_plus_research import (
    QuestionPlusResearch,
    ResearchItem,
    ResearchType,
)


async def test_prompt_optimizer() -> None:
    question = ForecastingTestManager.get_fake_binary_question()
    research_snapshot = QuestionPlusResearch(
        question=question,
        research_items=[
            ResearchItem(
                research="Something will happen!",
                type=ResearchType.ASK_NEWS_SUMMARIES,
            )
        ],
    )
    evaluator = BotEvaluator(
        input_questions=[research_snapshot],
        research_type=ResearchType.ASK_NEWS_SUMMARIES,
        concurrent_evaluation_batch_size=10,
        file_or_folder_to_save_benchmarks=None,
    )
    prompt_optimizer = BotOptimizer(
        evaluator=evaluator,
        iterations=2,
        forecast_llm=GeneralLlm(model="gpt-4.1-nano"),
        ideation_llm_name="gpt-4.1-nano",
        initial_prompt_population_size=3,
        survivors_per_iteration=3,
        mutated_prompts_per_survivor=2,
        breeded_prompts_per_iteration=2,
    )
    with MonetaryCostManager(1):
        optimized_result = (
            await prompt_optimizer.optimize_both_research_and_reasoning()
        )
    assert optimized_result is not None
    assert optimized_result.best_prompt_text is not None
