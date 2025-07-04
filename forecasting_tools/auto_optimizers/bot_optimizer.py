import logging

from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.auto_optimizers.bot_evaluator import BotEvaluator
from forecasting_tools.auto_optimizers.control_group_prompt import (
    ControlPrompt,
)
from forecasting_tools.auto_optimizers.customizable_bot import CustomizableBot
from forecasting_tools.auto_optimizers.prompt_data_models import (
    BotConfig,
    ResearchTool,
)
from forecasting_tools.auto_optimizers.prompt_optimizer import (
    ImplementedPrompt,
    OptimizationRun,
    PromptOptimizer,
    PromptScore,
    ScoredPrompt,
)
from forecasting_tools.benchmarking.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.data_models.questions import MetaculusQuestion

logger = logging.getLogger(__name__)


class BotOptimizer:

    @classmethod
    async def optimize_a_combined_research_and_reasoning_prompt(
        cls,
        questions: list[MetaculusQuestion],
        research_tools: list[ResearchTool],
        research_agent_llm: GeneralLlm,
        reasoning_llm: GeneralLlm,
        questions_batch_size: int,
        num_iterations_per_run: int,
        ideation_llm: str,
        remove_background_info: bool,
    ) -> OptimizationRun:
        logger.info(f"Loaded {len(questions)} questions")
        questions = [question.model_copy(deep=True) for question in questions]
        if remove_background_info:
            for question in questions:
                question.background_info = None
        prompt_purpose_explanation = clean_indents(
            """
            You are making a prompt for an AI to forecast binary questions about future events on prediction markets/aggregators.
            This needs to optimize forecasting accuracy as measured by log/brier scores.
            """
        )
        research_tool_limits = "\n###".join(
            [
                f"{tool.tool.name}\nMax calls allowed: {tool.max_calls}\nDescription: {tool.tool.description}"
                for tool in research_tools
            ]
        )
        prompt_requirements_explanation = clean_indents(
            f"""
            The prompt should be split into 2 main parts:
            - The research part of the prompt
            - The reasoning part of the prompt

            ## Prompt Format:
            The research part
            - should be used to generate a research report.
            - This research report will be passed to the reasoning prompt.
            - Should explicitly state limits on how much tools should be used (e.g. max tool calls overall or per step)
            The reasoning part
            - should be used to generate a forecast for a binary question.
            - The forecast must be a probability between 0 and 1.
            Deliminator:
            - The research and reasoning part of the prompt should be separated by the following string: {CustomizableBot.RESEARCH_REASONING_SPLIT_STRING}
            - There should be no other deliminators between the prompts, only this string.

            ## Research tools and their limits:
            {research_tool_limits}
            """
        )
        mutation_considerations = clean_indents(
            """
            As you develop the prompt consider:
            - Research formats and length (consider all varieties)
            - Research sources (consider all varieties)
            - Research strategies (i.e. which steps in which order, with what criteria for what steps)
            - Which tools to use and how much
            - Whether you want to call tools in parallel at each step or not
            - Reasoning formats and length (consider all varieties)
            - Reasoning strategies (i.e. which steps in which order, with what criteria for what steps)
            """
        )
        template_variables_explanation = clean_indents(
            f"""
            ## Research part of the prompt
            REQUIRED: The research part of the prompt should include all of the following variables:
            {CustomizableBot.REQUIRED_RESEARCH_PROMPT_VARIABLES}

            Optionally the research part of the prompt can include the following variables:
            {CustomizableBot.OPTIONAL_RESEARCH_PROMPT_VARIABLES}

            ## Reasoning part of the prompt
            REQUIRED: The reasoning part of the prompt should include the all of the following variables:
            {CustomizableBot.REQUIRED_REASONING_PROMPT_VARIABLES}

            Remember to only include the {{research}} variable once in the reasoning part of the prompt (this will have a lot of text in it, so we don't want to repeat it).
            """
        )

        evaluator = BotEvaluator(
            input_questions=questions,
            research_type=None,
            concurrent_evaluation_batch_size=questions_batch_size,
            file_or_folder_to_save_benchmarks="logs/forecasts/benchmarks/",
        )

        async def evaluate_combined_research_and_reasoning_prompts(
            combined_prompts: list[ImplementedPrompt],
        ) -> list[PromptScore]:
            configs = []
            for combined_prompt in combined_prompts:
                research_prompt, reasoning_prompt = (
                    CustomizableBot.split_combined_research_reasoning_prompt(
                        combined_prompt.text
                    )
                )
                configs.append(
                    BotConfig(
                        reasoning_prompt_template=reasoning_prompt,
                        research_prompt_template=research_prompt,
                        research_tools=research_tools,
                        research_llm=research_agent_llm,
                        reasoning_llm=reasoning_llm,
                        originating_idea=combined_prompt.idea,
                    )
                )
            evaluation_result = await evaluator.evaluate_bot_configs(configs)
            evaluated_prompts = evaluation_result.evaluated_prompts
            assert len(evaluated_prompts) == len(
                combined_prompts
            ), f"Number of evaluated prompts ({len(evaluated_prompts)}) does not match number of combined prompts ({len(combined_prompts)})"

            prompt_scores = []
            for evaluated_prompt in evaluated_prompts:
                prompt_scores.append(
                    PromptScore(
                        value=evaluated_prompt.score,
                        metadata={
                            "benchmark": evaluated_prompt.benchmark,
                        },
                    )
                )
            return prompt_scores

        optimizer = PromptOptimizer(
            initial_prompt=ControlPrompt.get_combined_prompt(),
            iterations=num_iterations_per_run,
            ideation_llm_name=ideation_llm,
            prompts_to_scores_func=evaluate_combined_research_and_reasoning_prompts,
            prompt_purpose_explanation=prompt_purpose_explanation,
            prompt_requirements_explanation=prompt_requirements_explanation,
            template_variables_explanation=template_variables_explanation,
            mutation_considerations=mutation_considerations,
            format_scores_func=cls._format_worst_scores_and_context,
            initial_prompt_population_size=20,
            survivors_per_iteration=5,
            mutated_prompts_per_survivor=3,
            breeded_prompts_per_iteration=5,
        )

        logger.info("Starting optimization run")
        optimization_run = await optimizer.create_optimized_prompt()
        logger.info("Optimization run complete")

        cls._log_best_prompts(optimization_run)
        return optimization_run

    @staticmethod
    def _log_best_prompts(optimization_run: OptimizationRun) -> None:
        best_prompts = optimization_run.scored_prompts
        best_prompts.sort(key=lambda x: x.score.value, reverse=True)
        best_prompts = best_prompts[:5]
        message = "Best prompts:\n"
        for sp in best_prompts:
            message += "\n\n------------------------------"
            message += f"\nScore: {sp.score.value}"
            message += f"\nIteration: {sp.prompt.iteration_number}"
            message += f"\nIdea Name: {sp.prompt.idea.short_name}"
            message += f"\nIdea Description: {sp.prompt.idea.full_text}"
            message += f"\nPrompt: {sp.prompt.text}"
            message += f"\nOriginating Ideas: {sp.prompt.originating_ideas}"
            message += f"\nCost: {sp.score.metadata['benchmark'].total_cost}"
        logger.info(message)

    @staticmethod
    async def _format_worst_scores_and_context(
        scored_prompt: ScoredPrompt,
    ) -> str:
        benchmark: BenchmarkForBot = scored_prompt.score.metadata["benchmark"]
        num_worst_reports = (
            3
            if len(benchmark.forecast_reports) > 3
            else len(benchmark.forecast_reports)
        )
        worst_reports = benchmark.get_bottom_n_forecast_reports(
            num_worst_reports
        )

        report_str = f"Below are the worst {num_worst_reports} scores from the previous prompt. These are baseline scores (100pts is perfect forecast, -897pts is worst possible forecast, and 0pt is forecasting 50%):\n"
        report_str += f"<><><><><><><><><><><><><><> TOP {num_worst_reports} WORST REPORTS <><><><><><><><><><><><><><>\n"
        for report in worst_reports:
            report_str += clean_indents(
                f"""
                ##  Question: {report.question.question_text} **(Score: {report.expected_baseline_score:.4f})**
                **Summary**
                ```{report.summary}```
                **Research**
                ```{report.research}```
                **First rationale**
                ```{report.first_rationale}```
                """
            )
        report_str += "<><><><><><><><><><><><><><> END OF REPORTS <><><><><><><><><><><><><><>\n"
        return report_str
