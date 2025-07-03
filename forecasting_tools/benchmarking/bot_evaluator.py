import logging

import typeguard

from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.benchmarking.benchmark_for_bot import BenchmarkForBot
from forecasting_tools.benchmarking.benchmarker import Benchmarker
from forecasting_tools.benchmarking.control_group_prompt import ControlPrompt
from forecasting_tools.benchmarking.customizable_bot import CustomizableBot
from forecasting_tools.benchmarking.prompt_data_models import (
    BotConfig,
    BotEvaluation,
    EvaluatedPrompt,
    PromptIdea,
)
from forecasting_tools.benchmarking.question_plus_research import (
    QuestionPlusResearch,
    ResearchType,
)
from forecasting_tools.data_models.questions import MetaculusQuestion
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot

logger = logging.getLogger(__name__)


class BotEvaluator:
    def __init__(
        self,
        input_questions: list[QuestionPlusResearch] | list[MetaculusQuestion],
        research_type: ResearchType | None,
        concurrent_evaluation_batch_size: int,
        file_or_folder_to_save_benchmarks: str | None,
    ) -> None:
        if research_type is None:
            input_questions = typeguard.check_type(
                input_questions, list[MetaculusQuestion]
            )
            self.evaluation_questions = input_questions
            self.research_snapshots = []
        else:
            input_questions = typeguard.check_type(
                input_questions, list[QuestionPlusResearch]
            )
            self.evaluation_questions = [
                snapshot.question for snapshot in input_questions
            ]
            self.research_snapshots = input_questions

        self.research_type = research_type
        self.concurrent_evaluation_batch_size = (
            concurrent_evaluation_batch_size
        )
        self.file_or_folder_to_save_benchmarks = (
            file_or_folder_to_save_benchmarks
        )

    async def evaluate_bot_configs(
        self, configurations: list[BotConfig]
    ) -> BotEvaluation:
        bots = self._configs_to_bots(configurations)
        bots = typeguard.check_type(bots, list[ForecastBot])
        benchmarker = Benchmarker(
            forecast_bots=bots,
            questions_to_use=self.evaluation_questions,
            concurrent_question_batch_size=self.concurrent_evaluation_batch_size,
            file_path_to_save_reports=self.file_or_folder_to_save_benchmarks,
        )
        benchmarks = await benchmarker.run_benchmark()
        evaluated_prompts: list[EvaluatedPrompt] = []
        for config, benchmark in zip(configurations, benchmarks):
            benchmark.forecast_bot_class_name = (
                config.originating_idea.short_name.replace(" ", "_")
            )
            if len(benchmark.forecast_reports) > 0:
                evaluated_prompts.append(
                    EvaluatedPrompt(bot_config=config, benchmark=benchmark)
                )
            else:
                logger.error(
                    f"Not including {config.originating_idea.short_name} in evaluation report because it has no forecast reports"
                )
        return BotEvaluation(evaluated_prompts=evaluated_prompts)

    def _configs_to_bots(
        self, configs: list[BotConfig]
    ) -> list[CustomizableBot]:
        bots = []
        for config in configs:
            if config.research_reports_per_question != 1:
                raise NotImplementedError(
                    "Currently only supports one research report per question"
                )
            custom_class_name = config.originating_idea.short_name.replace(
                " ", "_"
            )
            CustomBotClass = type(custom_class_name, (CustomizableBot,), {})
            bot = CustomBotClass(
                originating_idea=config.originating_idea,
                reasoning_prompt=config.reasoning_prompt_template,
                research_prompt=config.research_prompt_template,
                research_tools=config.research_tools,
                cached_research=self.research_snapshots,
                research_type=self.research_type,
                research_reports_per_question=config.research_reports_per_question,
                predictions_per_research_report=config.predictions_per_research_report,
                llms={
                    "default": config.reasoning_llm,
                },
                publish_reports_to_metaculus=False,
                enable_summarize_research=False,
            )
            bots.append(bot)
        return bots

    async def evaluate_best_benchmarked_prompts(
        self,
        benchmark_files: list[str],
        forecast_llm: GeneralLlm,
        top_n_prompts: int = 1,
        include_control_group_prompt: bool = True,
        include_worst_prompt: bool = False,
        research_reports_per_question: int = 1,
        num_predictions_per_research_report: int = 1,
    ) -> BotEvaluation:
        best_benchmarks = self._get_best_benchmark_prompts(
            benchmark_files, top_n_prompts, include_worst_prompt
        )

        logger.info(
            f"Evaluating {len(best_benchmarks)} prompts with {forecast_llm.model}. Prompts are the best scoring from files {benchmark_files}"
        )
        configs = []
        if include_control_group_prompt:
            control_group_config = BotConfig(
                reasoning_prompt_template=ControlPrompt.get_reasoning_prompt(),
                reasoning_llm=forecast_llm,
                originating_idea=PromptIdea(
                    short_name=f"Control Group v{ControlPrompt.version()}",
                    full_text="The control group is a group of questions that are not optimized for the prompt. It is used to evaluate the performance of the optimized prompt.",
                ),
                predictions_per_research_report=num_predictions_per_research_report,
                research_reports_per_question=research_reports_per_question,
            )
            configs.append(control_group_config)
        for benchmark in best_benchmarks:
            prompt = benchmark.forecast_bot_config["prompt"]
            logger.info(
                f"{benchmark.forecast_bot_class_name} - {benchmark.average_expected_baseline_score}:\n{prompt}"
            )
            best_prompt_config = BotConfig(
                reasoning_prompt_template=prompt,
                reasoning_llm=forecast_llm,
                originating_idea=PromptIdea(
                    short_name=f"{benchmark.forecast_bot_class_name}",
                    full_text=f"Evaluate the prompt from {benchmark.forecast_bot_class_name} (originally found from a different dataset/origin) with model {forecast_llm.model} and {len(self.evaluation_questions)} questions",
                ),
            )
            configs.append(best_prompt_config)
        evaluation_result = await self.evaluate_bot_configs(configs)
        return evaluation_result

    @classmethod
    def _get_best_benchmark_prompts(
        cls,
        file_paths: list[str],
        top_n_prompts: int = 1,
        include_worst_prompt: bool = False,
    ) -> list[BenchmarkForBot]:
        logger.info(
            f"Attempting to get the best {top_n_prompts} prompts from {file_paths}"
        )
        all_benchmarks = []
        for file_path in file_paths:
            benchmarks = BenchmarkForBot.load_json_from_file_path(file_path)
            logger.info(
                f"Loaded {len(benchmarks)} benchmarks from {file_path}"
            )
            for benchmark in benchmarks:
                if len(benchmark.forecast_reports) > 0:
                    all_benchmarks.append(benchmark)
        sorted_benchmarks = sorted(
            all_benchmarks,
            key=lambda x: x.average_expected_baseline_score,
            reverse=True,
        )
        best_benchmarks = sorted_benchmarks[:top_n_prompts]
        if include_worst_prompt:
            worst_benchmark = sorted_benchmarks[-1]
            return best_benchmarks + [worst_benchmark]
        return best_benchmarks
