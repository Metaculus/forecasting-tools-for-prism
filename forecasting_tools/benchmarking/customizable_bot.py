import copy
import logging
from dataclasses import dataclass

from pydantic import BaseModel, field_validator

from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AgentSdkLlm,
    AgentTool,
    AiAgent,
)
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.benchmarking.prompt_data_models import PromptIdea
from forecasting_tools.benchmarking.question_plus_research import (
    QuestionPlusResearch,
    ResearchType,
)
from forecasting_tools.data_models.forecast_report import ReasonedPrediction
from forecasting_tools.data_models.multiple_choice_report import (
    PredictedOptionList,
)
from forecasting_tools.data_models.numeric_report import NumericDistribution
from forecasting_tools.data_models.questions import (
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.forecast_bots.forecast_bot import ForecastBot
from forecasting_tools.forecast_helpers.structure_output import (
    structure_output,
)

logger = logging.getLogger(__name__)


class BinaryPrediction(BaseModel):
    prediction_in_decimal: float

    @field_validator("prediction_in_decimal")
    @classmethod
    def validate_prediction_range(cls, value: float) -> float:
        if value == 0:
            return 0.001
        if value == 1:
            return 0.999

        if value < 0.001:
            raise ValueError("Prediction must be at least 0.001")
        if value > 0.999:
            raise ValueError("Prediction must be at most 0.999")
        return value


@dataclass
class ResearchTool:
    tool: AgentTool
    max_calls: int | None


class ToolUsageTracker:
    def __init__(self, research_tools: list[ResearchTool]) -> None:
        self.tool_usage: dict[str, int] = {}
        self.tool_limits: dict[str, int | None] = {}

        for research_tool in research_tools:
            tool_name = research_tool.tool.name
            self.tool_usage[tool_name] = 0
            self.tool_limits[tool_name] = research_tool.max_calls

    def increment_usage(self, tool_name: str) -> None:
        if tool_name not in self.tool_usage:
            raise ValueError(f"Tool {tool_name} not found in usage tracker")

        self.tool_usage[tool_name] += 1
        current_usage = self.tool_usage[tool_name]
        max_calls = self.tool_limits[tool_name]

        if max_calls is not None and current_usage > max_calls:
            raise RuntimeError(
                f"Tool {tool_name} has exceeded its maximum calls limit of {max_calls}. "
                f"Current usage: {current_usage}"
            )


class CustomizableBot(ForecastBot):
    """
    A customizable bot that can be used to forecast questions.

    The flow goes:
    1. The bot is given a question
    2. Research is performed
        a. If the question is in the research snapshots, the bot uses the research snapshot.
        b. If the question is not in the research snapshots, the bot uses research prompt to agentically run the research tools.
    3. The bot forecasts the question using the reasoning prompt
    4. The bot returns the forecast

    See ForecastBot for more details.
    """

    def __init__(
        self,
        reasoning_prompt: str,
        research_prompt: str,
        research_tools: list[ResearchTool],
        cached_research: list[QuestionPlusResearch] | None,
        research_type: ResearchType | None,
        originating_idea: PromptIdea | None,
        parameters_to_exclude_from_config_dict: list[str] | None = [
            "research_snapshots"
        ],
        *args,
        **kwargs,
    ) -> None:
        super().__init__(
            *args,
            parameters_to_exclude_from_config_dict=parameters_to_exclude_from_config_dict,
            **kwargs,
        )
        self.reasoning_prompt = reasoning_prompt
        self.research_prompt = research_prompt
        self.tools = research_tools
        self.research_type = research_type
        self.originating_idea = originating_idea  # As of May 26, 2025 This parameter is logged in the config for the bot, even if not used here.

        if cached_research is not None:
            unique_questions = list(
                set(
                    [
                        snapshot.question.question_text
                        for snapshot in cached_research
                    ]
                )
            )
            if len(unique_questions) != len(cached_research):
                raise ValueError(
                    "Research snapshots must have unique questions"
                )
            if research_type is None:
                raise ValueError(
                    "Research type must be provided if cached research is provided"
                )
        else:
            if research_type is not None:
                raise ValueError(
                    "Research type must be None if cached research is None"
                )

        self.cached_research = cached_research or []

    @classmethod
    def _llm_config_defaults(cls) -> dict[str, str | GeneralLlm | None]:
        return {
            "default": None,
            "summarizer": None,
            "researcher": None,
        }

    def _create_tracked_tools(
        self, usage_tracker: ToolUsageTracker
    ) -> list[AgentTool]:
        tracked_tools = []

        for research_tool in self.tools:
            original_tool = research_tool.tool
            tracked_tool = copy.deepcopy(original_tool)
            original_on_invoke = tracked_tool.on_invoke_tool

            async def wrapped_on_invoke_tool(
                ctx,
                input_str,
                tool_name=original_tool.name,
                original_func=original_on_invoke,
            ):
                usage_tracker.increment_usage(tool_name)
                return await original_func(ctx, input_str)

            tracked_tool.on_invoke_tool = wrapped_on_invoke_tool

            tracked_tools.append(tracked_tool)

        return tracked_tools

    async def run_research(self, question: MetaculusQuestion) -> str:
        matching_snapshots = [
            snapshot
            for snapshot in self.cached_research
            if snapshot.question == question
        ]
        if len(matching_snapshots) == 1:
            return matching_snapshots[0].get_research_for_type(
                self.research_type
            )

        if len(matching_snapshots) > 1:
            raise ValueError(
                f"Expected 1 research snapshot for question {question.page_url}, got {len(matching_snapshots)}"
            )

        if not self.tools:
            raise ValueError(
                f"No research snapshot found for question {question.page_url} and no research tools available"
            )

        return await self._run_research_with_tools(question)

    async def _run_research_with_tools(
        self, question: MetaculusQuestion
    ) -> str:
        research_llm = self.get_llm("researcher")
        if not isinstance(research_llm, str):
            raise ValueError("Research LLM must be a string model name")

        formatted_prompt = self.research_prompt.format(
            question_text=question.question_text,
            background_info=question.background_info,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
        )

        usage_tracker = ToolUsageTracker(self.tools)
        tracked_tools = self._create_tracked_tools(usage_tracker)

        agent = AiAgent(
            name="Research Agent",
            instructions=formatted_prompt,
            model=AgentSdkLlm(model=research_llm),
            tools=tracked_tools,  # type: ignore
            handoffs=[],
        )

        result = await AgentRunner.run(
            agent,
            f"Please research the following question: {question.question_text}",
        )
        final_output = result.final_output
        if not isinstance(final_output, str):
            raise ValueError(
                f"Expected final output to be a string, got {type(final_output)}"
            )

        return final_output

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        required_variables = [
            "{question_text}",
            "{resolution_criteria}",
            "{today}",
            "{research}",
        ]

        for required_variable in required_variables:
            if required_variable not in self.reasoning_prompt:
                raise ValueError(
                    f"Prompt {self.reasoning_prompt} does not contain {required_variable}"
                )
        prompt = self.reasoning_prompt.format(
            question_text=question.question_text,
            background_info=question.background_info,
            resolution_criteria=question.resolution_criteria,
            fine_print=question.fine_print,
            today=question.date_accessed.strftime("%Y-%m-%d"),
            research=research,
        )
        reasoning = await self.get_llm("default", "llm").invoke(prompt)
        logger.info(f"Reasoning for URL {question.page_url}: {reasoning}")

        binary_prediction: BinaryPrediction = await structure_output(
            reasoning, BinaryPrediction
        )
        prediction = binary_prediction.prediction_in_decimal

        logger.info(
            f"Forecasted URL {question.page_url} with prediction: {prediction}"
        )
        if prediction >= 1:
            prediction = 0.999
        if prediction <= 0:
            prediction = 0.001
        return ReasonedPrediction(
            prediction_value=prediction, reasoning=reasoning
        )

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        raise NotImplementedError()

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        raise NotImplementedError()
