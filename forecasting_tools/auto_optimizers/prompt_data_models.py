from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from forecasting_tools.ai_models.agent_wrappers import AgentTool
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.cp_benchmarking.benchmark_for_bot import BenchmarkForBot


class PromptIdea(BaseModel):
    short_name: str
    full_text: str


@dataclass
class ResearchTool:
    tool: AgentTool
    max_calls: int | None


@dataclass
class BotConfig:
    reasoning_prompt_template: str
    research_prompt_template: str
    research_tools: list[ResearchTool]
    reasoning_llm: GeneralLlm | str
    research_llm: GeneralLlm | str
    originating_idea: PromptIdea
    research_reports_per_question: int = 1
    predictions_per_research_report: int = 1


@dataclass
class EvaluatedBot:
    bot_config: BotConfig
    benchmark: BenchmarkForBot

    @property
    def score(self) -> float:
        return self.benchmark.average_expected_baseline_score


@dataclass
class BotEvaluation:
    evaluated_prompts: list[EvaluatedBot]

    @property
    def best_bot(self) -> EvaluatedBot:
        sorted_evaluated_prompts = sorted(
            self.evaluated_prompts, key=lambda x: x.score, reverse=True
        )
        return sorted_evaluated_prompts[0]
