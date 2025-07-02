from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel

from forecasting_tools.ai_models.agent_wrappers import AgentTool
from forecasting_tools.ai_models.general_llm import GeneralLlm
from forecasting_tools.benchmarking.benchmark_for_bot import BenchmarkForBot


class PromptIdea(BaseModel):
    short_name: str
    idea: str


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
    reasoning_idea: PromptIdea
    original_research_idea: PromptIdea
    research_reports_per_question: int = 1
    predictions_per_research_report: int = 1


@dataclass
class EvaluatedPrompt:
    bot_config: BotConfig
    benchmark: BenchmarkForBot

    @property
    def score(self) -> float:
        return self.benchmark.average_expected_baseline_score

    @property
    def prompt_text(self) -> str:
        return self.bot_config.reasoning_prompt_template


@dataclass
class OptimizationResult:
    evaluated_prompts: list[EvaluatedPrompt]

    @property
    def best_prompt(self) -> EvaluatedPrompt:
        sorted_evaluated_prompts = sorted(
            self.evaluated_prompts, key=lambda x: x.score, reverse=True
        )
        return sorted_evaluated_prompts[0]

    @property
    def best_prompt_text(self) -> str:
        return self.best_prompt.prompt_text
