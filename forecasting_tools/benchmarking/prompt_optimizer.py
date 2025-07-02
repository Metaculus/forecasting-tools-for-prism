from __future__ import annotations

import asyncio
import logging
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from forecasting_tools.agents_and_tools.misc_tools import perplexity_pro_search
from forecasting_tools.ai_models.agent_wrappers import (
    AgentRunner,
    AgentSdkLlm,
    AiAgent,
)
from forecasting_tools.ai_models.ai_utils.ai_misc import clean_indents
from forecasting_tools.benchmarking.prompt_data_models import PromptIdea
from forecasting_tools.forecast_helpers.structure_output import (
    structure_output,
)

logger = logging.getLogger(__name__)


class ImplementedPrompt(BaseModel):
    text: str
    idea: PromptIdea
    iteration_number: int
    originating_ideas: list[PromptIdea]


class PromptScore(BaseModel):
    value: float
    metadata: dict[str, Any]


class ScoredPrompt(BaseModel):
    prompt: ImplementedPrompt
    score: PromptScore


class OptimizationRun(BaseModel):
    implemented_prompts: list[ScoredPrompt]

    @property
    def best_prompt(self) -> ScoredPrompt:
        return max(self.implemented_prompts, key=lambda x: x.score.value)


class PromptOptimizer:

    def __init__(
        self,
        initial_prompts: list[str],
        iterations: int,
        ideation_llm_name: str,
        prompts_to_scores_func: Callable[
            [list[str]], Awaitable[list[PromptScore]]
        ],  # Function that takes a list of prompts and returns a list of scores
        prompt_purpose_explanation: str,  # e.g. "You are making a prompt for an AI to forecast binary questions about future events. You want to optimize ..."
        prompt_requirements_explanation: str,  # e.g. "The prompt should ask an AI to forecast a binary question and require a final binary float as the output."
        format_scores_func: (
            Callable[[ScoredPrompt], Awaitable[str]] | None
        ) = None,  # Function that takes a list of scored prompts and returns a string of the scores and metadata to review in each iteration
        initial_prompt_population_size: int = 25,
        survivors_per_iteration: int = 5,
        mutated_prompts_per_survivor: int = 4,
        breeded_prompts_per_iteration: int = 5,
    ) -> None:
        self.initial_prompts = initial_prompts
        self.iterations = iterations
        self.ideation_llm_name = ideation_llm_name
        self.prompt_to_score_func = prompts_to_scores_func
        self.format_scores_func = format_scores_func
        self.prompt_purpose_explanation = prompt_purpose_explanation
        self.prompt_requirements_explanation = prompt_requirements_explanation
        self.initial_prompt_population_size = initial_prompt_population_size
        self.survivors_per_iteration = survivors_per_iteration
        self.mutated_prompts_per_survivor = mutated_prompts_per_survivor
        self.breeded_prompts_per_iteration = breeded_prompts_per_iteration

        if (
            self.mutated_prompts_per_survivor == 0
            and self.breeded_prompts_per_iteration == 0
        ):
            raise ValueError(
                "At least one of mutated_prompts_per_surviving_prompt or breeded_prompts_per_iteration must be greater than 0"
            )

        if (
            len(self.initial_prompts) > self.initial_prompt_population_size
            or self.initial_prompt_population_size < 1
        ):
            raise ValueError(
                f"Initial prompt population size must be greater than 0 and less than or equal to the number of initial prompts. Got {self.initial_prompt_population_size} and {len(self.initial_prompts)}."
            )

    async def create_optimized_prompt(self) -> OptimizationRun:
        initial_prompts: list[ImplementedPrompt] = [
            ImplementedPrompt(
                text=prompt,
                idea=PromptIdea(
                    short_name=f"Initial Seed {i}",
                    idea=f"The user-provided initial prompt {i}.",
                ),
                iteration_number=0,
                originating_ideas=[],
            )
            for i, prompt in enumerate(self.initial_prompts)
        ]
        prompts_still_needed = self.initial_prompt_population_size - len(
            initial_prompts
        )
        if prompts_still_needed > 0:
            additional_initial_prompts = await self._mutate_prompt(
                initial_prompts[0], prompts_still_needed
            )
            initial_prompts.extend(additional_initial_prompts)

        all_evaluated_prompts: list[ScoredPrompt] = []
        survivors: list[ScoredPrompt] = []
        newborn_prompts: list[ImplementedPrompt] = initial_prompts.copy()

        for iteration_num in range(self.iterations):
            logger.info(
                f"Starting iteration {iteration_num + 1}/{self.iterations} - Current population size: {len(newborn_prompts)}"
            )

            adult_prompts = await self._evaluate_new_members(newborn_prompts)
            all_evaluated_prompts.extend(adult_prompts)

            updated_population = survivors + adult_prompts
            survivors = await self._kill_the_weak(updated_population)

            newborn_prompts = await self._birth_new_prompts(survivors)

            self._log_duplicate_prompts(all_evaluated_prompts)

        return OptimizationRun(implemented_prompts=all_evaluated_prompts)

    async def _kill_the_weak(
        self, current_population: list[ScoredPrompt]
    ) -> list[ScoredPrompt]:
        current_population.sort(
            key=lambda x: x.score.value,
            reverse=True,
        )
        logger.debug(f"Current survivors: {current_population}")
        best_survivor = current_population[0]
        logger.info(
            f"Best survivor: {best_survivor.prompt.idea.short_name} with score {best_survivor.score.value:.4f}. Prompt:\n {best_survivor.prompt.text}"
        )
        return current_population[: self.survivors_per_iteration]

    async def _birth_new_prompts(
        self, surviving_population: list[ScoredPrompt]
    ) -> list[ImplementedPrompt]:
        mutated_prompts: list[ImplementedPrompt] = []
        mutation_tasks = [
            self._mutate_prompt(ep.prompt, self.mutated_prompts_per_survivor)
            for ep in surviving_population
        ]
        initial_mutation_results = await asyncio.gather(
            *mutation_tasks, return_exceptions=True
        )
        mutation_results: list[list[ImplementedPrompt]] = [
            result
            for result in initial_mutation_results
            if not isinstance(result, BaseException)
        ]
        for mutation_list in mutation_results:
            mutated_prompts.extend(mutation_list)
        logger.info(f"Generated {len(mutated_prompts)} mutated prompts.")

        bred_prompts: list[ImplementedPrompt] = []
        try:
            bred_prompts = await self._breed_prompts(surviving_population)
        except Exception as e:
            logger.error(f"Failed to breed prompts: {e}")
            bred_prompts = []
        logger.info(f"Generated {len(bred_prompts)} bred prompts.")

        new_prompts = mutated_prompts + bred_prompts
        return new_prompts

    async def _evaluate_new_members(
        self, prompts: list[ImplementedPrompt]
    ) -> list[ScoredPrompt]:
        scores = await self.prompt_to_score_func([p.text for p in prompts])
        return [
            ScoredPrompt(prompt=p, score=s) for p, s in zip(prompts, scores)
        ]

    async def _mutate_prompt(
        self,
        input_prompt: ScoredPrompt | ImplementedPrompt,
        num_mutations_to_generate: int,
    ) -> list[ImplementedPrompt]:
        if isinstance(input_prompt, ImplementedPrompt):
            scores_str = ""
            prompt = input_prompt
        else:
            if self.format_scores_func is not None:
                scores_str = await self.format_scores_func(input_prompt)
            else:
                scores_str = ""
            prompt = input_prompt.prompt

        agent_mutate_ideas = AiAgent(
            name="Prompt Mutator Ideator",
            model=AgentSdkLlm(self.ideation_llm_name),
            instructions=clean_indents(
                f"""
                You are an expert prompt engineer. Your task is to generate {num_mutations_to_generate} new PROMPT IDEAS by mutating an existing prompt.
                Your ideas are being used to optimize this prompt using a Genetic Algorithm inspired approach.
                We are highlighting exploration over exploitation, but do want to strike a balance.

                # Purpose of Prompt
                {self.prompt_purpose_explanation}

                # Prompt Requirements
                The final prompt (i.e. not the ideas you will generate) will have the below requirements. Another agent will implement these requirements.

                {self.prompt_requirements_explanation}

                # Instructions
                1. Please analyze the scores from the previous prompt and identify what went wrong.
                2. Run 3-10 searches on the web to find inspiration for novel prompt structures, techniques, and ideas that will solve the goal.
                3. Generate {num_mutations_to_generate} new, distinct PROMPT IDEAS based on the original.

                Please generate exactly {num_mutations_to_generate} new, distinct PROMPT IDEAS based on the original.
                Each mutation idea must be a concept for a new, complete prompt. The implemented prompt will:

                For each idea please sequentially follow these policies to determine how much you try to mutate the original prompt:
                1st idea: "slight modification, like changing wording, adding/removing a sentences or a small paragraph, reording steps, adding emphasis, etc",
                2nd idea: "significant variation, which should take a generally different approach and be a general rewrite while staying in general theme of the original",
                3rd idea: "highly diverse mutation/experiment that explores a substantially different structure or set of principles, focus on a completely different idea than in the original. Search until you find something novel.",
                nth idea: ... continue alternating between significant variation and highly diverse (not slight)...

                # Original Prompt Idea Details
                Name: {prompt.idea.short_name}
                Core Idea: {prompt.idea.idea}

                Original Prompt Template (for context only, do not reproduce it in your output):
                ```
                {prompt.text}
                ```

                # Scores from Original Prompt:
                {scores_str}

                # Format
                **Mutated Idea Title 1**
                New idea for prompt mutation 1, specifying in detail how to implement the prompt reflecting the target variation.

                **Mutated Idea Title 2**
                New idea for prompt mutation 2, specifying in detail how to implement the prompt reflecting the target variation.
                ...
                (up to {num_mutations_to_generate} ideas)
                """
            ),
            tools=[perplexity_pro_search],
        )

        mutation_agent_task = (
            f"Generate {num_mutations_to_generate} mutated prompt ideas for the prompt named '{prompt.idea.short_name}'. "
            f"Ensure each mutation aligns with the requested degree of variation."
        )
        output = await AgentRunner.run(agent_mutate_ideas, mutation_agent_task)
        mutated_ideas = await structure_output(
            output.final_output, list[PromptIdea]
        )
        logger.info(
            f"Successfully structured {len(mutated_ideas)} mutation ideas for prompt '{prompt.idea.short_name}'. Requested {num_mutations_to_generate}."
        )

        if len(mutated_ideas) != num_mutations_to_generate:
            logger.warning(
                f"Requested {num_mutations_to_generate} mutation ideas, but got {len(mutated_ideas)}. Returning {mutated_ideas[:num_mutations_to_generate]}"
            )
            mutated_ideas = mutated_ideas[:num_mutations_to_generate]

        implemented_prompts = await self._implement_prompt_ideas(mutated_ideas)
        logger.info(
            f"Successfully created {len(implemented_prompts)} implemented prompts from {len(mutated_ideas)} mutation ideas."
        )
        return implemented_prompts

    async def _breed_prompts(
        self, parent_scored_prompts: list[ScoredPrompt]
    ) -> list[ImplementedPrompt]:
        num_to_breed = self.breeded_prompts_per_iteration
        if num_to_breed == 0:
            return []
        if len(parent_scored_prompts) < 2:
            raise ValueError(
                f"Need at least 2 parent prompts, got {len(parent_scored_prompts)}."
            )

        raise NotImplementedError("Not implemented")

    async def _implement_prompt_ideas(
        self, prompt_ideas: list[PromptIdea]
    ) -> list[ImplementedPrompt]:
        raise NotImplementedError("Not implemented")

    def _log_duplicate_prompts(self, prompts: list[ScoredPrompt]) -> None:
        for prompt in prompts:
            if prompt.prompt.text in [ep.prompt.text for ep in prompts]:
                logger.warning(
                    f"Duplicate prompt template found: {prompt.prompt.text}"
                )
