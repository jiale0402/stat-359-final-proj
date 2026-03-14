"""Reward designs used in the GRPO reward-ablation project."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
import re
from typing import Dict

from arithmetic_llm.arithmetic_verifier import (
    ArithmeticVerifier,
)


_THINK_BLOCK_RE = re.compile(r"<think>(.*?)</think>", flags=re.IGNORECASE | re.DOTALL)
_STEP_RE = re.compile(r"Step\s+\d+\s*:", flags=re.IGNORECASE)
_EQUATION_RE = re.compile(r"[+-]?\d+\s*[+-]\s*[+-]?\d+\s*=\s*[+-]?\d+")
_FINAL_RESULT_RE = re.compile(
    r"Final Result\s*:\s*(ERROR|[+-]?\s*\d+)",
    flags=re.IGNORECASE,
)


class RewardDesign(str, Enum):
    """Reward settings described in the report."""

    H1_BINARY_ONLY = "h1_binary_only"
    H2_BINARY_PROCESS_FORMAT = "h2_binary_process_format"
    H3_BINARY_DISTANCE = "h3_binary_distance"


@dataclass(frozen=True)
class RewardWeights:
    """Linear reward weights for each experiment condition."""

    binary: float
    process: float = 0.0
    format: float = 0.0
    distance: float = 0.0
    formula: str = ""
    label: str = ""


REWARD_DESIGNS: Dict[RewardDesign, RewardWeights] = {
    RewardDesign.H1_BINARY_ONLY: RewardWeights(
        binary=1.0,
        formula="R = 1.0 * binary",
        label="H1: Binary-only reward",
    ),
    RewardDesign.H2_BINARY_PROCESS_FORMAT: RewardWeights(
        binary=0.8,
        process=0.15,
        format=0.05,
        formula="R = 0.8 * binary + 0.15 * process + 0.05 * format",
        label="H2: Binary + process + format reward",
    ),
    RewardDesign.H3_BINARY_DISTANCE: RewardWeights(
        binary=0.9,
        distance=0.1,
        formula="R = 0.9 * binary + 0.1 * exp(-|err| / 10)",
        label="H3: Binary + distance reward",
    ),
}


def resolve_reward_design(value: str | RewardDesign) -> RewardDesign:
    """Resolve a CLI or enum value into a RewardDesign."""
    if isinstance(value, RewardDesign):
        return value
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "h1": RewardDesign.H1_BINARY_ONLY,
        "binary": RewardDesign.H1_BINARY_ONLY,
        "binary_only": RewardDesign.H1_BINARY_ONLY,
        "h2": RewardDesign.H2_BINARY_PROCESS_FORMAT,
        "binary_process_format": RewardDesign.H2_BINARY_PROCESS_FORMAT,
        "process_format": RewardDesign.H2_BINARY_PROCESS_FORMAT,
        "h3": RewardDesign.H3_BINARY_DISTANCE,
        "binary_distance": RewardDesign.H3_BINARY_DISTANCE,
        "distance": RewardDesign.H3_BINARY_DISTANCE,
    }
    return aliases.get(normalized, RewardDesign(normalized))


class RewardAblationVerifier(ArithmeticVerifier):
    """Verifier with the three report reward settings."""

    def __init__(self, design: str | RewardDesign):
        super().__init__()
        self.design = resolve_reward_design(design)
        self.weights = REWARD_DESIGNS[self.design]

    def compute_binary_reward(self, generated_text: str, ground_truth: int) -> float:
        return float(super().compute_reward(generated_text, ground_truth))

    def compute_process_reward(self, generated_text: str) -> float:
        think_match = _THINK_BLOCK_RE.search(generated_text)
        reasoning_text = think_match.group(1) if think_match else generated_text
        reasoning_text = reasoning_text.strip()

        score = 0.0
        if reasoning_text:
            score += 0.2

        num_steps = min(len(_STEP_RE.findall(reasoning_text)), 3)
        score += 0.5 * (num_steps / 3.0)

        num_equations = min(len(_EQUATION_RE.findall(reasoning_text)), 3)
        score += 0.3 * (num_equations / 3.0)
        return min(score, 1.0)

    def compute_format_reward(self, generated_text: str) -> float:
        lower_text = generated_text.lower()
        think_start = lower_text.find("<think>")
        think_end = lower_text.find("</think>")
        final_match = _FINAL_RESULT_RE.search(generated_text)

        score = 0.0
        if think_start != -1:
            score += 0.25
        if think_end != -1 and think_end > think_start:
            score += 0.25
        if final_match is not None:
            score += 0.25
        if final_match is not None and (think_end == -1 or final_match.start() > think_end):
            score += 0.25
        return min(score, 1.0)

    def compute_distance_reward(self, generated_text: str, ground_truth: int) -> float:
        result = self.extract_final_result(generated_text)
        if result is None:
            return 0.0
        return math.exp(-abs(result - ground_truth) / 10.0)

    def score_response(self, generated_text: str, ground_truth: int) -> dict[str, float]:
        binary_reward = self.compute_binary_reward(generated_text, ground_truth)
        process_reward = self.compute_process_reward(generated_text)
        format_reward = self.compute_format_reward(generated_text)
        distance_reward = self.compute_distance_reward(generated_text, ground_truth)

        reward = self.weights.binary * binary_reward
        if self.design == RewardDesign.H2_BINARY_PROCESS_FORMAT:
            reward += self.weights.process * process_reward
            reward += self.weights.format * format_reward
        elif self.design == RewardDesign.H3_BINARY_DISTANCE:
            reward += self.weights.distance * distance_reward

        score = {
            "reward": float(reward),
            "binary_reward": float(binary_reward),
        }
        if self.design == RewardDesign.H2_BINARY_PROCESS_FORMAT:
            score["process_reward"] = float(process_reward)
            score["format_reward"] = float(format_reward)
        if self.design == RewardDesign.H3_BINARY_DISTANCE:
            score["distance_reward"] = float(distance_reward)
        return score
