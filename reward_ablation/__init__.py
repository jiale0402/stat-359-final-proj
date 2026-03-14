"""Reward-ablation project for Arithmetic LLM GRPO training."""

from importlib import import_module

__all__ = [
    "RewardAblationVerifier",
    "RewardDesign",
    "train_reward_ablation_experiment",
]


def __getattr__(name: str):
    if name in {"RewardAblationVerifier", "RewardDesign"}:
        module = import_module(".reward_designs", __name__)
        return getattr(module, name)
    if name == "train_reward_ablation_experiment":
        module = import_module(".experiment", __name__)
        return getattr(module, name)
    raise AttributeError(name)
