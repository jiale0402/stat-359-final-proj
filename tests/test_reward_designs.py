"""Tests for reward-ablation reward functions."""

from __future__ import annotations

import pytest

from reward_ablation.reward_designs import (
    RewardAblationVerifier,
    RewardDesign,
)


def test_h2_process_and_format_rewards_contribute_to_total_reward() -> None:
    verifier = RewardAblationVerifier(RewardDesign.H2_BINARY_PROCESS_FORMAT)
    generated_text = (
        "<think>\n"
        "Step 1: 10 - 3 = 7\n"
        "Step 2: 5 + 7 = 12\n"
        "</think>\n"
        "Final Result: 12"
    )

    score = verifier.score_response(generated_text, 12)

    assert score["binary_reward"] == pytest.approx(1.0)
    assert score["process_reward"] > 0.0
    assert score["format_reward"] > 0.0
    assert score["reward"] == pytest.approx(
        0.8 * score["binary_reward"]
        + 0.15 * score["process_reward"]
        + 0.05 * score["format_reward"]
    )


def test_h3_distance_reward_gives_partial_credit_for_nearby_answers() -> None:
    verifier = RewardAblationVerifier(RewardDesign.H3_BINARY_DISTANCE)

    score = verifier.score_response("Final Result: 10", 12)

    assert score["binary_reward"] == pytest.approx(0.0)
    assert 0.0 < score["distance_reward"] < 1.0
    assert score["reward"] == pytest.approx(0.1 * score["distance_reward"])
