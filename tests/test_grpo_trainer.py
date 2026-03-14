"""Regression tests for GRPO trainer masking and batching behavior."""

from __future__ import annotations

import torch
import torch.nn as nn

from arithmetic_llm.grpo_config import GRPOConfig
from arithmetic_llm.grpo_trainer import GRPOTrainer


class DummyTokenizer:
    """Tokenizer stub with variable-length prompt encodings."""

    token2id = {"<bos>": 0, "<pad>": 9}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        mapping = {
            "short": [1],
            "long": [1, 2, 3],
        }
        ids = mapping[text]
        if add_special_tokens:
            return [self.token2id["<bos>"], *ids]
        return ids

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        special_ids = {self.token2id["<bos>"], self.token2id["<pad>"]}
        visible = [str(token_id) for token_id in ids if not (skip_special_tokens and token_id in special_ids)]
        return " ".join(visible)


class PositionAwareModel(nn.Module):
    """Model that emits a deterministic token based on sequence position."""

    def __init__(self, vocab_size: int = 12):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1))
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape
        logits = torch.full(
            (batch_size, seq_len, self.vocab_size),
            fill_value=-1000.0,
            device=input_ids.device,
        )
        for position in range(seq_len):
            logits[:, position, min(position + 1, self.vocab_size - 1)] = 1000.0 + self.anchor
        return logits


class TinyPolicy(nn.Module):
    """Small model used to exercise optimizer stepping."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))


def test_compute_kl_divergence_excludes_prompt_tokens() -> None:
    trainer = GRPOTrainer(
        config=GRPOConfig(device="cpu", top_k=1),
        policy_model=TinyPolicy(),
        reference_model=TinyPolicy(),
        tokenizer=DummyTokenizer(),
    )
    policy_logits = torch.zeros(1, 4, 2)
    reference_logits = torch.zeros(1, 4, 2)

    policy_logits[0, 0, 0] = 5.0
    policy_logits[0, 1, 1] = 5.0

    unmasked_kl = trainer.compute_kl_divergence(policy_logits, reference_logits)
    masked_kl = trainer.compute_kl_divergence(
        policy_logits,
        reference_logits,
        token_mask=torch.tensor([[0, 0, 1]], dtype=torch.bool),
    )

    assert unmasked_kl.item() > 0.0
    assert masked_kl.item() == 0.0


def test_generate_candidates_uses_last_non_pad_token_logits() -> None:
    trainer = GRPOTrainer(
        config=GRPOConfig(
            device="cpu",
            top_k=1,
            top_p=1.0,
            temperature=1.0,
            max_gen_length=5,
        ),
        policy_model=PositionAwareModel(),
        reference_model=PositionAwareModel(),
        tokenizer=DummyTokenizer(),
    )

    texts, _ = trainer.generate_candidates(
        prompts=["short", "long"],
        num_candidates=1,
        temperature=1.0,
        top_k=1,
        top_p=1.0,
        max_gen_length=5,
    )

    assert texts[0][0].split()[-1] == "2"
    assert texts[1][0].split()[-1] == "4"


def test_train_flushes_partial_accumulation_step(tmp_path) -> None:
    trainer = GRPOTrainer(
        config=GRPOConfig(
            device="cpu",
            batch_size=1,
            num_epochs=1,
            gradient_accumulation_steps=2,
            log_every=1,
            save_every=100,
            eval_every=100,
            top_k=1,
        ),
        policy_model=TinyPolicy(),
        reference_model=TinyPolicy(),
        tokenizer=DummyTokenizer(),
    )

    def fake_train_step(prompts, ground_truth, do_step=True, loss_scale=1.0):
        (trainer.policy_model.weight * loss_scale).backward()
        return {
            "policy_loss": 1.0,
            "kl_divergence": 0.0,
            "total_loss": 1.0,
            "avg_reward": 0.5,
            "reward_rate": 0.5,
        }

    trainer.train_step = fake_train_step  # type: ignore[method-assign]
    train_dataloader = [
        (["p1"], [1]),
        (["p2"], [2]),
        (["p3"], [3]),
    ]

    result = trainer.train(train_dataloader=train_dataloader, output_dir=str(tmp_path))

    assert result["global_step"] == 2
