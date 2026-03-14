"""Training helpers for GRPO reward-ablation experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
import os
import random
from typing import Any, Dict, Iterator, List, Optional, Tuple

from arithmetic_llm.arithmetic_tokenizer import (
    ArithmeticBPETokenizer,
)
from arithmetic_llm.data_loader import ArithmeticDataset
from arithmetic_llm.evaluator import eval_expression
from arithmetic_llm.generator import ExpressionGenerator
from arithmetic_llm.grpo_config import GRPOConfig
from arithmetic_llm.grpo_trainer import GRPOTrainer

from .reward_designs import (
    REWARD_DESIGNS,
    RewardAblationVerifier,
    RewardDesign,
    resolve_reward_design,
)
from .wandb_utils import init_wandb_run, log_report_evidence


@dataclass(frozen=True)
class RewardAblationRunSpec:
    """Experiment settings mirrored from the report."""

    reward_design: str
    data_mode: str
    num_samples: int
    train_ratio: float
    seed: int
    max_depth: int
    num_range: Tuple[int, int]
    filter_invalid_instruction: bool
    candidate_sub_batch_size: Optional[int]


def _batch_iter(items: List[dict], batch_size: int) -> Iterator[Tuple[List[str], List[int]]]:
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        prompts = [entry["prompt"] for entry in batch]
        ground_truth = [entry["ground_truth"] for entry in batch]
        yield prompts, ground_truth


def _load_instruction_pairs(
    instruction_corpus_path: str,
    tokenizer: ArithmeticBPETokenizer,
    validate_expressions: bool,
) -> List[dict]:
    dataset = ArithmeticDataset(
        corpus_path=instruction_corpus_path,
        tokenizer=tokenizer,
        mode="instruction",
    )
    return dataset.get_instruction_pairs(validate_expressions=validate_expressions)


def _generate_prompt_pool(
    num_samples: int,
    max_depth: int,
    num_range: Tuple[int, int],
    seed: int,
) -> List[dict]:
    state = random.getstate()
    random.seed(seed)
    try:
        generator = ExpressionGenerator(
            max_depth=max_depth,
            num_range=num_range,
            invalid_rate=0.0,
        )
        pairs: List[dict] = []
        while len(pairs) < num_samples:
            expression = generator.generate()
            result = eval_expression(expression)
            if result["answer"] == "ERROR":
                continue
            pairs.append(
                {
                    "expression": expression,
                    "prompt": result["problem"] + " <think>",
                    "ground_truth": int(result["answer"]),
                }
            )
        return pairs
    finally:
        random.setstate(state)


def _split_pairs(
    pairs: List[dict],
    train_ratio: float,
    seed: int,
) -> Tuple[List[dict], List[dict]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}")
    shuffled = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    split_idx = min(max(split_idx, 1), len(shuffled) - 1)
    return shuffled[:split_idx], shuffled[split_idx:]


def _build_run_dir(output_dir: str, reward_design: RewardDesign) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_name = f"{reward_design.value}_{timestamp}"
    run_dir = os.path.join(output_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def train_reward_ablation_experiment(
    tokenizer_path: str,
    sft_checkpoint_path: str,
    output_dir: str,
    config: GRPOConfig,
    reward_design: str | RewardDesign,
    instruction_corpus_path: Optional[str] = None,
    data_mode: str = "generated",
    num_samples: int = 50000,
    train_ratio: float = 0.9,
    seed: int = 359,
    max_depth: int = 5,
    num_range: Tuple[int, int] = (1, 20),
    filter_invalid_instruction: bool = True,
    candidate_sub_batch_size: Optional[int] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_group: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
    report_pdf_path: Optional[str] = None,
    report_image_paths: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Train one H1/H2/H3 reward-ablation run."""
    reward_design = resolve_reward_design(reward_design)
    tokenizer = ArithmeticBPETokenizer()
    tokenizer.load(tokenizer_path)

    if data_mode == "instruction":
        if instruction_corpus_path is None:
            raise ValueError("instruction_corpus_path is required for instruction mode")
        pairs = _load_instruction_pairs(
            instruction_corpus_path=instruction_corpus_path,
            tokenizer=tokenizer,
            validate_expressions=filter_invalid_instruction,
        )
    elif data_mode == "generated":
        pairs = _generate_prompt_pool(
            num_samples=num_samples,
            max_depth=max_depth,
            num_range=num_range,
            seed=seed,
        )
    else:
        raise ValueError("data_mode must be 'generated' or 'instruction'")

    if len(pairs) < 2:
        raise ValueError("need at least two prompt pairs for a train/validation split")

    train_pairs, val_pairs = _split_pairs(
        pairs=pairs,
        train_ratio=train_ratio,
        seed=seed,
    )

    run_dir = _build_run_dir(output_dir, reward_design)
    train_dataloader = list(_batch_iter(train_pairs, config.batch_size))
    val_dataloader = list(_batch_iter(val_pairs, config.batch_size))

    run_spec = RewardAblationRunSpec(
        reward_design=reward_design.value,
        data_mode=data_mode,
        num_samples=len(pairs),
        train_ratio=train_ratio,
        seed=seed,
        max_depth=max_depth,
        num_range=num_range,
        filter_invalid_instruction=filter_invalid_instruction,
        candidate_sub_batch_size=candidate_sub_batch_size,
    )
    manifest = {
        "report_reward_design": REWARD_DESIGNS[reward_design].label,
        "reward_formula": REWARD_DESIGNS[reward_design].formula,
        "run_spec": asdict(run_spec),
        "grpo_config": config.to_dict(),
        "dataset": {
            "train_size": len(train_pairs),
            "val_size": len(val_pairs),
        },
        "artifacts": {
            "training_log": os.path.join(run_dir, "grpo_training_log.json"),
            "final_checkpoint": os.path.join(run_dir, "final_model.pt"),
            "best_checkpoint": os.path.join(run_dir, "best_model.pt"),
            "report_pdf": report_pdf_path,
            "report_images": report_image_paths or [],
        },
    }

    wandb_run = None
    if wandb_project:
        wandb_config = {
            "reward_design": reward_design.value,
            "reward_formula": REWARD_DESIGNS[reward_design].formula,
            "output_dir": run_dir,
            "tokenizer_path": tokenizer_path,
            "sft_checkpoint_path": sft_checkpoint_path,
            "data_mode": data_mode,
            "num_samples": len(pairs),
            "train_ratio": train_ratio,
            "seed": seed,
            "max_depth": max_depth,
            "num_range": list(num_range),
            "report_pdf_path": report_pdf_path,
            "report_image_paths": report_image_paths or [],
            "grpo_config": config.to_dict(),
        }
        wandb_run = init_wandb_run(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name or os.path.basename(run_dir),
            group=wandb_group,
            tags=wandb_tags,
            config=wandb_config,
        )
        log_report_evidence(
            wandb_run=wandb_run,
            report_pdf_path=report_pdf_path,
            report_image_paths=report_image_paths,
        )

    verifier = RewardAblationVerifier(reward_design)
    trainer = GRPOTrainer(
        config=config,
        sft_checkpoint_path=sft_checkpoint_path,
        tokenizer=tokenizer,
        verifier=verifier,
        wandb_run=wandb_run,
        candidate_sub_batch_size=candidate_sub_batch_size,
    )

    num_train_batches = math.ceil(len(train_pairs) / config.batch_size)
    total_steps = math.ceil(
        num_train_batches / max(1, config.gradient_accumulation_steps)
    ) * config.num_epochs
    trainer.reset_optimizer_and_scheduler(total_steps=total_steps)
    manifest_path = os.path.join(run_dir, "experiment_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    try:
        result = trainer.train(
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            output_dir=run_dir,
        )
        manifest["training_result"] = result
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)

        if wandb_run is not None:
            wandb_run.summary["manifest_path"] = manifest_path
            wandb_run.summary["run_dir"] = run_dir

        return {
            "run_dir": run_dir,
            "manifest_path": manifest_path,
            **result,
        }
    finally:
        if wandb_run is not None:
            wandb_run.finish()
