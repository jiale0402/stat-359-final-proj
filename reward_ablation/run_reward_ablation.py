"""CLI entry point for reward-ablation experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from arithmetic_llm.grpo_config import GRPOConfig

from .experiment import train_reward_ablation_experiment
from .reward_designs import RewardDesign


_EVIDENCE_DIR = Path(__file__).resolve().parent.parent / "report"
_DEFAULT_REPORT_PDF = _EVIDENCE_DIR / "final_report.pdf"
_DEFAULT_REPORT_IMAGES = [
    _EVIDENCE_DIR / "reward_accuracy_curves.png",
    _EVIDENCE_DIR / "h2_reward_component_curves.png",
    _EVIDENCE_DIR / "h3_reward_component_curves.png",
]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run GRPO reward-ablation experiments")
    parser.add_argument(
        "--reward-design",
        type=str,
        choices=[design.value for design in RewardDesign],
        required=True,
        help="Reward design from the report",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Path to tokenizer directory",
    )
    parser.add_argument(
        "--sft-checkpoint",
        type=str,
        required=True,
        help="Path to the SFT checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for reward-ablation run artifacts",
    )
    parser.add_argument(
        "--data-mode",
        type=str,
        choices=["generated", "instruction"],
        default="generated",
        help="Whether to train from a generated prompt pool or an instruction corpus",
    )
    parser.add_argument(
        "--instruction-corpus",
        type=str,
        default=None,
        help="Instruction corpus JSONL, required for instruction mode",
    )
    parser.add_argument("--num-samples", type=int, default=50000)
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=359)
    parser.add_argument("--max-depth", type=int, default=5)
    parser.add_argument("--num-range-min", type=int, default=1)
    parser.add_argument("--num-range-max", type=int, default=20)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--num-candidates", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--kl-penalty-coef", type=float, default=0.05)
    parser.add_argument("--max-gen-length", type=int, default=511)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument(
        "--candidate-sub-batch-size",
        type=int,
        default=None,
        help="Optional candidate-processing sub-batch size",
    )
    parser.add_argument(
        "--filter-invalid-instruction",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Validate instruction corpus expressions before GRPO",
    )
    parser.add_argument("--wandb-project", type=str, default=None)
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=None,
        help="Optional W&B tags",
    )
    parser.add_argument(
        "--report-pdf",
        type=str,
        default=str(_DEFAULT_REPORT_PDF),
        help="Report PDF to attach as a W&B artifact",
    )
    parser.add_argument(
        "--report-images",
        nargs="*",
        default=[str(path) for path in _DEFAULT_REPORT_IMAGES],
        help="Reward-curve figures to attach as W&B images",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if not os.path.exists(args.tokenizer):
        raise FileNotFoundError(args.tokenizer)
    if not os.path.exists(args.sft_checkpoint):
        raise FileNotFoundError(args.sft_checkpoint)
    if args.data_mode == "instruction" and not args.instruction_corpus:
        raise ValueError("--instruction-corpus is required for instruction mode")
    if args.num_range_min > args.num_range_max:
        raise ValueError("num-range-min must be <= num-range-max")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)

    config = GRPOConfig(
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        gradient_clip=args.gradient_clip,
        save_every=args.save_every,
        eval_every=args.eval_every,
        num_candidates=args.num_candidates,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        kl_penalty_coef=args.kl_penalty_coef,
        max_gen_length=args.max_gen_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_every=args.log_every,
    )
    config.validate()

    train_reward_ablation_experiment(
        tokenizer_path=args.tokenizer,
        sft_checkpoint_path=args.sft_checkpoint,
        output_dir=args.output_dir,
        config=config,
        reward_design=args.reward_design,
        instruction_corpus_path=args.instruction_corpus,
        data_mode=args.data_mode,
        num_samples=args.num_samples,
        train_ratio=args.train_ratio,
        seed=args.seed,
        max_depth=args.max_depth,
        num_range=(args.num_range_min, args.num_range_max),
        filter_invalid_instruction=args.filter_invalid_instruction,
        candidate_sub_batch_size=args.candidate_sub_batch_size,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_group=args.wandb_group,
        wandb_tags=args.wandb_tags,
        report_pdf_path=args.report_pdf,
        report_image_paths=args.report_images,
    )


if __name__ == "__main__":
    main()
