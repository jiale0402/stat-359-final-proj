# GRPO Reward Ablation for Arithmetic LLMs

Transformer-based arithmetic language model training stack with supervised fine-tuning, LoRA adapters, GRPO optimization, and reward-ablation experiments.

## Layout

- `arithmetic_llm/`: model, tokenizer, data generation, training, evaluation, LoRA, and GRPO code.
- `reward_ablation/`: H1/H2/H3 experiment entrypoints and log analysis tools.
- `tests/`: focused regression tests for reward logic and trainer behavior.
- `report/`: final report PDF and reward-curve figures.

## Setup

```bash
uv sync --dev
```

## Core Commands

Base pipeline:

```bash
uv run python -m arithmetic_llm.run_foundational_training --help
uv run python -m arithmetic_llm.run_instruction_training --help
uv run python -m arithmetic_llm.run_grpo_training --help
uv run python -m arithmetic_llm.run_evaluation --help
```

Reward-ablation runs:

```bash
uv run python -m reward_ablation.run_reward_ablation \
  --reward-design h1_binary_only \
  --tokenizer data/tokenizer \
  --sft-checkpoint models/instruction_YYYYMMDD_HHMMSS/best_model.pt \
  --output-dir models/reward_ablation \
  --data-mode generated \
  --num-samples 50000 \
  --train-ratio 0.9 \
  --num-epochs 3 \
  --batch-size 1 \
  --num-candidates 8 \
  --gradient-accumulation-steps 16 \
  --eval-every 50 \
  --wandb-project arithmetic-llm-grpo
```

Run comparison:

```bash
uv run python -m reward_ablation.analyze_reward_ablation \
  models/reward_ablation/h1_binary_only_YYYYMMDD_HHMMSS \
  models/reward_ablation/h2_binary_process_format_YYYYMMDD_HHMMSS \
  models/reward_ablation/h3_binary_distance_YYYYMMDD_HHMMSS
```

## Report

- [Final report PDF](./report/final_report.pdf)
- [Accuracy curves](./report/reward_accuracy_curves.png)
- [H2 reward components](./report/h2_reward_component_curves.png)
- [H3 reward components](./report/h3_reward_component_curves.png)
