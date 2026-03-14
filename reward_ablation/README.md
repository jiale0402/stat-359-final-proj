# Reward Ablation

This package contains the GRPO reward-ablation experiments used for the arithmetic reasoning runs.

Reward settings:

- `h1_binary_only`
- `h2_binary_process_format`
- `h3_binary_distance`

Launch:

```bash
uv run python -m reward_ablation.run_reward_ablation --help
```

Summarize completed runs:

```bash
uv run python -m reward_ablation.analyze_reward_ablation --help
```
