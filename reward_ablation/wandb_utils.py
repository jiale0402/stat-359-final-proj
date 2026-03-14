"""Weights & Biases helpers for online experiment logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional
import os


def init_wandb_run(
    project: str,
    config: Dict[str, Any],
    entity: Optional[str] = None,
    name: Optional[str] = None,
    group: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    job_type: str = "grpo_reward_ablation",
) -> Any:
    """Initialize an online W&B run."""
    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is required for online logging. Install dependencies first."
        ) from exc

    return wandb.init(
        project=project,
        entity=entity,
        name=name,
        group=group,
        tags=list(tags) if tags else None,
        job_type=job_type,
        config=config,
        mode="online",
    )


def log_report_evidence(
    wandb_run: Any,
    report_pdf_path: Optional[str],
    report_image_paths: Optional[Iterable[str]],
) -> None:
    """Attach the report PDF and reward-curve figures to the W&B run."""
    import wandb

    image_payload = {}
    for image_path in report_image_paths or []:
        if os.path.exists(image_path):
            image_key = Path(image_path).stem.replace(" ", "_")
            image_payload[f"evidence/{image_key}"] = wandb.Image(image_path)
    if image_payload:
        wandb_run.log(image_payload, step=0)

    if report_pdf_path and os.path.exists(report_pdf_path):
        artifact = wandb.Artifact("report_supporting_evidence", type="report")
        artifact.add_file(report_pdf_path, name=os.path.basename(report_pdf_path))
        for image_path in report_image_paths or []:
            if os.path.exists(image_path):
                artifact.add_file(image_path, name=os.path.basename(image_path))
        wandb_run.log_artifact(artifact)
