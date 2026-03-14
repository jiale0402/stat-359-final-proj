"""Summarize reward-ablation runs from trainer logs."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Iterable, List, Tuple


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _resolve_run_paths(path: str) -> Tuple[str, str | None]:
    if os.path.isdir(path):
        log_path = os.path.join(path, "grpo_training_log.json")
        manifest_path = os.path.join(path, "experiment_manifest.json")
    else:
        log_path = path
        manifest_path = None
        if os.path.basename(path) == "experiment_manifest.json":
            manifest_path = path
            log_path = os.path.join(os.path.dirname(path), "grpo_training_log.json")
    if not os.path.exists(log_path):
        raise FileNotFoundError(log_path)
    if manifest_path is not None and not os.path.exists(manifest_path):
        manifest_path = None
    return log_path, manifest_path


def _extract_series(log_entries: Iterable[dict], metric_name: str) -> List[Tuple[int, float]]:
    series: List[Tuple[int, float]] = []
    for entry in log_entries:
        metric_value = entry.get("metrics", {}).get(metric_name)
        if metric_value is None:
            continue
        series.append((int(entry["step"]), float(metric_value)))
    return series


def _step_to_threshold(series: List[Tuple[int, float]], threshold: float) -> int | None:
    for step, value in series:
        if value >= threshold:
            return step
    return None


def _late_mean(series: List[Tuple[int, float]], tail_size: int) -> float | None:
    if not series:
        return None
    tail = series[-tail_size:]
    return sum(value for _, value in tail) / len(tail)


def _summarize_run(
    path: str,
    threshold: float,
    tail_size: int,
) -> Dict[str, Any]:
    log_path, manifest_path = _resolve_run_paths(path)
    log_entries = _load_json(log_path)
    manifest = _load_json(manifest_path) if manifest_path else {}

    eval_series = _extract_series(log_entries, "val_reward_rate")
    if not eval_series:
        eval_series = _extract_series(log_entries, "reward_rate")

    summary = {
        "run_path": os.path.dirname(log_path),
        "reward_design": manifest.get("run_spec", {}).get("reward_design", "unknown"),
        "reward_formula": manifest.get("reward_formula"),
        "metric_name": "val_reward_rate" if _extract_series(log_entries, "val_reward_rate") else "reward_rate",
        "step_to_threshold": _step_to_threshold(eval_series, threshold),
        "late_mean": _late_mean(eval_series, tail_size),
        "num_points": len(eval_series),
    }
    return summary


def _print_table(rows: List[Dict[str, Any]]) -> None:
    headers = ["reward_design", "step_to_threshold", "late_mean", "num_points", "run_path"]
    widths = {
        header: max(
            len(header),
            *(len(str(row.get(header, ""))) for row in rows),
        )
        for header in headers
    }
    header_line = "  ".join(header.ljust(widths[header]) for header in headers)
    print(header_line)
    print("  ".join("-" * widths[header] for header in headers))
    for row in rows:
        print(
            "  ".join(
                str(row.get(header, "")).ljust(widths[header])
                for header in headers
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze reward-ablation runs")
    parser.add_argument("paths", nargs="+", help="Run directories or log/manifest paths")
    parser.add_argument("--threshold", type=float, default=0.30)
    parser.add_argument("--tail-size", type=int, default=5)
    parser.add_argument("--output-json", type=str, default=None)
    args = parser.parse_args()

    rows = [_summarize_run(path, args.threshold, args.tail_size) for path in args.paths]
    _print_table(rows)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(rows, handle, indent=2)


if __name__ == "__main__":
    main()
