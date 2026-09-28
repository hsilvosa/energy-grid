"""Render a benchmark table from current backtest JSON reports."""

from __future__ import annotations

import json
import sys
from glob import glob
from pathlib import Path


def main(paths: list[str]) -> int:
    if not paths:
        print("Usage: python scripts/benchmark_table.py reports/*.json", file=sys.stderr)
        return 2
    rows: list[str] = []
    expanded = [match for pattern in paths for match in (glob(pattern) or [pattern])]
    for path_name in expanded:
        report = json.loads(Path(path_name).read_text(encoding="utf-8"))
        required = {"country_code", "horizon_steps", "steps_per_day", "overall_metrics"}
        if required - report.keys():
            print(
                f"Skipping legacy report without forecast-origin metadata: {path_name}",
                file=sys.stderr,
            )
            continue
        metrics = report["overall_metrics"]
        horizon_hours = report["horizon_steps"] * 24 / report["steps_per_day"]
        rows.append(
            f"| {report['country_code']} | {report['target']} | {horizon_hours:g} | "
            f"{metrics['mae']:.2f} | {report['baseline_7d_metrics']['mae']:.2f} | "
            f"{report['calibrated_coverage'] * 100:.1f}% | {report['fold_count']} |"
        )
    if not rows:
        return 1
    print("| Market | Target | Lead (h) | MAE | 7d baseline MAE | P10-P90 coverage | Folds |")
    print("|---|---|---:|---:|---:|---:|---:|")
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
