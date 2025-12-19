"""Generate a single bar chart comparing model accuracies across languages.

Reads `results/*.log` (JSON) and expects filenames of the form `<model>_<lang>.log`.
Writes a PNG file (default: `model_performance.png`).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def split_model_lang(stem: str) -> tuple[str, str] | None:
    """Split a log filename stem into (model, lang).

    Supports historical naming variants like:
    - `<model>_en`
    - `<model>-en`
    - `<model>:en`
    """

    for lang in ("en", "no"):
        for sep in ("_", "-", ":"):
            suffix = f"{sep}{lang}"
            if stem.endswith(suffix) and len(stem) > len(suffix):
                return stem[: -len(suffix)], lang
    return None


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an overall model performance bar chart")
    parser.add_argument("--out-dir", type=Path, default=Path(os.getenv("OUT_DIR", "results")))
    parser.add_argument("--output", type=Path, default=Path("model_performance.png"))
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    out_dir: Path = args.out_dir
    log_files = sorted(out_dir.glob("*.log"))
    families: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))

    print(f"Found {len(log_files)} log files to process...")
    for log_path in log_files:
        try:
            stem = log_path.stem
            parsed = split_model_lang(stem)
            if parsed is None:
                print(f"Warning: skipping malformed log file name: {log_path.name}", file=sys.stderr)
                continue

            model_name, lang = parsed
            if "-all-gpus" in model_name:
                model_name = model_name.replace("-all-gpus", "")

            log_data = json.loads(log_path.read_text(encoding="utf-8"))
            accuracy = float(log_data.get("accuracy", 0.0))
            families[model_name][lang] = {"accuracy": accuracy}
        except Exception as exc:
            print(f"Warning: could not process file {log_path}: {exc}", file=sys.stderr)

    if not families:
        print("No results found in .log files.", file=sys.stderr)
        return 1

    fig, ax = plt.subplots(layout="constrained")
    width = 0.35
    models = sorted(families.keys())
    x = np.arange(len(models))

    en_scores = [families[m].get("en", {}).get("accuracy", 0.0) * 100 for m in models]
    no_scores = [families[m].get("no", {}).get("accuracy", 0.0) * 100 for m in models]

    ax.bar(x - width / 2, en_scores, width, label="English Prompt")
    ax.bar(x + width / 2, no_scores, width, label="Norwegian Prompt")
    ax.legend()
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlabel("Models")
    ax.set_title("Model Performance on the Dataset")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    plt.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output)
    print(f"Graph saved to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())