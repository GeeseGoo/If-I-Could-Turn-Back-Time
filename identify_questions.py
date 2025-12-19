"""Identify question IDs that *all* graded runs answered correctly/incorrectly.

This is useful for spotting systematically easy/hard items.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Find question IDs all models got correct/incorrect")
    parser.add_argument("--out-dir", type=Path, default=Path(os.getenv("OUT_DIR", "results")))
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(os.getenv("OUT_DIR", "results")) / "deepseek-r1:671b_en.json",
        help="Reference results file used to extract full question objects for incorrect IDs",
    )
    parser.add_argument("--output", type=Path, default=Path("all_models_incorrect_q.json"))
    parser.add_argument("--grade-key", default="grade")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    result_files = sorted(args.out_dir.glob("*.json"))
    if not result_files:
        raise SystemExit(f"no results JSON files found in {args.out_dir}")

    all_correct: set[int] | None = None
    all_incorrect: set[int] | None = None

    for p in result_files:
        raw = json.loads(p.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            continue

        correct_ids: set[int] = set()
        incorrect_ids: set[int] = set()
        for idx, item in enumerate(raw):
            qid = int(item.get("question_num", idx))
            grade = str(item.get(args.grade_key, "")).strip().upper()
            if grade == "T":
                correct_ids.add(qid)
            else:
                # Treat missing/unknown grades as incorrect by default.
                incorrect_ids.add(qid)

        if all_correct is None:
            all_correct = correct_ids
            all_incorrect = incorrect_ids
        else:
            all_correct &= correct_ids
            all_incorrect &= incorrect_ids

    all_correct = all_correct or set()
    all_incorrect = all_incorrect or set()
    print("answers all models answered correctly", sorted(all_correct))
    print("answers all models answered incorrectly", sorted(all_incorrect))

    if not args.reference.exists():
        raise SystemExit(f"reference file not found: {args.reference}")

    ref_raw = json.loads(args.reference.read_text(encoding="utf-8"))
    if not isinstance(ref_raw, list):
        raise SystemExit("reference JSON must be a list")

    incorrect: list[dict[str, Any]] = []
    for item in ref_raw:
        qid = int(item.get("question_num", -1))
        if qid in all_incorrect:
            incorrect.append(item)

    args.output.write_text(json.dumps(incorrect, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(incorrect)} items to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



