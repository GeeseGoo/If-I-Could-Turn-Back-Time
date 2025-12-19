"""Extract incorrectly answered questions from a graded results JSON file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def build_argparser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Extract incorrect questions from a results JSON")
  parser.add_argument(
    "--input",
    type=Path,
    default=Path(os.getenv("OUT_DIR", "results")) / "deepseek-r1:671b_en.json",
    help="Path to a graded results JSON (list of QA items)",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=Path("deepseek671b_incorrect_q.json"),
    help="Output JSON path",
  )
  parser.add_argument(
    "--grade-key",
    default="grade",
    help="Key holding grade values (default: grade)",
  )
  return parser


def main() -> int:
  args = build_argparser().parse_args()
  input_path: Path = args.input
  if not input_path.exists():
    raise SystemExit(f"input not found: {input_path}")

  raw = json.loads(input_path.read_text(encoding="utf-8"))
  if not isinstance(raw, list):
    raise SystemExit("expected input JSON to be a list")

  incorrect: list[dict[str, Any]] = []
  for item in raw:
    grade = str(item.get(args.grade_key, "")).strip().upper()
    if grade != "T":
      incorrect.append(item)

  args.output.write_text(json.dumps(incorrect, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
  print(f"wrote {len(incorrect)} incorrect items to {args.output}")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())


