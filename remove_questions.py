"""Remove specific questions from results JSON files.

This mutates the target JSON files in-place.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path


DEFAULT_QUESTIONS = {167, 828, 869}


def build_argparser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Remove selected question IDs from results JSON files")
  parser.add_argument(
    "--questions",
    default=",".join(str(q) for q in sorted(DEFAULT_QUESTIONS)),
    help="Comma-separated question IDs (default: 167,828,869)",
  )
  parser.add_argument(
    "--glob",
    dest="files_glob",
    default=str(Path(os.getenv("OUT_DIR", "results")) / "*.json"),
    help="Glob pattern for files to edit",
  )
  parser.add_argument(
    "--match",
    choices=["auto", "index", "question_num"],
    default="auto",
    help="Whether to match by list index, question_num field, or auto (prefer question_num)",
  )
  parser.add_argument("--dry-run", action="store_true", help="Do not write files")
  return parser


def main() -> int:
  args = build_argparser().parse_args()
  questions = {int(x.strip()) for x in str(args.questions).split(",") if x.strip()}
  targets = [Path(p) for p in sorted(glob.glob(args.files_glob))]
  if not targets:
    raise SystemExit(f"no files matched: {args.files_glob}")

  for path in targets:
    print(path)
    raw_text = path.read_text(encoding="utf-8")
    if not raw_text.strip():
      continue
    data = json.loads(raw_text)
    if not isinstance(data, list):
      continue

    out = []
    removed = 0
    for idx, item in enumerate(data):
      if args.match == "index":
        key = idx
      elif args.match == "question_num":
        key = int(item.get("question_num", idx))
      else:
        key = int(item.get("question_num", idx))

      if key in questions:
        removed += 1
        continue
      out.append(item)

    if args.dry_run:
      print(f"dry-run: would remove {removed} items from {path}")
      continue

    path.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"finished removing {removed} questions for {path}")

  return 0


if __name__ == "__main__":
  raise SystemExit(main())

