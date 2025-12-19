"""Rename result files to normalize naming conventions.

Historically, some file names used '-' before the language suffix (e.g. mimir-...-en)
while other scripts assume '_' (e.g. mimir-..._en). This script helps normalize.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def build_argparser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Rename files in results/ to a consistent pattern")
  parser.add_argument("--results-dir", type=Path, default=Path("results"))
  parser.add_argument(
    "--contains",
    default="mimir",
    help="Only rename files containing this substring (default: mimir)",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print changes without renaming",
  )
  return parser


def main() -> int:
  args = build_argparser().parse_args()
  results_dir: Path = args.results_dir
  if not results_dir.exists():
    raise SystemExit(f"results dir not found: {results_dir}")

  renamed = 0
  for name in os.listdir(results_dir):
    if args.contains and args.contains not in name:
      continue

    index = name.rfind("-")
    if index == -1:
      continue

    new_name = name[:index] + "_" + name[index + 1 :]
    if new_name == name:
      continue

    src_path = results_dir / name
    dest_path = results_dir / new_name
    if dest_path.exists():
      print(f"skip (dest exists): {dest_path}")
      continue

    if args.dry_run:
      print(f"would rename: {src_path.name} -> {dest_path.name}")
    else:
      src_path.rename(dest_path)
      print(f"renamed: {src_path.name} -> {dest_path.name}")
    renamed += 1

  print(f"done: renamed {renamed} files")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
