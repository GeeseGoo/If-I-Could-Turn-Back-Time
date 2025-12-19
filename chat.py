#!/usr/bin/env python3
"""Minimal CLI client for the IDUN chat completions endpoint."""

from __future__ import annotations

import argparse
import json
from typing import Any

import requests


IDUN_API_URL = "https://idun-llm.hpc.ntnu.no/api/chat/completions"


def build_argparser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Send a single chat completion request to IDUN")
  parser.add_argument("api_key", help="Bearer token")
  parser.add_argument("model", help="Model name (e.g., openai/gpt-oss-120b)")
  parser.add_argument("question", help="User message content")
  parser.add_argument("--timeout-s", type=float, default=60.0)
  parser.add_argument("--system", default=None, help="Optional system prompt")
  return parser


def chat_with_model(*, token: str, model: str, question: str, timeout_s: float, system: str | None) -> dict[str, Any]:
  headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
  messages: list[dict[str, str]] = []
  if system:
    messages.append({"role": "system", "content": system})
  messages.append({"role": "user", "content": question})
  payload = {"model": model, "messages": messages}
  resp = requests.post(IDUN_API_URL, headers=headers, json=payload, timeout=timeout_s)
  return resp.json()


def main() -> int:
  args = build_argparser().parse_args()
  response = chat_with_model(
    token=args.api_key,
    model=args.model,
    question=args.question,
    timeout_s=float(args.timeout_s),
    system=args.system,
  )
  print(json.dumps(response, indent=2, ensure_ascii=False))
  return 0


if __name__ == "__main__":
  raise SystemExit(main())