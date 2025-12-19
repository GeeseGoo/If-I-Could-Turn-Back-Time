"""Run a QA evaluation against an Ollama model.

Reads `qa_pairs.json` (or another file with the same schema) and produces:

- `results/<model>-<lang>.json`: list of QA items with model answers
- `results/<model>-<lang>.log`: metadata about the run (duration, SLURM info, etc.)

This script intentionally does *not* grade answers. Use `aggregate.py` (grader)
to populate `grade` fields and accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from ollama import Client


THINK_REGEX = re.compile(r"<think>.*</think>\n?", flags=re.DOTALL)


@dataclass(frozen=True)
class EvalConfig:
    model: str
    lang: str
    num_q: int
    qa_path: Path
    out_dir: Path
    ollama_host: str
    max_retries: int
    retry_sleep_s: float
    strip_think: bool


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run QA evaluation (generate answers) using Ollama")
    parser.add_argument("--model", default=os.getenv("MODEL"), help="Ollama model name (or set MODEL)")
    parser.add_argument(
        "--models",
        default=os.getenv("MODELS", ""),
        help="Optional space-separated model list; used only if --model/MODEL is unset",
    )
    parser.add_argument("--lang", default=os.getenv("LANG", "en"), help="Language tag used in filenames")
    parser.add_argument("--num-q", type=int, default=int(os.getenv("NUM_Q", "1000")), help="Number of questions")
    parser.add_argument("--qa-path", type=Path, default=Path(os.getenv("QA_PATH", "qa_pairs.json")))
    parser.add_argument("--out-dir", type=Path, default=Path(os.getenv("OUT_DIR", "results")))
    parser.add_argument("--ollama-host", default=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-sleep-s", type=float, default=10.0)
    parser.add_argument("--no-strip-think", action="store_true", help="Do not strip <think>...</think> blocks")
    return parser


def _pick_model(explicit_model: str | None, models_str: str) -> str:
    if explicit_model:
        return explicit_model
    models = [m for m in models_str.split() if m]
    if models:
        return models[0]
    raise SystemExit("No model specified. Provide --model or set MODEL (or MODELS).")


def _sanitize_filename_component(name: str) -> str:
    return name[name.rfind("/") + 1 :]


def _load_pairs(path: Path, num_q: int) -> list[dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "qa_pairs" not in raw:
        raise ValueError(f"Invalid QA file schema: expected object with 'qa_pairs': {path}")
    pairs = raw["qa_pairs"]
    if not isinstance(pairs, list):
        raise ValueError(f"Invalid QA file schema: 'qa_pairs' must be a list: {path}")
    return pairs[:num_q]


def _get_system_prompt(lang: str) -> str:
    is_no = lang.lower().startswith("no")
    if is_no:
        return (
            "Du er ekspert på norsk språk og verdenshistorie før 1940. "
            "Svar kort (1–2 setninger) og bruk kun kunnskap før 1940. Svar på norsk."
        )
    return (
        "You are an expert in answering history quizzes using only knowledge from up to 1940. "
        "Answer in one or two sentences, using only knowledge up to 1940."
    )


def run_eval(config: EvalConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    client = Client(host=config.ollama_host)

    pairs = _load_pairs(config.qa_path, config.num_q)
    system_prompt = _get_system_prompt(config.lang)

    qa_results: list[dict[str, Any]] = []
    start = time.time()

    print(f"starting eval: model={config.model} lang={config.lang} num_q={len(pairs)}")
    for idx, pair in enumerate(pairs):
        question = str(pair.get("question", ""))
        ref_answer = str(pair.get("answer", ""))

        ans: str
        last_exc: Exception | None = None
        for attempt in range(1, config.max_retries + 1):
            try:
                ans = client.chat(
                    model=config.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": question},
                    ],
                )["message"]["content"]

                if config.strip_think:
                    ans = re.sub(THINK_REGEX, "", str(ans)).strip()
                else:
                    ans = str(ans).strip()
                break
            except Exception as exc:
                last_exc = exc
                print(f"error on question {idx}, attempt {attempt}/{config.max_retries}: {exc}")
                if attempt < config.max_retries:
                    time.sleep(config.retry_sleep_s)
        else:
            ans = f"ERROR: {last_exc}" if last_exc else "ERROR"

        if (idx + 1) % 25 == 0 or idx == 0:
            print("evaluating question", idx, "time:", datetime.now())

        qa_results.append(
            {
                "question_num": idx,
                "question": question,
                "model_ans": ans,
                "correct_ans": ref_answer,
            }
        )

    duration_s = round(time.time() - start, 3)
    result_log = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "model": config.model,
        "lang": config.lang,
        "num_q": config.num_q,
        "duration_s": duration_s,
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "slurm_array_task_id": os.getenv("SLURM_ARRAY_TASK_ID"),
        "node": os.getenv("SLURMD_NODENAME"),
    }
    print("finished eval in", duration_s, "seconds")
    return result_log, qa_results


def main() -> int:
    load_dotenv()
    args = build_argparser().parse_args()
    model = _pick_model(args.model, args.models)
    config = EvalConfig(
        model=model,
        lang=str(args.lang).lower(),
        num_q=int(args.num_q),
        qa_path=Path(args.qa_path),
        out_dir=Path(args.out_dir),
        ollama_host=str(args.ollama_host),
        max_retries=int(args.max_retries),
        retry_sleep_s=float(args.retry_sleep_s),
        strip_think=not bool(args.no_strip_think),
    )

    result_log, qa_results = run_eval(config)
    file_stem = f"{_sanitize_filename_component(config.model)}-{config.lang}"

    (config.out_dir / f"{file_stem}.log").write_text(
        json.dumps(result_log, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (config.out_dir / f"{file_stem}.json").write_text(
        json.dumps(qa_results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

