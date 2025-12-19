# Q&A (History QA evaluation)

This folder contains scripts to:

1. Run a history Q&A benchmark against local **Ollama** models (answers only)
2. Grade the answers using the **IDUN** chat completions API (T/F)
3. Generate summary plots and simple statistical comparisons

> Note: the directory name contains an `&`. In shell commands, use quotes: `cd "Q&A"`.

## Setup

### Python environment

From this folder:

```bash
cd "Q&A"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

You can set these in your shell or in a local `.env` file (not committed).

- `OLLAMA_HOST` (default: `http://localhost:11434`)
- `MODEL` (required for evaluation unless you pass `--model`)
- `LANG` (default: `en`)
- `NUM_Q` (default: `1000`)
- `QA_PATH` (default: `qa_pairs.json`)
- `OUT_DIR` (default: `results`)
- `IDUN_API_KEY` (required for grading)

## Dataset (qa_pairs.json)

The question/answer dataset file is **not committed to GitHub**.

To run evaluations you must provide it locally.

- Put the dataset file in this folder (recommended): `Q&A/qa_pairs.json`
- Or point to it elsewhere with `--qa-path ...` or by setting `QA_PATH`

### Expected JSON schema

The scripts expect a JSON object with a top-level `qa_pairs` list:

```json
{
	"qa_pairs": [
		{"question": "...", "answer": "..."}
	]
}
```

If you have a different schema, adapt/convert it into the above format.

## Workflow

### 1) Generate model answers (no grading)

Produces:
- `results/<model>-<lang>.json`
- `results/<model>-<lang>.log`

Example:

```bash
python evaluate.py --model deepseek-r1:14b --lang en --num-q 1000 --qa-path qa_pairs.json --out-dir results
```

### 2) Grade answers (adds `grade` + accuracy)

Reads `results/*.json` and updates them in-place by adding `grade` (`T`/`F`).
Also writes/updates matching `results/<stem>.log` files with:
- `accuracy`
- `correct_answers`
- `invalid_answers`

```bash
export IDUN_API_KEY="..."
python aggregate.py --out-dir results --max-workers 100
```

### 3) Backfill GPU count (optional)

Some older `.log` files may be missing `num_gpus`.

```bash
python add_gpu_count.py --results-dir results
```

### 4) Generate plots

Overall accuracy per model:

```bash
python generate_graph.py --out-dir results --output model_performance.png
```

Per-family accuracy graphs:

```bash
python graph_families.py --out-dir results --graph-dir family_graphs
```

Pareto frontier (accuracy vs GPU-hours):

```bash
python pareto_analysis.py --results-dir results --out-dir .
```

## Analysis utilities

### McNemar test

Flexible version (supports `.json` or `.txt` files listing multiple JSONs):

```bash
python calculate_mcnemar.py results/modelA_en.json results/modelB_en.json
```

Minimal version (assumes aligned single JSONs):

```bash
python calculate_b_c.py results/modelA_en.json results/modelB_en.json
```

### Identify universally easy/hard questions

```bash
python identify_questions.py --out-dir results --reference results/deepseek-r1:671b_en.json --output all_models_incorrect_q.json
```

### Extract incorrect questions from one run

```bash
python find_false_questions.py --input results/deepseek-r1:671b_en.json --output deepseek671b_incorrect_q.json
```

### Remove problematic questions (mutates JSON files)

```bash
python remove_questions.py --questions 167,828,869 --glob "results/*.json" --dry-run
python remove_questions.py --questions 167,828,869 --glob "results/*.json"
```

### Rename files to normalize suffix conventions

```bash
python sanitize_names.py --results-dir results --contains mimir --dry-run
python sanitize_names.py --results-dir results --contains mimir
```

## Data and outputs

This repository includes existing `results/` and generated graphs (e.g. `family_graphs/`, `*.png`).
Do **not** delete them when preparing the project for GitHub.
