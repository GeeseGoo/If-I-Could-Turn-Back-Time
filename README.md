# Q&A (History QA evaluation on HPC with Slurm + Ollama)

This folder contains scripts and Slurm job files to:

1. Run a history Q&A benchmark against **Ollama** models (answers only)
2. Grade the answers using the **IDUN** chat completions API (T/F)
3. Generate plots and simple statistical comparisons

## Important notes

- The dataset file (`qa_pairs.json`) is **not committed to GitHub**.
- The primary workflow is **Slurm `sbatch`** with a job array over `models.txt`.
- Models must be available in Ollama **before** you run the evaluation array.

## Dataset (qa_pairs.json)

Provide the dataset locally.

- Recommended location: `Q&A/qa_pairs.json`
- Or set `QA_PATH` / pass `--qa-path` to point elsewhere.

Expected JSON schema:

```json
{
  "qa_pairs": [
    {"question": "...", "answer": "..."}
  ]
}
```

## Models list (models.txt)

`models.txt` must contain **one Ollama model name per line**. Example:

```text
llama3.1:8b
deepseek-r1:14b
qwen3:32b
```

## Setup

From this folder:

```bash
cd "Q&A"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you run via Slurm, make sure the Slurm scripts match your cluster:

- Edit `#SBATCH --account=...`
- Edit `#SBATCH --partition=...`
- Adjust `--gres=gpu:...`, time, memory, etc.

Environment variables (optional):

- `QA_PATH` (default: `qa_pairs.json`)
- `OUT_DIR` (default: `results`)
- `NUM_Q` (default: `1000`)
- `LANGS` (default in Slurm job: `"en no"`)
- `IDUN_API_KEY` (required for grading)

## Workflow (recommended): Slurm job array

### Step 0 — Make sure Ollama has the models

Before running the evaluation array, ensure each model in `models.txt` exists in your Ollama store.

On the machine/environment where you can run Ollama commands:

```bash
ollama show <model>
# if missing:
ollama pull <model>
```

The evaluation job (`evaluate_array.slurm`) will **fail early** if `ollama show` does not find the model.

### Step 1 — Run evaluation across all models

This submits one Slurm array task per model line in `models.txt`. Each task:

- allocates a GPU node
- starts `ollama serve` in the task (in parallel across array tasks)
- runs `evaluate.py` for each language in `LANGS`
- writes outputs to `results/`

Port note: array tasks may share the same physical node. To avoid port conflicts,
`evaluate_array.slurm` uses a per-task port: `PORT = BASE_PORT + SLURM_ARRAY_TASK_ID`.

Outputs per model+language:

- `results/<model>-<lang>.json` (model answers)
- `results/<model>-<lang>.log` (metadata like runtime + Slurm IDs)

Submit (from `Q&A/`):

```bash
mkdir -p logs results
N=$(wc -l < models.txt)
sbatch --array=0-$(($N-1)) evaluate_array.slurm
```

### Step 2 — Grade answers (adds `grade` + accuracy)

Grading uses the IDUN API to label each answer `T`/`F`, then writes accuracy to the matching `.log`.

Option A (interactive):

```bash
export IDUN_API_KEY="..."
python aggregate.py --out-dir results --max-workers 100
```

Option B (Slurm batch):

```bash
export IDUN_API_KEY="..."
sbatch aggregate.slurm
```

## Plots

Overall model performance:

```bash
python generate_graph.py --out-dir results --output model_performance.png
```

Per-family graphs:

```bash
python graph_families.py --out-dir results --graph-dir family_graphs
```

Pareto frontier (accuracy vs GPU-hours):

```bash
python pareto_analysis.py --results-dir results --out-dir .
```

## Utilities

- McNemar test (flexible):

```bash
python calculate_mcnemar.py results/modelA_en.json results/modelB_en.json
```

- McNemar (minimal, aligned JSONs only):

```bash
python calculate_b_c.py results/modelA_en.json results/modelB_en.json
```

- Identify universally hard questions:

```bash
python identify_questions.py --out-dir results --reference results/deepseek-r1:671b_en.json --output all_models_incorrect_q.json
```

## Data and outputs

This repository intentionally includes existing `results/` and graphs.
Do **not** delete them.
