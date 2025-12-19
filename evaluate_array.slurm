#!/bin/bash
#SBATCH --job-name=qa_eval
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/eval_%A_%a.out
#SBATCH --error=logs/eval_%A_%a.err

set -euo pipefail

# Q&A evaluation job-array runner (simple).
#
# Each array task selects 1 model from models.txt, starts an Ollama server on a
# unique local port (so multiple tasks can share the same node), then queries
# Ollama for every question in qa_pairs.json and writes outputs to results/.

mkdir -p logs results

module load ollama/0.6.0-GCCcore-13.3.0-CUDA-12.6.0
module load Python/3.13.5-GCCcore-14.3.0

MODELS_FILE=${MODELS_FILE:-models.txt}
QA_PATH=${QA_PATH:-qa_pairs.json}
OUT_DIR=${OUT_DIR:-results}
LANGS=${LANGS:-"en no"}
NUM_Q=${NUM_Q:-1000}

# Unique per-task port so array tasks can run on the same node.
BASE_PORT=${BASE_PORT:-11434}
PORT=$((BASE_PORT + SLURM_ARRAY_TASK_ID))
if [[ "$PORT" -ge 65535 ]]; then
  echo "Computed port out of range: $PORT (BASE_PORT=$BASE_PORT)" >&2
  echo "Set BASE_PORT lower, or reduce array size." >&2
  exit 2
fi

if [[ ! -f "$MODELS_FILE" ]]; then
  echo "models file not found: $MODELS_FILE" >&2
  exit 2
fi
if [[ ! -f "$QA_PATH" ]]; then
  echo "QA dataset not found: $QA_PATH" >&2
  echo "Provide it locally (not committed). See README." >&2
  exit 2
fi

MODEL=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MODELS_FILE" | tr -d '\r')
if [[ -z "$MODEL" ]]; then
  echo "No model found for task id $SLURM_ARRAY_TASK_ID (check $MODELS_FILE)" >&2
  exit 2
fi

echo "Node: ${SLURMD_NODENAME:-unknown}"
echo "Model: $MODEL"
echo "Dataset: $QA_PATH"
echo "OUT_DIR: $OUT_DIR"
echo "LANGS: $LANGS"
echo "Ollama port: $PORT"

if [[ -f ".venv/bin/activate" ]]; then
  # Prefer local venv if present.
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Start Ollama server locally on the allocated node.
# IMPORTANT: bind to localhost and use a per-task port to avoid collisions.
export OLLAMA_HOST="127.0.0.1:${PORT}"
ollama serve > "logs/ollama_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${PORT}.log" 2>&1 &
OLLAMA_PID=$!

# Ensure we shut down Ollama on exit.
cleanup() {
  if kill -0 "$OLLAMA_PID" 2>/dev/null; then
    kill "$OLLAMA_PID" || true
  fi
}
trap cleanup EXIT

# Wait for the server to accept connections.
python - <<'PY'
import os, socket, time
host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
host, port = host.split(":")
addr = (host, int(port))
for _ in range(60):
    try:
        with socket.create_connection(addr, timeout=1):
            break
    except OSError:
        time.sleep(1)
else:
  raise SystemExit(f"ollama server did not become ready on {addr[0]}:{addr[1]}")
PY

# Use the local Ollama server (client expects scheme).
export OLLAMA_HOST="http://127.0.0.1:${PORT}"

# Fail early if the model isn't already available.
# (Users should import/pull models ahead of time.)
if ! ollama show "$MODEL" >/dev/null 2>&1; then
  echo "Model not found in local Ollama registry: $MODEL" >&2
  echo "Import/pull it before running this job (see README)." >&2
  exit 2
fi

for LANG in $LANGS; do
  echo "Running evaluate.py lang=$LANG"
  python evaluate.py \
    --model "$MODEL" \
    --lang "$LANG" \
    --num-q "$NUM_Q" \
    --qa-path "$QA_PATH" \
    --out-dir "$OUT_DIR" \
    --ollama-host "$OLLAMA_HOST"
done
