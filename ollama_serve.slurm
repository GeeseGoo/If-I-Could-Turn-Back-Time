#!/bin/bash
#SBATCH --job-name=ollama_serve
#SBATCH --partition=GPUQ
#SBATCH --account=ie-idi
#SBATCH --time=0-06:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --output=logs/ollama_serve_%j.out
#SBATCH --error=logs/ollama_serve_%j.err

set -euo pipefail
mkdir -p logs

module load ollama/0.6.0-GCCcore-13.3.0-CUDA-12.6.0

# Bind on all interfaces so you can reach it from the same node / forwarded port.
export OLLAMA_HOST=0.0.0.0

echo "Starting ollama serve on node ${SLURMD_NODENAME:-unknown}"
ollama serve
