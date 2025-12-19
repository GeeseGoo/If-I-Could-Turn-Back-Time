#!/bin/bash
#SBATCH --partition=CPUQ
#SBATCH --account=ie-idi
#SBATCH --time=0-00:45:00     # 0 days and 15 minutes limit
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=32     # 2 CPU cores
#SBATCH --mem=32G              # 5 gigabytes memory
#SBATCH --output=logs/aggregate_%j.out
#SBATCH --error=logs/aggregate_%j.err
set -euo pipefail

mkdir -p logs

module load Python/3.13.5-GCCcore-14.3.0
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python aggregate.py > logs/aggregate.log
