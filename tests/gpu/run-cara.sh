#!/bin/bash
# Run the GPU kernel verification + benchmark on a CARA A100 node.
# Invoke as:  ssh cara "bash fwhFoam/tests/gpu/run-cara.sh [args...]"
set -e
module load cuda/12.2.1 >/dev/null 2>&1

HERE="$(cd "$(dirname "$0")" && pwd)"
srun -p rome-a100 -A 2002498 --gres=gpu:1 --cpus-per-task=16 --mem=200G \
     --time=00:45:00 \
     "$HOME/fwh-venv/bin/python" -u "$HERE/benchmark_gpu.py" "$@"
