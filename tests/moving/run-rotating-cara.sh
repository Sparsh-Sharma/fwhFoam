#!/bin/bash
# Run the rotating-source (moving-surface) validation on CARA.
# Invoke as:  ssh cara "bash fwhFoam/tests/moving/run-rotating-cara.sh [args...]"
set -e
module load gcc/12.3.0 openfoam/2306 >/dev/null 2>&1
export WM_PROJECT_USER_DIR="$HOME/OpenFOAM/shar_sp-v2306"
export FOAM_USER_APPBIN="$WM_PROJECT_USER_DIR/platforms/$WM_OPTIONS/bin"
export FOAM_USER_LIBBIN="$WM_PROJECT_USER_DIR/platforms/$WM_OPTIONS/lib"
export LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:$LD_LIBRARY_PATH"

HERE="$(cd "$(dirname "$0")" && pwd)"
WD="${FWH_WORKDIR:-/scratch/ws25/shar_sp-P2/fwhrot}"
"$HOME/fwh-venv/bin/python" "$HERE/verify_rotating_source.py" \
    --fwhsolve "$FOAM_USER_APPBIN/fwhSolve" \
    --workdir "$WD" \
    --save "$WD/rotating_signals.npz" \
    "$@"
