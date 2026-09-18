#!/bin/bash
# Run the convergence study on CARA (shared scratch workdir).
set -e
module load gcc/12.3.0 openfoam/2306 >/dev/null 2>&1
export WM_PROJECT_USER_DIR="$HOME/OpenFOAM/$USER-v2306"
export FOAM_USER_APPBIN="$WM_PROJECT_USER_DIR/platforms/$WM_OPTIONS/bin"
export FOAM_USER_LIBBIN="$WM_PROJECT_USER_DIR/platforms/$WM_OPTIONS/lib"
export LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:$LD_LIBRARY_PATH"
VENV="$HOME/fwh-venv/bin/python"
HERE="$(cd "$(dirname "$0")" && pwd)"
"$VENV" "$HERE/convergence.py" \
    --fwhsolve "$FOAM_USER_APPBIN/fwhSolve" \
    --workdir "${FWH_WORKDIR:-/scratch/ws25/$USER-P2/fwhconv}" "$@"
