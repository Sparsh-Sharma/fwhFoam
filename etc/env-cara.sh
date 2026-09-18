#!/bin/bash
# Source this (or run commands through it) for the fwhFoam runtime
# environment on DLR CARA:  bash etc/env-cara.sh <command...>
module load gcc/12.3.0 openfoam/2306 >/dev/null 2>&1

export WM_PROJECT_USER_DIR="$HOME/OpenFOAM/$USER-v2306"
export FOAM_USER_APPBIN="$WM_PROJECT_USER_DIR/platforms/$WM_OPTIONS/bin"
export FOAM_USER_LIBBIN="$WM_PROJECT_USER_DIR/platforms/$WM_OPTIONS/lib"
export LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:$LD_LIBRARY_PATH"
export PATH="$FOAM_USER_APPBIN:$PATH"

if [ $# -gt 0 ]; then
    exec "$@"
fi
