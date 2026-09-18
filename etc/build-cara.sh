#!/bin/bash
# Build fwhFoam on DLR CARA (spack openfoam/2306 module).
# The spack module hardcodes another user's WM_PROJECT_USER_DIR, so the
# user locations are overridden here.
set -e

module load gcc/12.3.0 openfoam/2306 >/dev/null 2>&1

export WM_PROJECT_USER_DIR="$HOME/OpenFOAM/$USER-v2306"
export FOAM_USER_APPBIN="$WM_PROJECT_USER_DIR/platforms/$WM_OPTIONS/bin"
export FOAM_USER_LIBBIN="$WM_PROJECT_USER_DIR/platforms/$WM_OPTIONS/lib"
export LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:$LD_LIBRARY_PATH"
export PATH="$FOAM_USER_APPBIN:$PATH"

mkdir -p "$FOAM_USER_APPBIN" "$FOAM_USER_LIBBIN"

cd "$(dirname "$0")/.."
./Allwmake
