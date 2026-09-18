#!/bin/bash
#SBATCH --job-name=fwhCyl
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

# fwhFoam tutorial: 2D cylinder Aeolian tone, run on CARA.
# Submit from the case directory:  sbatch run_cylinder.slurm

set -e
module load gcc/12.3.0 openfoam/2306 >/dev/null 2>&1

export WM_PROJECT_USER_DIR="$HOME/OpenFOAM/$USER-v2306"
export FOAM_USER_APPBIN="$WM_PROJECT_USER_DIR/platforms/$WM_OPTIONS/bin"
export FOAM_USER_LIBBIN="$WM_PROJECT_USER_DIR/platforms/$WM_OPTIONS/lib"
export LD_LIBRARY_PATH="$FOAM_USER_LIBBIN:$LD_LIBRARY_PATH"
export PATH="$FOAM_USER_APPBIN:$PATH"

. "$WM_PROJECT_DIR/bin/tools/RunFunctions"

restore0Dir
runApplication blockMesh
runApplication checkMesh -constant

runApplication decomposePar
srun -n "$SLURM_NTASKS" pimpleFoam -parallel > log.pimpleFoam 2>&1
runApplication reconstructPar -latestTime

echo "Simulation complete."
