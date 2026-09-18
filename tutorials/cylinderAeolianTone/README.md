# Tutorial: Aeolian tone of a circular cylinder (Re = 100)

Laminar, two-dimensional flow past a circular cylinder sheds a von Kármán
vortex street. The periodic shedding produces a fluctuating lift force and
hence a **dipole** acoustic field (the *Aeolian tone*) at the vortex-
shedding (Strouhal) frequency, radiating most strongly normal to the flow.
This case demonstrates the `fwh` runtime function object on an
**impermeable** wall surface.

## Setup

| Quantity | Value |
|----------|-------|
| Solver | `pimpleFoam` (incompressible, laminar) |
| Free-stream velocity `U` | 1 m/s |
| Cylinder diameter `D` | 1 m |
| Kinematic viscosity `ν` | 0.01 m²/s |
| Reynolds number `Re = UD/ν` | 100 |
| Expected Strouhal number `St = fD/U` | ≈ 0.164–0.17 |
| Mesh | circular O-grid, 4 blocks, ~28 000 cells |
| Time step | 0.0025 s (fixed — FW-H needs a constant interval) |

A small transverse component in the initial `U` field seeds the shedding;
it decays while the shedding becomes self-sustaining.

The `fwh` function object (see `system/controlDict`) integrates the
cylinder wall pressure to four observers at radius 75 D and also dumps the
raw surface data (`writeSurfaceData true`) for offline reprocessing.
Because the CFD is incompressible, the FW-H prediction is the low-Mach
loading (dipole) noise — the standard Curle/FW-H approach for the Aeolian
tone; the choice of `c0` sets only the acoustic wavelength, not the tone
frequency.

## Run

Locally:

```bash
./Allrun
```

On a SLURM cluster:

```bash
sbatch run_cylinder.slurm
```

Results appear in
`postProcessing/fwhWall/acousticData/observer_<name>.dat`.

## Post-process

```bash
python plot_cylinder.py --case . --out cylinder.pdf --U 1.0 --D 1.0
```

This plots the far-field pressure spectrum (peaking at the shedding
Strouhal number, cross-checked against the lift-coefficient spectrum) and
the RMS-pressure directivity (a dipole lobe pattern peaking normal to the
flow).

## Optional: a permeable control surface

To compute the noise on a **permeable** off-body surface instead of (or in
addition to) the wall, generate a cylindrical control surface and add a
second `fwh` function object:

```bash
python makeFwhSurface.py            # writes constant/triSurface/fwhCylinder.stl
```

```cpp
fwhPermeable
{
    type    fwh;   libs (fwhFoam);
    c0 340.0;  rho0 1.225;  U0 (0 0 0);
    surface
    {
        type        sampled;
        surfaceDict
        {
            type            sampledTriSurfaceMesh;
            surface         fwhCylinder.stl;
            source          cells;
        }
        interpolationScheme cellPoint;
        checkOrientation    true;
    }
    observers { mic_up { position (0 75 0); } }
    executeControl timeStep;  executeInterval 8;
}
```

For a 2D case the control surface is an open cylindrical shell (no end
caps), so the flux through the empty direction cancels by symmetry.

## Offline reprocessing

With `writeSurfaceData true`, per-processor raw files
`surfaceData_proc*.fwh` are written. Recompute observer signals (e.g. at a
new observer array, or with a different `c0`) without re-running the CFD:

```bash
fwhSolve -dict fwhSolveDict
```

An example `fwhSolveDict` is provided in this directory.
