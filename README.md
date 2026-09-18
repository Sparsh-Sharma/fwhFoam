<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="doc/assets/fwhfoam-logo-dark.svg">
    <img src="doc/assets/fwhfoam-logo.svg" alt="fwhFoam" width="420">
  </picture>
</p>

<p align="center">
  <b>A Ffowcs Williams–Hawkings acoustic solver for OpenFOAM,<br>with on-surface acoustic source localization.</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/OpenFOAM-v2306-1d7ea3">
  <img src="https://img.shields.io/badge/C%2B%2B-14-00599C?logo=cplusplus&logoColor=white">
  <img src="https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/License-GPL--3.0-green">
  <img src="https://img.shields.io/badge/verified-analytic%20~0.1%25-brightgreen">
  <img src="https://img.shields.io/badge/convergence-2nd%20order-brightgreen">
</p>

---

`fwhFoam` predicts far-field aerodynamic sound from an unsteady OpenFOAM
flow field. It implements **Farassat's formulation 1A** of the Ffowcs
Williams–Hawkings (FW-H) equation with a source-time-dominant
("advanced-time") algorithm, for both

- **impermeable** integration surfaces (solid wall patches — loading and
  thickness noise of the body), and
- **permeable** integration surfaces (an off-body control surface that
  encloses the sources — captures quadrupole/turbulence noise as well),

from a **single, unified code path**: the impermeable case is the exact
`u·n = v·n` special case of the permeable formulation. A **uniform mean
flow** (wind-tunnel frame) is handled in closed form through the
Garrick-triangle emission-time relation, with no retarded-time
root-finding.

From the same permeable-surface data, `fwhFoam` also computes **acoustic
source localization on the integration surface** — a surface sound-power
density `σ(x)` (the diffraction-filter / surface-power theory of Delfs &
Ruck) that maps where the radiated sound originates, with the power identity
`∮σ dS = P_far-field` as a built-in consistency check.

It ships as:

| Component | What it is |
|-----------|------------|
| `libfwhFoam` | a runtime **function object** `fwh` — compute observer signals on the fly during any OpenFOAM run |
| `fwhSolve` | an **offline utility** — recompute observer signals from stored surface data, with different observers/ambient conditions, without re-running the CFD |
| `pyfwh` | a **Python package** — surface-data I/O, analytic reference fields, spectra, surface generators, and the **`localization` (σ) source-mapping** module |
| `sigma_localize` | a **source-localization driver** — turns permeable surface data into a σ map (VTK/CSV) plus the `∮σ dS = P` identity check |
| `tests/` | an **analytic verification suite**, a **convergence study**, and a **σ-localization verification** |
| `tutorials/` | a runnable **2D cylinder Aeolian-tone** case |

---

## Why another FW-H code?

Most open FW-H tools are either buried inside a larger solver, limited to
impermeable surfaces, or lack a transparent, independently verifiable
implementation. `fwhFoam` is deliberately small, documented against the
equations it solves, and **verified to ~0.1 % against analytic solutions**
(stationary monopole, dipole, and a convected monopole at Mach 0.3), with
demonstrated second-order spatial and temporal convergence. Everything
needed to reproduce those results is in this repository.

---

## Method summary

The acoustic pressure at an observer is split into thickness and loading
contributions,

```
4π p'_T(x,t) = ∫_S [ ρ₀ U̇ₙ / (r(1−Mᵣ)²)  +  ρ₀ Uₙ c₀(Mᵣ−M²) / (r²(1−Mᵣ)³) ]_ret dS
4π p'_L(x,t) = ∫_S [ L̇ᵣ / (c₀ r(1−Mᵣ)²) + (Lᵣ−L_M)/(r²(1−Mᵣ)²) + Lᵣ(Mᵣ−M²)/(r²(1−Mᵣ)³) ]_ret dS
```

with the permeable-surface source vectors

```
Uᵢ = vᵢ + (ρ/ρ₀)(uᵢ − vᵢ)          (mass flux)
Lᵢ = p' n̂ᵢ + ρ uᵢ (uₙ − vₙ)         (momentum flux)
```

evaluated at the emission (retarded) time. `u` is the fluid velocity and
`v` the surface velocity relative to the quiescent medium; `n̂` is the
outward unit normal; `M = v/c₀` is the surface Mach vector and
`Mᵣ = M·r̂`. For a solid wall `uₙ = vₙ` and the classical impermeable form
is recovered exactly.

Rather than search for retarded times, each source time-level is
propagated *forward* to the observer: its contribution is deposited onto a
uniform observer-time grid by exact piecewise-linear resampling between
consecutive (Doppler-shifted) arrival times. Time derivatives use
second-order central differences over three stored source levels. A
uniform mean flow `U₀` is treated by a Galilean transformation to the
medium-rest frame; the emission-to-reception delay then follows from the
closed-form Garrick-triangle solution.

See [`doc/theory.md`](doc/theory.md) for the full derivation, assumptions
and limitations.

---

## Requirements

- **OpenFOAM** v2306 (or a nearby ESI/`.com` release). Developed and
  tested with `openfoam/2306`.
- A C++14 compiler (as used by your OpenFOAM build).
- **Python ≥ 3.9** with **NumPy** (for `pyfwh`, the tests and plotting).

## Build

```bash
# with an OpenFOAM environment sourced (wmake on PATH):
./Allwmake
```

This builds `libfwhFoam.so` into `$FOAM_USER_LIBBIN` and `fwhSolve` into
`$FOAM_USER_APPBIN`. Clean with `./Allwclean`.

> **HPC note (DLR CARA/spack).** The spack `openfoam/2306` module hardcodes
> another user's `WM_PROJECT_USER_DIR`. Use the provided
> [`etc/build-cara.sh`](etc/build-cara.sh), which overrides the user
> install paths and loads `gcc/12.3.0` before `openfoam/2306`.

## Quick start (runtime function object)

Add to `system/controlDict`:

```cpp
functions
{
    fwh1
    {
        type            fwh;
        libs            (fwhFoam);

        c0              340.29;      // speed of sound [m/s]
        rho0            1.225;       // ambient density [kg/m^3]
        U0              (0 0 0);     // uniform mean flow (wind-tunnel frame)

        surface
        {
            type        patches;    // impermeable: solid wall patches
            patches     (cylinder);
        }

        observers
        {
            mic1 { position (0 10 0); }
            mic2 { position (10 0 0); }
        }

        executeControl  timeStep;
        executeInterval 8;           // dt_src = 8 * deltaT (must be constant)
        writeControl    writeTime;
    }
}
```

For a **permeable** surface, swap the `surface` sub-dictionary:

```cpp
surface
{
    type            sampled;
    surfaceDict
    {
        type        sampledTriSurfaceMesh;
        surface     fwhSurface.stl;     // in constant/triSurface/
        source      cells;
    }
    interpolationScheme cellPoint;
    checkOrientation    true;           // auto-orient a closed surface
}
```

Observer signals are written to
`postProcessing/fwh1/acousticData/observer_<name>.dat` with columns
`t  p'  p'_thickness  p'_loading` and a header giving the valid time
window (the interval over which every surface element has contributed).

**Important:** the source sampling interval must be constant, so use a
**fixed** `deltaT` (disable `adjustTimeStep`).

## Offline reprocessing (`fwhSolve`)

Set `writeSurfaceData true;` in the function object to also dump the raw
surface data (`surfaceData.fwh`). Then, with an `fwhSolveDict`:

```cpp
dataFile    "postProcessing/fwh1/acousticData/surfaceData.fwh";
outputDir   "fwhSolve-output";
c0          340.29;
rho0        1.225;
U0          (0 0 0);
observers   { farMic { position (0 100 0); } }
```

```bash
fwhSolve -dict fwhSolveDict
```

This lets you move observers, change ambient conditions or the mean-flow
speed, and reprocess — all without touching the CFD.

## Source localization (σ maps)

With a **permeable** surface and `writeSurfaceData true`, the raw
per-processor data can be turned into a surface sound-power localization
map:

```bash
python tools/sigma_localize.py \
    --data "postProcessing/fwh1/acousticData/surfaceData_proc*.fwh" \
    --fmin 100 --fmax 5000 --c0 340.29 --rho0 1.225 --out sigma_map
```

This writes `sigma_map.vtk` (open in ParaView) and `sigma_map.csv`, and
prints the power identity `∮σ dS` vs `P_far-field` per spectral line and
band-integrated — a built-in check that the localization and the far-field
prediction are consistent. See [`doc/theory.md`](doc/theory.md) §8.

## Verify the build

```bash
# analytic FW-H verification (monopole/dipole/convected monopole)
python tests/analytic/run_analytic.py --fwhsolve $FOAM_USER_APPBIN/fwhSolve --case all
# spatial + temporal convergence
python tests/analytic/convergence.py  --fwhsolve $FOAM_USER_APPBIN/fwhSolve
# sigma-localization verification (power identity on a sphere)
python tests/localization/verify_sigma.py
# Python unit tests
pytest tools/tests
```

Expected: all analytic FW-H cases pass at a few ×0.1 % relative L2 error;
the convergence study reports observed order ≈ 2 in both space and time;
the σ power identity holds to a few tenths of a percent.

## Tutorial

[`tutorials/cylinderAeolianTone`](tutorials/cylinderAeolianTone) — laminar
flow past a 2D cylinder at Re = 100, producing the Aeolian tone at the
vortex-shedding (Strouhal) frequency, with a dipole directivity peaking
normal to the flow. Run locally with `./Allrun`, or on a SLURM cluster
with `sbatch run_cylinder.slurm`.

## Repository layout

```
src/fwhFoam/            core library (formulation + function object)
applications/fwhSolve/  offline solver
tools/pyfwh/            Python companion package
tests/analytic/         analytic verification + convergence study
tutorials/              runnable OpenFOAM cases
doc/                    theory and format documentation
paper/                  SoftwareX manuscript sources
```

## Limitations (v1)

- Static (non-moving, non-deforming) integration surface.
- Constant source-time sampling interval.
- Subsonic surface/mean-flow Mach number (`|U₀| < c₀`).
- Quadrupole volume sources are captured only through a permeable
  surface (no explicit volume integration).

## Citing

If you use `fwhFoam` in academic work, please cite the accompanying paper
(see [`CITATION.cff`](CITATION.cff) and [`paper/`](paper)).

## License

GPL-3.0-or-later. `fwhFoam` is an unofficial extension to OpenFOAM and is
not approved or endorsed by OpenCFD Ltd or the OpenFOAM Foundation. See
[`LICENSE`](LICENSE).
