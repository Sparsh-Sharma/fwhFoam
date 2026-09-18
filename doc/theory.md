# fwhFoam — theory and implementation notes

This note documents the equations `fwhFoam` solves, the algorithm, and
the assumptions and limitations of the current version. It is intended to
be read alongside the source in `src/fwhFoam/fwhFormulation1A.{H,C}`.

## 1. The Ffowcs Williams–Hawkings equation

The FW-H equation (Ffowcs Williams & Hawkings, 1969) recasts the
Navier–Stokes equations as an inhomogeneous wave equation for the
pressure perturbation `p' = p − p₀` in an unbounded quiescent medium of
density `ρ₀` and sound speed `c₀`, with sources on and outside a control
surface `S` defined by `f(x,t)=0` (with `f>0` outside, `∇f = n̂`):

```
□² p'(x,t) =  ∂/∂t [ Q_i n̂_i δ(f) ]        (monopole / thickness)
            − ∂/∂x_i [ L_ij n̂_j δ(f) ]      (dipole / loading)
            + ∂²/∂x_i∂x_j [ T_ij H(f) ]      (quadrupole)
```

where `H` is the Heaviside function, `δ` the Dirac delta, `T_ij` the
Lighthill stress tensor, and for a **permeable** surface moving with
velocity `v` in a flow with fluid velocity `u`:

```
Q_i  = ρ₀ v_i + ρ(u_i − v_i)                         (mass-flux vector)
L_ij = P_ij + ρ u_i (u_j − v_j)
P_ij = (p − p₀) δ_ij − τ_ij  ≈  p' δ_ij               (viscous stress τ neglected)
```

For a solid impermeable wall `u_i n̂_i = v_i n̂_i`, the mass-flux term
reduces to `Q_i n̂_i = ρ₀ v_i n̂_i` and the loading term to
`L_ij n̂_j = p' n̂_i` — the classical Curle/solid-surface form. `fwhFoam`
implements the permeable form and obtains the impermeable case as its
`u_n = v_n` limit, so both use one code path.

## 2. Farassat's formulation 1A

Neglecting the volume quadrupole (captured instead by choosing a
permeable surface that encloses the turbulent region), the free-space
solution is the retarded-time surface integral. Farassat's formulation 1A
(Farassat 2007; Brentner & Farassat 1998) writes it with the time
derivatives taken *inside* the integral and expressed through the source
motion, giving the thickness (`T`) and loading (`L`) pressures used in the
code:

```
4π p'_T(x,t) = ∫_S [ ρ₀(U̇_n + U_ṅ) / (r(1−M_r)²) ]_ret dS
             + ∫_S [ ρ₀ U_n (r Ṁ_r + c₀(M_r − M²)) / (r²(1−M_r)³) ]_ret dS

4π p'_L(x,t) = (1/c₀) ∫_S [ L̇_r / (r(1−M_r)²) ]_ret dS
             + ∫_S [ (L_r − L_M) / (r²(1−M_r)²) ]_ret dS
             + (1/c₀) ∫_S [ L_r (r Ṁ_r + c₀(M_r − M²)) / (r²(1−M_r)³) ]_ret dS
```

with

- `U_i = v_i + (ρ/ρ₀)(u_i − v_i)` — the mass-flux velocity,
- `L_i = p' n̂_i + ρ u_i (u_n − v_n)` — the loading vector,
- `r` the distance from the (retarded) source to the observer, `r̂` the
  unit radiation vector,
- `M = v/c₀` the surface Mach vector, `M_r = M·r̂`, `M² = M·M`,
- subscript `n` a projection on `n̂`, subscript `r` on `r̂`,
  `L_M = L·M`,
- a dot denoting the source-time derivative.

**Current version.** `fwhFoam` v1 targets a **static** integration surface
(`v = 0` in the surface's own frame; motion enters only through a uniform
mean flow, see §4). Then `ṅ = 0`, `Ṁ = 0`, and the terms simplify to the
forms coded in `emitLevel()`:

```
4π p'_T = ∫_S ρ₀ [ U̇_n /(r(1−M_r)²) + U_n c₀(M_r−M²)/(r²(1−M_r)³) ]_ret dS
4π p'_L = ∫_S [ L̇_r /(c₀ r(1−M_r)²) + (L_r−L_M)/(r²(1−M_r)²)
               + L_r c₀(M_r−M²)/(r²(1−M_r)³) ]_ret dS
```

Time derivatives `U̇`, `L̇` use second-order central differences over three
stored source-time levels.

## 3. Advanced-time ("source-time-dominant") algorithm

Instead of solving the implicit retarded-time equation
`τ = t − r(τ)/c₀` for each observer time `t`, `fwhFoam` marches in source
time. For each source element at emission time `τ`, the signal arrives at
the observer at

```
t = τ + T,     T = emission-to-reception delay.
```

For a medium at rest `T = r/c₀ = |x − y|/c₀`. Each element therefore
contributes a value that lands at a (generally non-grid) observer time
`τ + T`. Because `T` differs from element to element (and, under a mean
flow, is Doppler-shifted), the arrivals are non-uniform. `fwhFoam`
deposits each element's contribution onto a **uniform observer-time grid**
of spacing `Δt` (equal to the source sampling interval) by exact
piecewise-linear resampling between the arrival times of consecutive
source levels (`fwhObserverSignal::addSegment`). Summing over all elements
and all source levels builds the complete observer signal.

This *advanced-time* scheme (Casalino 2003) avoids root-finding, is
naturally single-pass and streaming (only three source levels are held in
memory), and parallelises trivially: each MPI rank integrates its own
faces onto its own copy of the grid, and the partial signals are summed
by a fixed-size reduction.

**Valid window.** A grid node is only complete once *every* surface
element has contributed to it. Locally this window is
`[τ_first + max_S T , τ_last + min_S T]`; globally it is the intersection
across ranks/files. Values outside it are start/stop transients and are
flagged by the `validWindow` in the output header.

## 4. Uniform mean flow (Garrick triangle)

For a wind-tunnel-type problem the CFD is run in a frame where the medium
convects at a uniform velocity `U₀` and the body is fixed. FW-H assumes a
quiescent medium, so `fwhFoam` transforms to the medium-rest frame by a
Galilean shift: the surface and every observer translate at `−U₀`. The
relative geometry is time-invariant (both translate together), and the
surface Mach vector becomes `M = −U₀/c₀`.

The emission-to-reception delay `T` in the moving-medium (equivalently,
translating-source) problem satisfies

```
|d − U₀ T| = c₀ T,     d = x_obs − y,
```

i.e. the quadratic

```
(c₀² − U₀²) T² + 2 (d·U₀) T − |d|² = 0,
```

with the physical (positive) root

```
T = [ −d·U₀ + √( (d·U₀)² + (c₀²−U₀²)|d|² ) ] / (c₀² − U₀²).
```

This is the closed-form **Garrick-triangle** solution
(`fwhFormulation1A::solveDelay`); the radiation vector is
`r̂ = (d − U₀T)/|d − U₀T|` and `r = c₀T`. It requires `|U₀| < c₀`
(subsonic). One can show `c₀T` equals the convected phase distance `R*`
of the convected-wave Green's function, so the scheme reproduces convected
monopole/dipole fields exactly (see §6).

## 5. Source terms in the medium-rest frame

With `v = −U₀` (surface) and `u_medium = u_cfd − U₀` (fluid), the
mass-flux and loading vectors evaluate to

```
U_i = −U₀ + (ρ/ρ₀) u_cfd
L_i = p' n̂_i + ρ (u_cfd − U₀)(u_cfd · n̂)
```

as coded in `addTimeLevel()`. A steady uniform flow through a permeable
surface (`p'=0`, `ρ=ρ₀`, `u_cfd=U₀`) gives `U_i = 0` and radiates nothing,
as it must — the mean-flow through-flux cancels identically. (When feeding
analytic data onto a test surface, the *total* fluid velocity `U₀ + u'`
must be supplied, not just the perturbation `u'`; otherwise a spurious
steady thickness term appears.)

## 6. Verification

The `tests/analytic` suite feeds exact acoustic fields onto a geodesic
icosphere (a permeable surface enclosing the sources) and compares the
predicted observer pressure to the analytic far field:

| Case | Exercises | Result |
|------|-----------|--------|
| Monopole (medium at rest) | thickness term `U̇_n` | ~0.1 % relative L2 |
| Dipole (medium at rest) | loading term `L_r`, `L̇_r` | ~0.1 % (radiating directions) |
| Convected monopole, M=0.3 | Garrick delay + convective `(M_r−M²)` terms | ~0.1 % |

`tests/analytic/convergence.py` shows the error decreasing at
**second order** in both surface resolution (`h ~ 1/√N_faces`) and time
step, consistent with midpoint-rule surface quadrature and central-
difference time derivatives on a smooth field.

## 7. Assumptions and limitations (v1)

1. **Static surface.** No mesh motion or surface deformation; rotor/rotating
   surfaces are not yet supported (the `ṅ`, `Ṁ` terms are dropped).
2. **Uniform mean flow only.** `U₀` is a single constant vector; sheared or
   non-uniform mean flows are outside the analogy's assumptions.
3. **Constant sampling interval.** Time derivatives and the resampling
   assume a fixed `Δt`; use a fixed solver time step.
4. **Subsonic.** `|U₀| < c₀`.
5. **No explicit quadrupole.** Volume sources are represented only by
   choosing a permeable surface enclosing them; there is no volume
   integration or quadrupole-correction term.
6. **Viscous surface stress neglected** in `L_ij` (standard for
   aeroacoustic FW-H at high Reynolds number).

## 8. Acoustic source localization (the sigma method)

Beyond the far-field pressure, `fwhFoam` can answer *where on the
integration surface the radiated sound comes from*. This uses the
diffraction-filter / surface-power-density theory of Delfs & Ruck (2024)
and its exact spectral formulation, applied to the permeable-surface
Cauchy data `(p, v_n)` that the FW-H surface already carries.

Working in the frequency domain, write the complex surface data at angular
frequency `omega = c0 k` as

```
phi = p_hat(xi; omega),     psi = dp/dn = -i omega rho0 v_n_hat(xi; omega).
```

The radiating field of this Cauchy pair has the far-field directivity
`F(xhat) = (1/4pi) closed_int [ i k (xhat·n) phi - psi ] e^{i k xhat·xi} dS`
and radiated power `P = (1/2 rho0 c0) closed_int |F|^2 dOmega`. Performing
the angular integrals in closed form yields an exact **surface
sound-power density**

```
sigma(xi) = (1/(32 pi^2 rho0 c0)) Re closed_int {
      k^2 (n·W·n') phi phi'*  -  i k (n·V) phi psi'*
    + i k (n'·V) psi phi'*    +  J0 psi psi'* } dS(xi')
```

with the closed-form kernels `W_ij = 4pi[delta_ij j1(z)/z - dh_i dh_j j2(z)]`,
`V_i = 4pi i j1(z) dh_i`, `J0 = 4pi j0(z)`, `z = k|xi-xi'|`,
`dh = (xi-xi')/|xi-xi'|`. By construction

```
closed_int sigma dS = P,
```

so the same surface data yield both a **localization map** `sigma(xi)` and,
independently, the far-field power — and the two must agree. `fwhFoam`
reports this identity as a built-in self-check. On a rigid/impermeable wall
`psi = 0` and only the pressure-pressure block survives, recovering the
classical rigid-wall density (with its known far-field completeness
defect). `sigma` is *signed*: radiated power is a two-point (pairwise)
quantity, so a surface element may carry locally negative density.

The method is implemented in `pyfwh.localization` and driven by
`tools/sigma_localize.py`, which reads the permeable surface-data files,
FFTs the per-face time histories, integrates `sigma` over a chosen
frequency band, and writes a surface map for visualization. It is verified
in `tests/localization/verify_sigma.py`: for an analytic monopole and
dipole enclosed by an icosphere, `closed_int sigma dS`, the far-field
power and the exact radiated power agree to a few tenths of a percent, and
`sigma` reproduces the correct spatial pattern (uniform for the monopole,
`cos^2` for the dipole).

This capability is what distinguishes `fwhFoam` from other OpenFOAM FW-H
implementations, which provide the far-field signal only.

## References

- J. E. Ffowcs Williams, D. L. Hawkings, *Sound generation by turbulence
  and surfaces in arbitrary motion*, Phil. Trans. R. Soc. A 264 (1969)
  321–342.
- K. S. Brentner, F. Farassat, *Analytical comparison of the acoustic
  analogy and Kirchhoff formulation for moving surfaces*, AIAA J. 36 (8)
  (1998) 1379–1386.
- F. Farassat, *Derivation of Formulations 1 and 1A of Farassat*, NASA
  TM-2007-214853, 2007.
- A. Di Francescantonio, *A new boundary integral formulation for the
  prediction of sound radiation*, J. Sound Vib. 202 (4) (1997) 491–509.
- D. Casalino, *An advanced time approach for acoustic analogy
  predictions*, J. Sound Vib. 261 (4) (2003) 583–612.
- J. W. Delfs, B. Ruck, *A quantity to identify turbulence related sound
  generation on surfaces*, J. Sound Vib. 586 (2024) 118490.
