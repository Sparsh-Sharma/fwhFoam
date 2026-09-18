#!/usr/bin/env python3
r"""Verify the sigma surface sound-power localization against analytic fields.

The sigma method must satisfy, at every frequency, the power identity

    closed_int sigma dS  ==  P_farfield  ==  P_pointwise  ==  P_analytic

for exact acoustic Cauchy data placed on a closed surface enclosing the
source. We check this on a geodesic icosphere enclosing:

  * a monopole  -> isotropic radiation (thickness-like source),
  * a dipole    -> cos^2 directivity (loading-like source).

For a time-harmonic field of angular frequency omega the exact radiated
power is obtained independently from the surface acoustic intensity
P_analytic = closed_int (1/2) Re(p_hat conj(vn_hat)) dS using the analytic
(p_hat, vn_hat); the sigma two-point density and the far-field amplitude
must both reproduce it.

Usage:
  python verify_sigma.py [--sub 3] [--tol 0.02]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 "..", "..", "tools")))
from pyfwh import analytic, geometry, localization  # noqa: E402


def cauchy_on_surface(field, cen, nrm, omega):
    """Exact complex (p_hat, vn_hat) of an analytic field on the surface."""
    # analytic fields carry the exp(i omega t) convention; take the complex
    # phasor at t=0 (pressure()/velocity() already return complex phasors)
    phat = field.pressure(cen, 0.0)
    u = field.velocity(cen, 0.0)
    vnhat = np.einsum("ni,ni->n", u, nrm)
    return np.asarray(phat, complex), np.asarray(vnhat, complex)


def run_case(field, omega, rho, c, sub, radius, tol, tag):
    cen, nrm, area, _, _ = geometry.icosphere(radius=radius, subdivisions=sub)
    phat, vnhat = cauchy_on_surface(field, cen, nrm, omega)

    # analytic reference power (surface acoustic intensity of the exact field)
    P_true = localization.pointwise_power(nrm, area, phat, vnhat)

    sigma, P_sigma = localization.sigma_density(cen, nrm, area, phat, vnhat,
                                                omega, rho, c)
    P_far, _, _ = localization.farfield_power(cen, nrm, area, phat, vnhat,
                                              omega, rho, c)

    e_sig = abs(P_sigma - P_true) / abs(P_true)
    e_far = abs(P_far - P_true) / abs(P_true)
    ok = (e_sig < tol) and (e_far < tol)
    print(f"  {tag:9s} faces={len(cen):5d}  P_true={P_true:.4e} W  "
          f"|sigmaInt err={e_sig:.2e}  farfield err={e_far:.2e}  "
          f"sigma range=[{sigma.min():.2e},{sigma.max():.2e}]  "
          f"[{'OK' if ok else 'FAIL'}]")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=3)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--freq", type=float, default=200.0)
    args = ap.parse_args()

    c0, rho0 = 340.29, 1.225
    omega = 2 * np.pi * args.freq

    print(f"sigma-localization verification (f={args.freq} Hz, kR="
          f"{omega/c0*args.radius:.3f}, sub={args.sub}):")
    ok = True
    ok &= run_case(analytic.MonopoleField(1e-3, omega, c0=c0, rho0=rho0),
                   omega, rho0, c0, args.sub, args.radius, args.tol, "monopole")
    ok &= run_case(analytic.DipoleField(1e-3, omega, axis=(0, 1, 0),
                                        c0=c0, rho0=rho0),
                   omega, rho0, c0, args.sub, args.radius, args.tol, "dipole")

    print("\n" + ("SIGMA VERIFICATION PASSED" if ok else "SIGMA FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
