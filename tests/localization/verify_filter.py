#!/usr/bin/env python3
r"""Verify the diffraction filter against the analytic sphere eigenvalues.

The diffraction filter (double-layer operator) is diagonal in the spherical
harmonics on a sphere, with eigenvalue g_l(kR). Feeding a real spherical
harmonic Y_l^0 as the surface pressure and forming the Rayleigh quotient
(p . p_S)/(p . p) must recover g_l to discretization accuracy, and the
incompressible limit must approach -1/(2l+1).

Usage:
  python verify_filter.py [--sub 4] [--tol 3e-3]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 "..", "..", "tools")))
from pyfwh import geometry, filter as flt  # noqa: E402


def ylm0(centroids, ell):
    from scipy.special import sph_harm_y
    x, y, z = centroids.T
    r = np.linalg.norm(centroids, axis=1)
    theta = np.arccos(np.clip(z / r, -1, 1))
    phi = np.arctan2(y, x)
    return np.real(sph_harm_y(ell, 0, theta, phi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=4)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=3e-3)
    args = ap.parse_args()

    c, n, a, _, _ = geometry.icosphere(radius=args.radius, subdivisions=args.sub)
    print(f"diffraction-filter verification: {len(c)} faces, "
          f"area sum {a.sum():.5f} (4pi = {4*np.pi:.5f})")

    worst = 0.0
    print("  filter gain g_l (mesh vs analytic):")
    for kR in (0.5, 2.0):
        k = kR / args.radius
        for ell in (1, 2, 4):
            p = ylm0(c, ell)
            pS = flt.acoustic_surface_pressure(p, c, n, a, k)
            gain = (p @ pS) / (p @ p)
            ga = flt.sphere_eigenvalue(ell, kR)
            e = abs(gain - ga); worst = max(worst, e)
            print(f"    kR={kR:.1f} l={ell}: mesh {gain:+.4f}  "
                  f"analytic {ga:+.4f}  |d|={e:.2e}")

    print("  incompressible limit (kR=0.05) vs -1/(2l+1):")
    for ell in (1, 2, 3):
        p = ylm0(c, ell)
        pS = flt.acoustic_surface_pressure(p, c, n, a, 0.05 / args.radius)
        gain = (p @ pS) / (p @ p)
        exact = -1.0 / (2 * ell + 1)
        e = abs(gain.real - exact); worst = max(worst, e)
        print(f"    l={ell}: mesh {gain.real:+.4f}  exact {exact:+.4f}  "
              f"|d|={e:.2e}")

    ok = worst < args.tol
    print(f"\nworst error {worst:.2e}  ->  "
          + ("FILTER VERIFICATION PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
