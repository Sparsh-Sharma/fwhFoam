#!/usr/bin/env python3
r"""Compute a surface sound-power localization (sigma) map from fwhFoam data.

Reads the permeable-surface data written by the fwh function object
(surfaceData_proc*.fwh), transforms the surface time histories to
frequency-domain Cauchy data (p_hat, vn_hat), and computes the surface
sound-power density sigma over a chosen frequency band. Writes:

  * <out>.vtk  -- a point cloud of face centres carrying sigma (and the
                  local pressure RMS) for visualization in ParaView;
  * <out>.csv  -- x,y,z,nx,ny,nz,area,sigma per face;
  * a console report of the power identity  int sigma dS  vs  P_farfield
    per spectral line and band-integrated.

Usage:
  python sigma_localize.py --data "case/postProcessing/fwh1/acousticData/surfaceData_proc*.fwh" \
      --fmin 100 --fmax 5000 --c0 340.29 --rho0 1.225 --out sigma_map
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))
from pyfwh import io, localization  # noqa: E402


def load_surface(pattern):
    """Read and concatenate one or more surfaceData_proc*.fwh files.

    All files must share the same time base (they are pieces of one
    integration surface written by different MPI ranks).
    """
    files = sorted(glob.glob(pattern))
    if not files:
        sys.exit(f"no files match {pattern!r}")
    parts = [io.read(f) for f in files]
    t0 = parts[0].times
    for d in parts[1:]:
        if d.n_times != parts[0].n_times or not np.allclose(d.times, t0):
            sys.exit("surface-data files have inconsistent time bases")
    cen = np.concatenate([d.centres for d in parts], axis=0)
    nrm = np.concatenate([d.normals for d in parts], axis=0)
    area = np.concatenate([d.areas for d in parts], axis=0)
    p = np.concatenate([d.p for d in parts], axis=1)
    u = np.concatenate([d.u for d in parts], axis=1)
    rho = (np.concatenate([d.rho for d in parts], axis=1)
           if parts[0].rho is not None else None)
    merged = io.FWHData(cen, nrm, area, t0, p, u, rho)
    print(f"loaded {len(files)} file(s), {cen.shape[0]} faces, "
          f"{t0.size} time levels, dt={np.mean(np.diff(t0)):.3e}")
    return merged


def write_vtk(path, cen, sigma, prms):
    """Legacy-VTK POLYDATA point cloud with sigma and pressure-RMS scalars."""
    n = cen.shape[0]
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\nfwhFoam sigma localization\n")
        f.write("ASCII\nDATASET POLYDATA\n")
        f.write(f"POINTS {n} float\n")
        for c in cen:
            f.write(f"{c[0]:.6e} {c[1]:.6e} {c[2]:.6e}\n")
        f.write(f"VERTICES {n} {2*n}\n")
        for i in range(n):
            f.write(f"1 {i}\n")
        f.write(f"POINT_DATA {n}\n")
        f.write("SCALARS sigma float 1\nLOOKUP_TABLE default\n")
        for s in sigma:
            f.write(f"{s:.6e}\n")
        f.write("SCALARS p_rms float 1\nLOOKUP_TABLE default\n")
        for s in prms:
            f.write(f"{s:.6e}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="glob for surfaceData_proc*.fwh")
    ap.add_argument("--c0", type=float, default=340.29)
    ap.add_argument("--rho0", type=float, default=1.225)
    ap.add_argument("--fmin", type=float, default=0.0)
    ap.add_argument("--fmax", type=float, default=np.inf)
    ap.add_argument("--out", default="sigma_map")
    ap.add_argument("--chunk", type=int, default=None,
                    help="row block size to bound memory for large surfaces")
    ap.add_argument("--window", default="hann")
    args = ap.parse_args()

    data = load_surface(args.data)
    freqs, phat, vnhat, _ = localization.cauchy_spectrum(data, window=args.window)

    fmax = args.fmax if np.isfinite(args.fmax) else freqs.max()
    sig, P_band, lines = localization.sigma_band(
        data.centres, data.normals, data.areas, freqs, phat, vnhat,
        args.rho0, args.c0, args.fmin, fmax, chunk=args.chunk)

    print(f"\nBand [{args.fmin:.3g}, {fmax:.3g}] Hz  ({len(lines)} lines)")
    print(f"{'freq[Hz]':>10} {'P_sigma[W]':>13} {'P_far[W]':>13} "
          f"{'P_point[W]':>13} {'|1-Pf/Ps|':>10}")
    # report the loudest few lines
    lines_sorted = sorted(lines, key=lambda L: -abs(L[1]))
    for fr, Ps, Pf, Pp in lines_sorted[:8]:
        rel = abs(1 - Pf / Ps) if Ps else float("nan")
        print(f"{fr:10.3g} {Ps:13.4e} {Pf:13.4e} {Pp:13.4e} {rel:10.2e}")
    P_far_band = sum(L[2] for L in lines)
    print(f"\nBand-integrated:  int sigma dS = {P_band:.4e} W   "
          f"P_farfield = {P_far_band:.4e} W   "
          f"rel.diff = {abs(1 - P_far_band/P_band):.2e}")

    prms = np.sqrt(np.mean((data.p - data.p.mean(axis=0))**2, axis=0))
    write_vtk(args.out + ".vtk", data.centres, sig, prms)
    with open(args.out + ".csv", "w") as f:
        f.write("x,y,z,nx,ny,nz,area,sigma\n")
        for c, nn, a, s in zip(data.centres, data.normals, data.areas, sig):
            f.write(f"{c[0]:.6e},{c[1]:.6e},{c[2]:.6e},"
                    f"{nn[0]:.6e},{nn[1]:.6e},{nn[2]:.6e},{a:.6e},{s:.6e}\n")
    print(f"\nwrote {args.out}.vtk and {args.out}.csv")


if __name__ == "__main__":
    main()
