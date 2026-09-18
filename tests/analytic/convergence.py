#!/usr/bin/env python3
"""Spatial and temporal convergence study for fwhFoam.

Runs the analytic monopole case at a sequence of icosphere subdivision
levels (spatial refinement) and samples-per-period values (temporal
refinement), recording the relative L2 error of the far-field pressure
against the exact solution. Writes a CSV and prints observed orders of
convergence.

Usage:
  python convergence.py --fwhsolve /path/to/fwhSolve --workdir DIR
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 "..", "..", "tools")))
from pyfwh import io, analytic, geometry, spectra  # noqa: E402
import run_analytic as ra  # noqa: E402


def one_run(fwhsolve, workdir, field, c0, rho0, U0, f0,
            sub, ppw, periods, radius, tag):
    surf = geometry.icosphere(radius=radius, subdivisions=sub)
    nF = surf[0].shape[0]
    T = 1.0 / f0
    dt = T / ppw
    nT = int(periods * ppw) + 1
    times = np.arange(nT) * dt

    data = ra.build_case(field, surf, times, U0)
    datafile = os.path.join(workdir, f"{tag}.fwh")
    data.write(datafile)

    lam = c0 / f0
    observers = {"mic_y": (0.0, 10 * lam, 0.0)}
    outdir = os.path.join(workdir, f"{tag}-out")
    dictpath = os.path.join(workdir, f"fwhSolveDict.{tag}")
    ra.write_dict(dictpath, datafile, outdir, c0, rho0, U0, observers)
    ra.run_solve(fwhsolve, workdir, dictpath)

    sig = io.read_observer(os.path.join(outdir, "observer_mic_y.dat"))
    tw = sig["meta"].get("validWindow", (times[0], times[-1]))
    t = sig["t"]
    mask = (t >= tw[0]) & (t <= tw[1])
    tpred = t[mask]
    ppred = sig["p"][mask]
    pexact = ra.analytic_observer(field, np.array([0.0, 10 * lam, 0.0]), tpred)
    return nF, dt, ra.rel_l2(ppred, pexact)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwhsolve", required=True)
    ap.add_argument("--workdir", default="_conv_work")
    ap.add_argument("--freq", type=float, default=340.29)
    ap.add_argument("--radius", type=float, default=1.0)
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    c0, rho0 = 340.29, 1.225
    omega = 2 * np.pi * args.freq
    field = analytic.MonopoleField(1e-3, omega, c0=c0, rho0=rho0)

    print("# Spatial convergence (ppw=128 fixed):")
    print("# sub  nFaces      h~1/sqrt(F)   relL2")
    spatial = []
    for sub in [1, 2, 3, 4]:
        nF, dt, err = one_run(args.fwhsolve, args.workdir, field, c0, rho0,
                              np.zeros(3), args.freq, sub, 128, 8,
                              args.radius, f"sp{sub}")
        h = 1.0 / np.sqrt(nF)
        spatial.append((sub, nF, h, err))
        print(f"  {sub}   {nF:6d}    {h:.4e}   {err:.4e}")

    print("\n# Temporal convergence (sub=4 fixed):")
    print("# ppw   dt            relL2")
    temporal = []
    for ppw in [16, 32, 64, 128]:
        nF, dt, err = one_run(args.fwhsolve, args.workdir, field, c0, rho0,
                              np.zeros(3), args.freq, 4, ppw, 8,
                              args.radius, f"tp{ppw}")
        temporal.append((ppw, dt, err))
        print(f"  {ppw:3d}   {dt:.4e}   {err:.4e}")

    # Observed orders (log-log slope of successive pairs)
    def order(xs, es):
        xs = np.asarray(xs); es = np.asarray(es)
        return np.polyfit(np.log(xs), np.log(es), 1)[0]

    print("\n# Observed orders of convergence:")
    hs = [s[2] for s in spatial]
    es = [s[3] for s in spatial]
    print(f"  spatial (error ~ h^p):  p = {order(hs, es):.2f}")
    dts = [t[1] for t in temporal]
    et = [t[2] for t in temporal]
    print(f"  temporal (error ~ dt^q): q = {order(dts, et):.2f}")

    with open(os.path.join(args.workdir, "convergence.csv"), "w") as f:
        f.write("kind,param,nFaces_or_dt,relL2\n")
        for sub, nF, h, err in spatial:
            f.write(f"spatial,{sub},{nF},{err}\n")
        for ppw, dt, err in temporal:
            f.write(f"temporal,{ppw},{dt},{err}\n")
    print(f"\nWrote {os.path.join(args.workdir, 'convergence.csv')}")


if __name__ == "__main__":
    main()
