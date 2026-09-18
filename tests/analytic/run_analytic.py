#!/usr/bin/env python3
"""End-to-end analytic verification of fwhFoam.

For each test case:
  1. build a closed icosphere FW-H surface enclosing an analytic source,
  2. sample the exact (p', u, rho) of that source on the surface over a
     number of periods and write an FWH-DATA file,
  3. run the compiled `fwhSolve` utility,
  4. compare the predicted observer pressure to the exact analytic
     far-field pressure at the same observers,
  5. report the relative L2 error and amplitude/phase error.

Cases:
  monopole  -- thickness (U_n) dominated, medium at rest
  dipole    -- loading (L_r) dominated, medium at rest, cos-directivity
  convected -- monopole in uniform mean flow (Garrick / convective term)

Usage:
  python run_analytic.py --fwhsolve /path/to/fwhSolve [--case all]
                         [--sub 3] [--ppw 64] [--periods 6] [--radius 1]
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 "..", "..", "tools")))
from pyfwh import io, analytic, geometry, spectra  # noqa: E402


def build_case(field, surf, times, U0):
    """Sample an analytic field on the surface for all times -> FWHData."""
    centres, normals, areas, _, _ = surf
    nF = centres.shape[0]
    nT = times.size
    p = np.zeros((nT, nF))
    u = np.zeros((nT, nF, 3))
    rho = np.zeros((nT, nF))
    Umean = getattr(field, "U_mean", np.zeros(3))
    for k, t in enumerate(times):
        p[k] = np.real(field.pressure(centres, t))
        # Total CFD fluid velocity = mean flow + acoustic perturbation
        u[k] = Umean + np.real(field.velocity(centres, t))
        rho[k] = field.rho0 + np.real(field.density(centres, t))
    return io.FWHData(centres, normals, areas, times, p, u, rho)


def analytic_observer(field, xobs, times):
    return np.array([np.real(field.pressure(np.atleast_2d(xobs), t))[0]
                     for t in times])


def rel_l2(a, b):
    return np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)


def run_solve(fwhsolve, workdir, dictname="fwhSolveDict"):
    env = os.environ.copy()
    r = subprocess.run([fwhsolve, "-dict", dictname],
                       cwd=workdir, capture_output=True, text=True, env=env)
    if r.returncode != 0:
        print(r.stdout)
        print(r.stderr, file=sys.stderr)
        raise RuntimeError("fwhSolve failed")
    return r.stdout


def write_dict(path, datafile, outdir, c0, rho0, U0, observers):
    lines = [
        "/*--------------------------------*- C++ -*----------------------*\\",
        "| fwhSolve configuration (analytic verification)                  |",
        "\\*---------------------------------------------------------------*/",
        f'dataFile    "{datafile}";',
        f'outputDir   "{outdir}";',
        f"c0          {c0};",
        f"rho0        {rho0};",
        f"U0          ({U0[0]} {U0[1]} {U0[2]});",
        "observers",
        "{",
    ]
    for name, pos in observers.items():
        lines.append(f"    {name} {{ position ({pos[0]} {pos[1]} {pos[2]}); }}")
    lines += ["}", ""]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def case_monopole(args, workdir):
    c0, rho0 = 340.29, 1.225
    f0 = args.freq
    omega = 2 * np.pi * f0
    field = analytic.MonopoleField(amplitude=1.0e-3, omega=omega,
                                   c0=c0, rho0=rho0)
    return _generic(args, workdir, field, c0, rho0, np.zeros(3), f0,
                    "monopole")


def case_dipole(args, workdir):
    c0, rho0 = 340.29, 1.225
    f0 = args.freq
    omega = 2 * np.pi * f0
    field = analytic.DipoleField(amplitude=1.0e-3, omega=omega,
                                 axis=(0, 1, 0), c0=c0, rho0=rho0)
    return _generic(args, workdir, field, c0, rho0, np.zeros(3), f0,
                    "dipole")


def case_convected(args, workdir):
    c0, rho0 = 340.29, 1.225
    f0 = args.freq
    omega = 2 * np.pi * f0
    U0 = np.array([0.3 * c0, 0.0, 0.0])
    field = analytic.ConvectedMonopoleField(amplitude=1.0e-3, omega=omega,
                                            U0=U0, c0=c0, rho0=rho0)
    return _generic(args, workdir, field, c0, rho0, U0, f0, "convected")


def _generic(args, workdir, field, c0, rho0, U0, f0, tag):
    surf = geometry.icosphere(radius=args.radius, subdivisions=args.sub)
    nF = surf[0].shape[0]

    T = 1.0 / f0
    dt = T / args.ppw
    nT = int(args.periods * args.ppw) + 1
    times = np.arange(nT) * dt

    data = build_case(field, surf, times, U0)
    datafile = os.path.join(workdir, f"{tag}.fwh")
    data.write(datafile)

    lam = c0 / f0
    observers = {
        "mic_x": (10 * lam, 0.0, 0.0),
        "mic_y": (0.0, 10 * lam, 0.0),
        "mic_45": (10 * lam / np.sqrt(2), 10 * lam / np.sqrt(2), 0.0),
    }
    outdir = os.path.join(workdir, f"{tag}-out")
    dictpath = os.path.join(workdir, f"fwhSolveDict.{tag}")
    write_dict(dictpath, datafile, outdir, c0, rho0, U0, observers)

    run_solve(args.fwhsolve, workdir, dictpath)

    print(f"\n=== case: {tag}  (faces={nF}, dt={dt:.3e}, "
          f"records={nT}) ===")

    # First pass: gather exact amplitudes to identify silent directions
    exact_amp = {}
    per_obs = {}
    for name, pos in observers.items():
        sig = io.read_observer(os.path.join(outdir, f"observer_{name}.dat"))
        tw = sig["meta"].get("validWindow", (times[0], times[-1]))
        t = sig["t"]
        mask = (t >= tw[0]) & (t <= tw[1])
        if mask.sum() < 8:
            per_obs[name] = None
            continue
        tpred = t[mask]
        ppred = sig["p"][mask]
        pexact = analytic_observer(field, np.asarray(pos), tpred)
        a_exact = spectra.amplitude_at(pexact, tpred[1] - tpred[0], f0)
        exact_amp[name] = a_exact
        per_obs[name] = (tpred, ppred, pexact, a_exact)

    amax = max(exact_amp.values()) if exact_amp else 0.0
    ok = True
    results = {}
    for name in observers:
        rec = per_obs[name]
        if rec is None:
            print(f"  {name:7s} valid window too small, skipping")
            continue
        tpred, ppred, pexact, a_exact = rec
        # Skip observers in a radiation null (exact amplitude negligible):
        # relative error is meaningless there.
        if amax > 0 and a_exact < 1e-3 * amax:
            print(f"  {name:7s} radiation null (|p|={a_exact:.2e} Pa), skipped")
            continue
        err = rel_l2(ppred, pexact)
        a_pred = spectra.amplitude_at(ppred, tpred[1] - tpred[0], f0)
        amp_err = abs(a_pred - a_exact) / max(abs(a_exact), 1e-300)
        results[name] = (err, amp_err)
        flag = "OK" if err < args.tol else "FAIL"
        if err >= args.tol:
            ok = False
        print(f"  {name:7s} relL2={err:8.4f}  ampErr={amp_err:8.4f}  "
              f"|p|={a_exact:.3e} Pa  [{flag}]")
    return ok, results


CASES = {
    "monopole": case_monopole,
    "dipole": case_dipole,
    "convected": case_convected,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwhsolve", required=True, help="path to fwhSolve")
    ap.add_argument("--case", default="all", choices=list(CASES) + ["all"])
    ap.add_argument("--workdir", default="_analytic_work")
    ap.add_argument("--sub", type=int, default=3, help="icosphere subdivisions")
    ap.add_argument("--ppw", type=int, default=64, help="samples per period")
    ap.add_argument("--periods", type=int, default=6)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--freq", type=float, default=340.29,
                    help="source frequency [Hz] (default: 1/lambda=1m)")
    ap.add_argument("--tol", type=float, default=0.05)
    args = ap.parse_args()

    os.makedirs(args.workdir, exist_ok=True)
    cases = list(CASES) if args.case == "all" else [args.case]

    all_ok = True
    for c in cases:
        ok, _ = CASES[c](args, args.workdir)
        all_ok = all_ok and ok

    print("\n" + ("ALL CASES PASSED" if all_ok else "SOME CASES FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
