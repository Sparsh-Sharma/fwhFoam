#!/usr/bin/env python3
r"""Verify the GPU Farassat-1A solver (pyfwh.gpu.farassat_1a).

Same analytic cases as ``tests/analytic/run_analytic.py`` --- a monopole, a
dipole and a convected monopole sampled exactly on an icosphere --- but
solved with the vectorized (NumPy/CuPy) port of the advanced-time
formulation-1A core instead of the compiled ``fwhSolve``. The predicted
observer pressure must match the analytic far-field signal, and the CPU and
GPU backends must agree to machine precision.

Usage:
  python verify_fwh_gpu.py [--sub 3] [--ppw 64] [--periods 6] [--tol 0.02]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 "..", "..", "tools")))
from pyfwh import analytic, geometry, gpu, io  # noqa: E402


def build_data(field, surf, times):
    centres, normals, areas, _, _ = surf
    nF, nT = centres.shape[0], times.size
    p = np.zeros((nT, nF))
    u = np.zeros((nT, nF, 3))
    rho = np.zeros((nT, nF))
    Umean = getattr(field, "U_mean", np.zeros(3))
    for k, t in enumerate(times):
        p[k] = np.real(field.pressure(centres, t))
        u[k] = Umean + np.real(field.velocity(centres, t))
        rho[k] = field.rho0 + np.real(field.density(centres, t))
    return io.FWHData(centres, normals, areas, times, p, u, rho)


def rel_l2(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


def run_case(tag, field, U0, args, observers):
    c0, rho0 = field.c0, field.rho0
    f0 = field.omega / (2 * np.pi)
    dt = 1.0 / (f0 * args.ppw)
    nT = int(args.periods * args.ppw) + 3
    times = dt * np.arange(nT)
    surf = geometry.icosphere(radius=args.radius, subdivisions=args.sub)
    data = build_data(field, surf, times)

    backends = [("cpu", np)]
    if gpu.available():
        backends.append(("gpu", gpu.get_xp(True)))

    ok = True
    res = {}
    for name, xp in backends:
        t0 = time.perf_counter()
        res[name] = gpu.farassat_1a(data, observers, c0, rho0, U0=U0, xp=xp)
        gpu.synchronize(xp)
        elapsed = time.perf_counter() - t0

        # normalize by the strongest observer so a directivity null
        # (analytic signal ~ 0) does not blow up the relative error
        exact = {}
        for oname, pos in observers.items():
            tg, _ = res[name][oname]
            exact[oname] = np.array([
                np.real(field.pressure(np.atleast_2d(pos), t))[0]
                for t in tg])
        ref = max(np.linalg.norm(pex) for pex in exact.values())
        e = max(
            float(np.linalg.norm(res[name][o][1] - exact[o]) / ref)
            for o in observers)
        good = e < args.tol
        ok &= good
        print(f"  {tag:10s} [{name}] relL2={e:.2e}  ({elapsed:6.2f}s)  "
              f"[{'OK' if good else 'FAIL'}]")

    if "gpu" in res:
        dmax = max(
            rel_l2(res["gpu"][k][1], res["cpu"][k][1]) for k in observers)
        good = dmax < 1e-12
        ok &= good
        print(f"  {tag:10s} cpu-vs-gpu maxrel={dmax:.2e}  "
              f"[{'OK' if good else 'FAIL'}]")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=3)
    ap.add_argument("--ppw", type=int, default=64)
    ap.add_argument("--periods", type=float, default=6)
    ap.add_argument("--radius", type=float, default=1.0)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--freq", type=float, default=200.0)
    args = ap.parse_args()

    c0, rho0 = 340.29, 1.225
    omega = 2 * np.pi * args.freq
    observers = {
        "mic090": (0.0, 20.0, 0.0),
        "mic045": (14.142, 14.142, 0.0),
        "mic000": (20.0, 0.0, 0.0),
    }

    print("GPU Farassat-1A verification "
          f"(f={args.freq} Hz, sub={args.sub}, gpu={gpu.available()}):")
    ok = True
    ok &= run_case("monopole",
                   analytic.MonopoleField(1e-3, omega, c0=c0, rho0=rho0),
                   np.zeros(3), args, observers)
    ok &= run_case("dipole",
                   analytic.DipoleField(1e-3, omega, axis=(0, 1, 0),
                                        c0=c0, rho0=rho0),
                   np.zeros(3), args, observers)
    U0 = np.array([0.3 * c0, 0.0, 0.0])
    ok &= run_case("convected",
                   analytic.ConvectedMonopoleField(1e-3, omega, U0=U0,
                                                   c0=c0, rho0=rho0),
                   U0, args, observers)

    print("\n" + ("GPU FW-H VERIFICATION PASSED" if ok else "GPU FW-H FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
