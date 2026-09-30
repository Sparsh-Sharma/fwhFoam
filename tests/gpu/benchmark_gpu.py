#!/usr/bin/env python3
r"""Verify and benchmark the GPU localization kernels (pyfwh.gpu).

Two stages:

1. Correctness. On a small icosphere with analytic monopole+dipole Cauchy
   data, the backend-agnostic kernels in ``pyfwh.gpu`` must reproduce the
   reference implementations (``pyfwh.localization`` / ``pyfwh.filter``) to
   machine precision, on the CPU backend and, when present, on the GPU.
   The sigma power identity (integral sigma dS == P_farfield) is re-checked
   on the GPU result.

2. Benchmark. Times one frequency line of the sigma density and of the
   diffraction filter at increasing face counts, CPU (NumPy, same reduced
   algebra) vs GPU (CuPy), and reports the speedup. Results are appended to
   ``benchmark_gpu.json`` next to this script.

Usage:
  python benchmark_gpu.py [--subs 3 4 5 6] [--freq 500] [--skip-bench]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 "..", "..", "tools")))
from pyfwh import analytic, filter as dfilter, geometry, gpu, localization  # noqa: E402


def relerr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


def cauchy(field, cen, nrm):
    phat = np.asarray(field.pressure(cen, 0.0), complex)
    u = field.velocity(cen, 0.0)
    vnhat = np.asarray(np.einsum("ni,ni->n", u, nrm), complex)
    return phat, vnhat


def verify(freq, tol=1e-9):
    c0, rho0 = 340.29, 1.225
    omega = 2 * np.pi * freq
    k = omega / c0
    cen, nrm, area, _, _ = geometry.icosphere(radius=1.0, subdivisions=3)
    mono = analytic.MonopoleField(1e-3, omega, c0=c0, rho0=rho0)
    dip = analytic.DipoleField(1e-3, omega, axis=(0, 1, 0), c0=c0, rho0=rho0)
    phat = sum(cauchy(f, cen, nrm)[0] for f in (mono, dip))
    vnhat = sum(cauchy(f, cen, nrm)[1] for f in (mono, dip))

    sig_ref, P_ref = localization.sigma_density(cen, nrm, area, phat, vnhat,
                                                omega, rho0, c0)
    pS_ref = dfilter.acoustic_surface_pressure(phat, cen, nrm, area, k)

    backends = [("cpu", np)]
    if gpu.available():
        backends.append(("gpu", gpu.get_xp(True)))
    else:
        print("  (no GPU present: verifying the CPU backend only)")

    ok = True
    for name, xp in backends:
        sig, P = gpu.sigma_density(cen, nrm, area, phat, vnhat, omega,
                                   rho0, c0, xp=xp)
        pS = gpu.acoustic_surface_pressure(phat, cen, nrm, area, k, xp=xp)
        es, ef = relerr(sig, sig_ref), relerr(pS, pS_ref)
        eP = abs(P - P_ref) / abs(P_ref)
        P_far, _, _ = localization.farfield_power(cen, nrm, area, phat,
                                                  vnhat, omega, rho0, c0)
        eI = abs(P - P_far) / abs(P_far)
        good = es < tol and ef < tol and eP < tol and eI < 0.02
        ok &= good
        print(f"  [{name}] sigma err={es:.2e}  filter err={ef:.2e}  "
              f"P err={eP:.2e}  identity err={eI:.2e}  "
              f"[{'OK' if good else 'FAIL'}]")
    return ok


def bench_one(fun, repeats, xp):
    fun()                                   # warm-up / JIT / alloc
    gpu.synchronize(xp)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fun()
    gpu.synchronize(xp)
    return (time.perf_counter() - t0) / repeats


def benchmark(subs, freq):
    c0, rho0 = 340.29, 1.225
    omega = 2 * np.pi * freq
    k = omega / c0
    have_gpu = gpu.available()
    if have_gpu:
        dev = gpu.get_xp(True).cuda.runtime.getDeviceProperties(0)
        gname = dev["name"].decode()
        print(f"\nGPU: {gname}")
    else:
        gname = None
        print("\nNo GPU: CPU timings only")

    rows = []
    for sub in subs:
        cen, nrm, area, _, _ = geometry.icosphere(radius=1.0,
                                                  subdivisions=sub)
        N = len(cen)
        omega_ = omega
        rng = np.random.default_rng(0)
        phat = (rng.standard_normal(N) + 1j * rng.standard_normal(N))
        vnhat = (rng.standard_normal(N) + 1j * rng.standard_normal(N)) / (rho0 * c0)

        reps = 3 if N <= 6000 else 1
        row = {"N": N, "freq": freq}

        t = bench_one(lambda: gpu.sigma_density(
            cen, nrm, area, phat, vnhat, omega_, rho0, c0, xp=np), reps, np)
        row["sigma_cpu_s"] = t
        t = bench_one(lambda: gpu.acoustic_surface_pressure(
            phat, cen, nrm, area, k, xp=np), reps, np)
        row["filter_cpu_s"] = t

        if have_gpu:
            xp = gpu.get_xp(True)
            t = bench_one(lambda: gpu.sigma_density(
                cen, nrm, area, phat, vnhat, omega_, rho0, c0, xp=xp),
                max(reps, 5), xp)
            row["sigma_gpu_s"] = t
            t = bench_one(lambda: gpu.acoustic_surface_pressure(
                phat, cen, nrm, area, k, xp=xp), max(reps, 5), xp)
            row["filter_gpu_s"] = t
            row["sigma_speedup"] = row["sigma_cpu_s"] / row["sigma_gpu_s"]
            row["filter_speedup"] = row["filter_cpu_s"] / row["filter_gpu_s"]
            print(f"  N={N:6d}  sigma {row['sigma_cpu_s']:8.3f}s -> "
                  f"{row['sigma_gpu_s']:7.4f}s ({row['sigma_speedup']:6.1f}x)"
                  f"   filter {row['filter_cpu_s']:8.3f}s -> "
                  f"{row['filter_gpu_s']:7.4f}s ({row['filter_speedup']:6.1f}x)")
        else:
            print(f"  N={N:6d}  sigma {row['sigma_cpu_s']:8.3f}s"
                  f"   filter {row['filter_cpu_s']:8.3f}s")
        rows.append(row)

    # --- FW-H solver benchmark: faces x observers x time ----------------
    from pyfwh import io as fio
    fwh_rows = []
    nobs, nT = 64, 512
    ang = np.linspace(0, 2 * np.pi, nobs, endpoint=False)
    observers = {f"m{i:03d}": (50 * np.cos(a), 50 * np.sin(a), 0.0)
                 for i, a in enumerate(ang)}
    for sub in subs:
        cen, nrm, area, _, _ = geometry.icosphere(radius=1.0,
                                                  subdivisions=sub)
        N = len(cen)
        rng = np.random.default_rng(1)
        times = np.arange(nT) / (nT * 1.0) * 0.1
        p = 1e-3 * rng.standard_normal((nT, N))
        u = 1e-4 * rng.standard_normal((nT, N, 3))
        data = fio.FWHData(cen, nrm, area, times, p, u, None)

        row = {"N": N, "nObs": nobs, "nT": nT}
        reps = 1
        row["fwh_cpu_s"] = bench_one(lambda: gpu.farassat_1a(
            data, observers, 340.29, 1.225, xp=np), reps, np)
        if have_gpu:
            xp = gpu.get_xp(True)
            row["fwh_gpu_s"] = bench_one(lambda: gpu.farassat_1a(
                data, observers, 340.29, 1.225, xp=xp), 3, xp)
            row["fwh_speedup"] = row["fwh_cpu_s"] / row["fwh_gpu_s"]
            print(f"  N={N:6d} x {nobs} obs x {nT} steps  FW-H "
                  f"{row['fwh_cpu_s']:8.3f}s -> {row['fwh_gpu_s']:7.4f}s "
                  f"({row['fwh_speedup']:6.1f}x)")
        else:
            print(f"  N={N:6d} x {nobs} obs x {nT} steps  FW-H "
                  f"{row['fwh_cpu_s']:8.3f}s")
        fwh_rows.append(row)

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "benchmark_gpu.json")
    with open(out, "w") as f:
        json.dump({"gpu": gname, "rows": rows, "fwh_rows": fwh_rows},
                  f, indent=2)
    print(f"\nwritten: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", type=int, nargs="+", default=[3, 4, 5, 6])
    ap.add_argument("--freq", type=float, default=500.0)
    ap.add_argument("--skip-bench", action="store_true")
    args = ap.parse_args()

    print(f"GPU-kernel verification (f={args.freq} Hz):")
    ok = verify(args.freq)
    print("\n" + ("GPU KERNEL VERIFICATION PASSED" if ok else "GPU FAILED"))
    if not ok:
        sys.exit(1)
    if not args.skip_bench:
        benchmark(args.subs, args.freq)
    sys.exit(0)


if __name__ == "__main__":
    main()
