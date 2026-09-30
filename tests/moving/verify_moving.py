#!/usr/bin/env python3
r"""Verify the moving-surface formulation by surface-motion invariance.

The far field of a STATIC source enclosed by a permeable surface must not
depend on how that surface moves. A static analytic monopole is therefore
enclosed by

  * a RIGIDLY ROTATING icosphere (tip Mach 0.3): every motion term is
    exercised --- moving face positions, rotating normals (n-dot), surface
    velocity in U and L, and M-dot through the centripetal acceleration;
  * an OSCILLATING (translating) icosphere (velocity amplitude 0.2 c0):
    translation terms and periodic Doppler.

In both cases the observer signal must reproduce the analytic monopole far
field to discretization accuracy, as the static surface does.

Usage:
  python verify_moving.py --fwhsolve /path/to/fwhSolve [--sub 3] [--tol 0.05]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "analytic"))
from pyfwh import io, analytic, geometry  # noqa: E402
import run_analytic as ra  # noqa: E402


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def build_moving(field, surf, times, motion):
    """Sample the analytic field on a moving surface; return FWHData."""
    cen0, nrm0, area0, _, _ = surf
    nF, nT = cen0.shape[0], times.size
    p = np.zeros((nT, nF)); u = np.zeros((nT, nF, 3))
    rho = np.zeros((nT, nF))
    cf_t = np.zeros((nT, nF, 3)); n_t = np.zeros((nT, nF, 3))
    v_t = np.zeros((nT, nF, 3)); dA_t = np.zeros((nT, nF))
    for k, t in enumerate(times):
        if motion["type"] == "rotate":
            R = rotz(motion["omega"] * t)
            cen = cen0 @ R.T
            nrm = nrm0 @ R.T
            v = np.cross(np.array([0, 0, motion["omega"]]), cen)
        else:                                   # oscillate along z
            A, Om = motion["A"], motion["Om"]
            cen = cen0 + np.array([0, 0, A * np.sin(Om * t)])
            nrm = nrm0
            v = np.tile([0, 0, A * Om * np.cos(Om * t)], (nF, 1))
        cf_t[k] = cen; n_t[k] = nrm; v_t[k] = v; dA_t[k] = area0
        p[k] = np.real(field.pressure(cen, t))
        u[k] = np.real(field.velocity(cen, t))
        rho[k] = field.rho0 + np.real(field.density(cen, t))
    return io.FWHData(cen0, nrm0, area0, times, p, u, rho,
                      vsurf=v_t, cf_t=cf_t, n_t=n_t, dA_t=dA_t)


def run_case(args, workdir, motion, tag, saved=None):
    c0, rho0 = 340.29, 1.225
    f0 = 200.0
    field = analytic.MonopoleField(1e-3, 2 * np.pi * f0, c0=c0, rho0=rho0)
    surf = geometry.icosphere(radius=1.0, subdivisions=args.sub)

    dt = (1.0 / f0) / args.ppw
    times = np.arange(int(args.periods * args.ppw) + 1) * dt
    data = build_moving(field, surf, times, motion)
    datafile = os.path.join(workdir, f"{tag}.fwh")
    data.write(datafile)

    lam = c0 / f0
    observers = {"mic_x": (10 * lam, 0, 0), "mic_y": (0, 10 * lam, 0),
                 "mic_z": (0, 0, 10 * lam)}
    outdir = os.path.join(workdir, f"{tag}-out")
    dictpath = os.path.join(workdir, f"dict.{tag}")
    ra.write_dict(dictpath, datafile, outdir, c0, rho0, np.zeros(3), observers)
    ra.run_solve(args.fwhsolve, workdir, dictpath)

    print(f"=== {tag} (faces={data.n_faces}, records={times.size}) ===")
    ok = True
    for name, pos in observers.items():
        sig = io.read_observer(os.path.join(outdir, f"observer_{name}.dat"))
        tw = sig["meta"].get("validWindow", (times[0], times[-1]))
        m = (sig["t"] >= tw[0]) & (sig["t"] <= tw[1])
        if m.sum() < 8:
            print(f"  {name}: window too small"); ok = False; continue
        tpred, ppred = sig["t"][m], sig["p"][m]
        pex = ra.analytic_observer(field, np.asarray(pos), tpred)
        err = ra.rel_l2(ppred, pex)
        flag = "OK" if err < args.tol else "FAIL"
        ok &= err < args.tol
        print(f"  {name}: relL2={err:.4f} [{flag}]")
        if saved is not None and name == "mic_x":
            saved[f"{tag}_t"] = tpred
            saved[f"{tag}_pred"] = ppred
            saved[f"{tag}_exact"] = pex
            saved[f"{tag}_err"] = err
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwhsolve", required=True)
    ap.add_argument("--workdir", default="/tmp/fwhmoving")
    ap.add_argument("--sub", type=int, default=3)
    ap.add_argument("--ppw", type=int, default=64)
    ap.add_argument("--periods", type=int, default=6)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--save", default=None,
                    help="write mic_x signals of both cases to this .npz")
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    c0 = 340.29
    saved = {} if args.save else None
    ok = True
    # rotation: tip speed 0.3 c0 at R=1
    ok &= run_case(args, args.workdir,
                   dict(type="rotate", omega=0.3 * c0), "rotating", saved)
    # oscillation: amplitude 0.15, velocity amplitude 0.2 c0
    A = 0.15
    ok &= run_case(args, args.workdir,
                   dict(type="oscillate", A=A, Om=0.2 * c0 / A),
                   "oscillating", saved)

    if args.save:
        np.savez(args.save, **saved)
        print("saved", args.save)

    print("\nMOVING-SURFACE VERIFICATION " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
