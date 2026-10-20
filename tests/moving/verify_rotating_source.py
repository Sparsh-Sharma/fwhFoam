#!/usr/bin/env python3
r"""Moving-SOURCE validation: a rotating monopole (rotor-like).

The surface-motion invariance test (verify_moving.py) encloses a *static*
source in a moving surface. Reviewers asked for a genuinely moving source.
Here a pulsating monopole orbits the z axis at tip Mach 0.3 (a rotating
source), and its exact far field (analytic, retarded-time) is the reference.

Two FW-H computations must both reproduce that reference:

  * CO-ROTATING surface: a sphere that rotates rigidly with the source, so the
    moving-surface path is exercised with a moving source (moving faces,
    rotating normals, surface velocity, n-dot, M-dot);
  * STATIONARY surface: a fixed sphere enclosing the whole orbit, an
    independent cross-check through the static-surface path.

Agreement of both with the analytic rotating-source field, to discretisation
accuracy, validates the moving-surface capability on a moving source.

Usage:
  python verify_rotating_source.py --fwhsolve /path/to/fwhSolve
      [--sub 3] [--ppw 48] [--revs 1.5] [--tol 0.05] [--save out.npz]
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
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def sample(field, surf, times, rotating, Omega):
    """Sample the moving-source field on a (possibly rotating) sphere."""
    cen0, nrm0, area0, _, _ = surf
    nF, nT = cen0.shape[0], times.size
    p = np.zeros((nT, nF)); u = np.zeros((nT, nF, 3)); rho = np.zeros((nT, nF))
    cf = np.zeros((nT, nF, 3)); nt = np.zeros((nT, nF, 3))
    vt = np.zeros((nT, nF, 3)); dA = np.zeros((nT, nF))
    wz = np.array([0.0, 0.0, Omega])
    for k, t in enumerate(times):
        if rotating:
            R = rotz(Omega * t)
            cen = cen0 @ R.T
            nrm = nrm0 @ R.T
            v = np.cross(wz, cen)
        else:
            cen, nrm, v = cen0, nrm0, np.zeros_like(cen0)
        cf[k], nt[k], vt[k], dA[k] = cen, nrm, v, area0
        p[k] = np.real(field.pressure(cen, t))
        u[k] = np.real(field.velocity(cen, t))
        rho[k] = field.rho0 + np.real(field.density(cen, t))
    return io.FWHData(cen0, nrm0, area0, times, p, u, rho,
                      vsurf=vt, cf_t=cf, n_t=nt, dA_t=dA)


def run_case(args, workdir, field, Omega, rotating, tag, observers, saved):
    c0, rho0 = field.c0, field.rho0
    surf = geometry.icosphere(radius=args.a, subdivisions=args.sub)
    data = sample(field, surf, args.times, rotating, Omega)
    datafile = os.path.join(workdir, f"{tag}.fwh"); data.write(datafile)
    outdir = os.path.join(workdir, f"{tag}-out")
    dictpath = os.path.join(workdir, f"dict.{tag}")
    ra.write_dict(dictpath, datafile, outdir, c0, rho0, np.zeros(3), observers)
    ra.run_solve(args.fwhsolve, workdir, dictpath)

    print(f"=== {tag} (faces={data.n_faces}, records={args.times.size}, "
          f"{'rotating' if rotating else 'stationary'} surface) ===")
    ok = True
    for name, pos in observers.items():
        sig = io.read_observer(os.path.join(outdir, f"observer_{name}.dat"))
        tw = sig["meta"].get("validWindow", (args.times[0], args.times[-1]))
        m = (sig["t"] >= tw[0]) & (sig["t"] <= tw[1])
        if m.sum() < 8:
            print(f"  {name}: window too small"); ok = False; continue
        tp, pp = sig["t"][m], sig["p"][m]
        pex = ra.analytic_observer(field, np.asarray(pos), tp)
        err = ra.rel_l2(pp, pex)
        ok &= err < args.tol
        print(f"  {name}: relL2={err:.4f} [{'OK' if err < args.tol else 'FAIL'}]")
        if saved is not None and name == "mic_z":
            saved[f"{tag}_t"] = tp; saved[f"{tag}_pred"] = pp
            saved[f"{tag}_exact"] = pex; saved[f"{tag}_err"] = err
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fwhsolve", required=True)
    ap.add_argument("--workdir", default="/tmp/fwhrot")
    ap.add_argument("--sub", type=int, default=3)
    ap.add_argument("--ppw", type=int, default=48)
    ap.add_argument("--revs", type=float, default=1.5)
    ap.add_argument("--a", type=float, default=1.0, help="sphere radius [m]")
    ap.add_argument("--R0", type=float, default=0.5, help="orbit radius [m]")
    ap.add_argument("--mach", type=float, default=0.3, help="tip Mach")
    ap.add_argument("--freq", type=float, default=200.0)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--save", default=None)
    args = ap.parse_args()
    os.makedirs(args.workdir, exist_ok=True)

    c0, rho0 = 340.29, 1.225
    Omega = args.mach * c0 / args.R0
    field = analytic.RotatingMonopoleField(1e-3, 2 * np.pi * args.freq,
                                           R0=args.R0, Omega=Omega,
                                           c0=c0, rho0=rho0)
    dt = (1.0 / args.freq) / args.ppw
    nT = int(args.revs * (2 * np.pi / Omega) / dt) + 1
    args.times = np.arange(nT) * dt
    lam = c0 / args.freq
    observers = {"mic_z": (0, 0, 20 * lam),        # on-axis (rotation axis)
                 "mic_x": (20 * lam, 0, 0),         # in-plane
                 "mic_d": (14.1 * lam, 0, 14.1 * lam)}
    print(f"rotating monopole: tipMach={field.tip_mach:.3f}, "
          f"f_rot={Omega/(2*np.pi):.1f} Hz, f_ac={args.freq:.0f} Hz, "
          f"records={nT}, span={args.revs} rev")

    saved = {} if args.save else None
    ok = True
    ok &= run_case(args, args.workdir, field, Omega, True, "corotating",
                   observers, saved)
    ok &= run_case(args, args.workdir, field, Omega, False, "stationary",
                   observers, saved)
    if args.save:
        np.savez(args.save, **saved); print("saved", args.save)
    print("\nROTATING-SOURCE VALIDATION " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
