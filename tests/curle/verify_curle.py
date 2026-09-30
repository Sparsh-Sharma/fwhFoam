#!/usr/bin/env python3
r"""Verify the Curle formulation mode.

Two checks on real cylinder-wall data:

1. *Reduction identity.* On a static no-slip wall, formulation 1A reduces
   exactly to Curle's loading integral, so `formulation Curle` and
   `formulation Farassat1A` must produce identical observer signals from
   the same wall data.

2. *Compact-dipole cross-check.* For an acoustically compact body the
   Curle integral reduces to the point-dipole field of the unsteady
   aerodynamic force,
       p'(x,t) = rhat.F(t-r/c0)/(4 pi r^2) + rhat.Fdot(t-r/c0)/(4 pi c0 r),
   which is evaluated here from the recorded force history and compared
   with the surface-integral prediction at the cross-flow observer.

Usage:
  python verify_curle.py --f1a DIR --curle DIR --coeffs coefficient.dat \
      --pos 0 75 0 [--rho 1.225 --U 1 --A 1 --c0 340 --tmin 70]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 "..", "..", "tools")))
from pyfwh import io, spectra  # noqa: E402


def read_coeffs(path):
    hdr, rows = [], []
    for ln in open(path):
        if ln.startswith("#"):
            hdr = ln[1:].split(); continue
        v = ln.split()
        if v: rows.append([float(x) for x in v])
    a = np.array(rows)
    icd = hdr.index("Cd") if "Cd" in hdr else 1
    icl = hdr.index("Cl") if "Cl" in hdr else 4
    return a[:, 0], a[:, icd], a[:, icl]


def compact_curle(tq, F, pos, c0):
    """Point-dipole (near+far) pressure at `pos` from force-on-fluid F(t)."""
    r = np.linalg.norm(pos); rhat = pos / r
    tret = tq - r / c0
    Fr = F @ rhat if F.ndim == 2 else F * 1.0
    dF = np.gradient(Fr, tq)
    # evaluate at emission time then shift: p(t) uses F(t - r/c0)
    return tret + r / c0, Fr / (4 * np.pi * r**2) + dF / (4 * np.pi * c0 * r)


def tone_amp(t, p, tmin):
    m = t >= tmin
    t, p = t[m], p[m] - p[m].mean()
    dt = np.mean(np.diff(t))
    f, g = spectra.psd(p, dt)
    k = 1 + int(np.argmax(g[1:]))
    return f[k], spectra.amplitude_at(p, dt, f[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--f1a", required=True)
    ap.add_argument("--curle", required=True)
    ap.add_argument("--coeffs", default="")
    ap.add_argument("--surface", default="",
                    help="glob of surfaceData_proc*.fwh: compute the "
                         "pressure force from the surface data itself")
    ap.add_argument("--observer", default="mic_090")
    ap.add_argument("--pos", nargs=3, type=float, default=[0, 75, 0])
    ap.add_argument("--rho", type=float, default=1.225)
    ap.add_argument("--U", type=float, default=1.0)
    ap.add_argument("--A", type=float, default=1.0)
    ap.add_argument("--c0", type=float, default=340.0)
    ap.add_argument("--tmin", type=float, default=70.0)
    args = ap.parse_args()

    sA = io.read_observer(os.path.join(args.f1a,
                                       f"observer_{args.observer}.dat"))
    sC = io.read_observer(os.path.join(args.curle,
                                       f"observer_{args.observer}.dat"))

    # 1. reduction identity
    n = min(len(sA["p"]), len(sC["p"]))
    num = np.linalg.norm(sA["p"][:n] - sC["p"][:n])
    den = max(np.linalg.norm(sA["p"][:n]), 1e-30)
    print(f"[1] Farassat1A vs Curle on wall data: relL2 diff = {num/den:.2e}"
          f"  -> {'IDENTICAL (reduction verified)' if num/den < 1e-12 else 'DIFFER'}")

    # 2. compact-dipole cross-check at the tone.
    # The dipole strength is the PRESSURE force the surface exerts on the
    # fluid, computed from the same surface data the integral uses (the
    # total aerodynamic force also contains the viscous part, which Curle's
    # pressure-only loading integral deliberately omits).
    import glob as _glob
    Ffiles = sorted(_glob.glob(args.coeffs)) if "*" in args.coeffs else None
    if args.surface:
        parts = [io.read(f) for f in sorted(_glob.glob(args.surface))]
        tq = parts[0].times
        Ffluid = np.zeros((len(tq), 3))
        for d in parts:
            Ffluid += np.einsum("tn,nj,n->tj", d.p, d.normals, d.areas)
    else:
        tq, Cd, Cl = read_coeffs(args.coeffs)
        q = 0.5 * args.rho * args.U**2 * args.A
        Ffluid = -q * np.stack([Cd, Cl, np.zeros_like(Cl)], axis=1)
    tc, pc = compact_curle(tq, Ffluid, np.array(args.pos), args.c0)
    f0s, aS = tone_amp(sC["t"], sC["p"], args.tmin)
    f0c, aC = tone_amp(tc, pc, args.tmin)
    rel = abs(aS - aC) / max(aC, 1e-30)
    print(f"[2] tone: surface-integral St={f0s:.4f} |p|={aS:.4e} Pa | "
          f"compact dipole St={f0c:.4f} |p|={aC:.4e} Pa | diff {rel*100:.1f}%")

    ok = (num / den < 1e-12) and (rel < 0.10)
    print("CURLE VERIFICATION " + ("PASSED" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
