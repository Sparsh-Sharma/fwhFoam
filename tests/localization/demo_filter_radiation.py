#!/usr/bin/env python3
r"""Ground-truthed, reproducible demonstration of the diffraction filter.

This case is fully self-contained --- it generates its own surface and its own
surface-pressure field, with no external data --- so it can be rerun by anyone
with the released code, and it has a known ground truth.

A closed, streamlined body (a prolate spheroid) carries a complex
surface-pressure field that is the sum of two known parts at one frequency:

  * an ACOUSTIC part: the outgoing trace of a point monopole placed inside the
    body near the downstream tip --- a genuinely radiating field whose surface
    footprint varies on the acoustic scale k = omega/c0;
  * a HYDRODYNAMIC part: a convecting surface-pressure wavepacket travelling
    along the body at a low convection Mach number M_c, so its surface
    wavenumber k_c = k/M_c is far larger than k. This is non-radiating, and it
    is made 30 dB louder than the acoustic part, as a turbulent boundary-layer
    footprint is on a real body.

The raw surface-pressure magnitude is therefore dominated by the hydrodynamic
wavepacket and peaks where there is no acoustic source. Applying the released
diffraction filter (pyfwh.filter / pyfwh.gpu, numerically identical) returns
the acoustic surface pressure p_S, which should suppress the non-radiating
wavepacket and recover the true (known) source region near the tip.

Ground-truth metrics printed and asserted:
  * the filtered map's peak lies in the source region, not mid-body;
  * |p_S| correlates with the known acoustic field, not the hydrodynamic one;
  * the hydrodynamic mid-body region is suppressed by many dB relative to raw.

Usage:
  python demo_filter_radiation.py [--sub 4] [--freq 2000] [--out fig.png]
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "tools")))
from pyfwh import geometry, filter as dfilter, gpu  # noqa: E402


def prolate_spheroid(sub, a, b):
    """Closed prolate spheroid (x-elongated) from a deformed icosphere.

    Returns face centroids, outward unit normals and areas.
    """
    _, _, _, verts, faces = geometry.icosphere(radius=1.0, subdivisions=sub)
    v = verts.copy()
    v[:, 0] *= a
    v[:, 1] *= b
    v[:, 2] *= b
    v0, v1, v2 = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    cen = (v0 + v1 + v2) / 3.0
    nrm = np.cross(v1 - v0, v2 - v0)
    area = 0.5 * np.linalg.norm(nrm, axis=1)
    nrm = nrm / np.linalg.norm(nrm, axis=1)[:, None]
    # orient outward (spheroid centred at origin)
    flip = np.einsum("ij,ij->i", nrm, cen) < 0
    nrm[flip] *= -1.0
    return cen, nrm, area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", type=int, default=5)
    ap.add_argument("--freq", type=float, default=2000.0)
    ap.add_argument("--a", type=float, default=0.30, help="semi-length [m]")
    ap.add_argument("--b", type=float, default=0.05, help="semi-thickness [m]")
    ap.add_argument("--Mc", type=float, default=0.08, help="convection Mach")
    ap.add_argument("--gap", type=float, default=18.0,
                    help="hydrodynamic-over-acoustic level [dB]")
    ap.add_argument("--out", default=os.path.join(HERE, "filter_radiation.png"))
    args = ap.parse_args()

    c0, rho0 = 340.0, 1.225
    omega = 2 * np.pi * args.freq
    k = omega / c0
    kc = k / args.Mc                      # convective (hydrodynamic) wavenumber

    cen, nrm, area = prolate_spheroid(args.sub, args.a, args.b)
    N = len(cen)
    x, z = cen[:, 0], cen[:, 2]

    # resolution check: faces per convective wavelength (axial)
    dx = np.median(np.abs(np.diff(np.sort(np.unique(np.round(x, 4))))))
    ppw_c = (2 * np.pi / kc) / max(dx, 1e-9)

    # --- known acoustic source: monopole just inside the downstream tip ------
    s = np.array([0.75 * args.a, 0.0, 0.0])
    r = np.linalg.norm(cen - s, axis=1)
    p_ac = np.exp(-1j * k * r) / (4 * np.pi * r)      # outgoing monopole trace

    # --- known hydrodynamic wavepacket: convecting, tapered over the body ----
    env = np.exp(-((x - 0.0) / (0.6 * args.a)) ** 2)  # centred mid-body
    p_hy = env * np.exp(1j * kc * x)

    # scale so the hydrodynamic part is `gap` dB louder than the acoustic part
    p_ac /= np.sqrt(np.mean(np.abs(p_ac) ** 2))
    p_hy /= np.sqrt(np.mean(np.abs(p_hy) ** 2))
    A_h = 10 ** (args.gap / 20)
    phat = p_ac + A_h * p_hy

    # --- released diffraction filter -----------------------------------------
    # chunked evaluator (the GPU-capable path, run on the CPU backend) so the
    # dense N x N operator need not be materialised at this resolution
    pS = gpu.acoustic_surface_pressure(phat, cen, nrm, area, k, xp=np)
    # on a small mesh, confirm it matches the dense reference operator exactly
    if N <= 6000:
        pS_dense = dfilter.acoustic_surface_pressure(phat, cen, nrm, area, k)
        assert np.max(np.abs(pS - pS_dense)) / np.max(np.abs(pS)) < 1e-10

    # --- ground-truth metrics -------------------------------------------------
    raw = np.abs(phat)
    flt = np.abs(pS)
    aco = np.abs(p_ac)
    hyd = np.abs(p_hy)

    def corr(u, v):
        u = u - u.mean(); v = v - v.mean()
        return float(u @ v / (np.linalg.norm(u) * np.linalg.norm(v)))

    src = x > 0.5 * args.a                 # downstream-tip source region
    mid = np.abs(x) < 0.3 * args.a         # mid-body hydrodynamic region

    raw_peak_x = x[np.argmax(raw)]
    flt_peak_x = x[np.argmax(flt)]
    supp_mid = 20 * np.log10(np.median(raw[mid]) / np.median(flt[mid]))
    c_flt_ac = corr(flt, aco)
    c_raw_ac = corr(raw, aco)
    c_flt_hy = corr(flt, hyd)

    print(f"faces N = {N},  f = {args.freq:.0f} Hz,  k = {k:.1f} 1/m,  "
          f"kc = {kc:.1f} 1/m  (kc/k = {kc/k:.1f}),  ~{ppw_c:.0f} faces/lambda_c")
    print(f"raw  |p| peak at x/a = {raw_peak_x/args.a:+.2f}  "
          f"(hydrodynamic; true source at x/a = +0.75)")
    print(f"filt |pS| peak at x/a = {flt_peak_x/args.a:+.2f}  "
          f"(should be near the source)")
    print(f"mid-body hydrodynamic suppression: {supp_mid:.1f} dB")
    print(f"corr(|pS|,|p_ac|) = {c_flt_ac:+.2f}   "
          f"corr(|p_raw|,|p_ac|) = {c_raw_ac:+.2f}   "
          f"corr(|pS|,|p_hy|) = {c_flt_hy:+.2f}")

    ok = (flt_peak_x > 0.4 * args.a          # filter peak in source region
          and raw_peak_x < 0.4 * args.a      # raw peak misled to mid-body
          and supp_mid > 10.0                # real suppression
          and c_flt_ac > 0.5                 # filter recovers acoustic field
          and c_raw_ac < c_flt_ac)           # better than raw
    print("\nFILTER RADIATION DEMO " + ("PASSED" if ok else "FAILED"))

    _render(cen, nrm, args, raw, flt, s, supp_mid)
    sys.exit(0 if ok else 1)


def _render(cen, nrm, args, raw, flt, s, supp_mid):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    up = nrm[:, 1] > 0.2                      # one side of the body
    xa = cen[up, 0] / args.a
    za = cen[up, 2] / args.b
    tri = Triangulation(xa, za)
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.2), sharex=True,
                             layout="constrained")
    for ax, val, lab in [(axes[0], flt, r"$|\hat p_S|$ (filtered: sources)"),
                         (axes[1], raw, r"$|\hat p|$ (raw)")]:
        v = val[up]
        dB = 20 * np.log10(np.maximum(v / v.max(), 1e-3))
        cf = ax.tricontourf(tri, dB, levels=np.linspace(-40, 0, 21),
                            cmap="magma", extend="min")
        ax.axvline(s[0] / args.a, color="cyan", ls="--", lw=1.2)
        ax.set_ylabel(lab + "\n$z/b$")
        ax.set_aspect("auto")
    axes[1].set_xlabel(r"$x/a$  (dashed: true acoustic source)")
    cb = fig.colorbar(cf, ax=axes, shrink=0.9)
    cb.set_label(r"dB re max")
    fig.suptitle(f"Diffraction filter on a controlled field: a "
                 f"{args.gap:.0f}-dB-louder convecting wavepacket\nis "
                 f"suppressed by {supp_mid:.0f} dB; the buried tip source "
                 f"is recovered")
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
