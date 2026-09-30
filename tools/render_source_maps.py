#!/usr/bin/env python3
r"""Render on-surface acoustic source-localization maps (colour images).

Takes per-face fields on a surface --- the raw wall-pressure magnitude
|p_hat|^2 and the diffraction-filtered acoustic surface pressure |p_S|^2 at
one or more frequencies (as produced by ``pyfwh.filter`` from surface
spectra) --- and renders chord--span maps of the suction side in dB:
the raw pressure shows the spread hydrodynamic turbulence, the filtered
field localizes the actual acoustic sources (e.g. at the trailing edge).

Input: an .npz with arrays
    cen (N,3), nrm (N,3), area (N,), freqs (F,),
    praw2_<f> (N,), pS2_<f> (N,)   for each frequency f in freqs.

Usage:
  python render_source_maps.py --data sigma_maps.npz --out source_maps.png
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation


def chordwise(cen):
    """Chordwise fraction x/c along the principal axis of the section."""
    xy = cen[:, :2].astype(float)
    xym = xy - xy.mean(0)
    w, V = np.linalg.eigh(np.cov(xym.T))
    e = V[:, np.argmax(w)]
    if e[0] < 0:
        e = -e
    sc = xym @ e
    sc = sc - sc.min()
    return sc / sc.max()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="source_maps.png")
    ap.add_argument("--floor", type=float, default=-40.0)
    args = ap.parse_args()

    d = np.load(args.data)
    cen, nrm = d["cen"], d["nrm"]
    freqs = [int(f) for f in d["freqs"]]

    ss = nrm[:, 1] > 0.0                      # suction (upper) side
    xc = chordwise(cen)[ss]
    zc = cen[ss, 2]
    tri = Triangulation(xc, zc)

    fig, axes = plt.subplots(2, len(freqs), figsize=(10.5, 3.6),
                             sharex=True, sharey=True, layout="constrained")
    if len(freqs) == 1:
        axes = axes.reshape(2, 1)
    levels = np.linspace(args.floor, 0, 21)
    for j, fb in enumerate(freqs):
        praw = d[f"praw2_{fb}"][ss]
        pS = d[f"pS2_{fb}"][ss]
        ref = praw.max()
        for i, (val, lab) in enumerate([(pS, r"$|\hat p_S|$ (sources)"),
                                        (praw, r"$|\hat p|$ (raw)")]):
            dB = 10.0 * np.log10(np.maximum(val / ref, 10**(args.floor / 10)))
            cf = axes[i, j].tricontourf(tri, dB, levels=levels,
                                        cmap="magma", extend="min")
            cf.set_rasterized(True)
            axes[i, j].set_aspect("auto")
            axes[i, j].set_xlim(0, 1)
            if i == 0:
                axes[i, j].set_title(f"{fb} Hz")
            if j == 0:
                axes[i, j].set_ylabel(lab + "\n$z$ [m]")
            if i == 1:
                axes[i, j].set_xlabel(r"$x/c$")
    cb = fig.colorbar(cf, ax=axes, shrink=0.85, pad=0.015,
                      ticks=np.arange(args.floor, 1, 10))
    cb.set_label(r"dB re $\max|\hat p|$")
    fig.suptitle("On-surface source localization: raw wall pressure vs "
                 "diffraction-filtered acoustic surface pressure "
                 "(SD7003 suction side, LES data)")
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print("wrote", args.out, f"({ss.sum()} suction-side faces)")


if __name__ == "__main__":
    main()
