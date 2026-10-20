#!/usr/bin/env python3
"""Plot the rotating-source (moving-surface) validation -> rotating_source.pdf.

A pulsating monopole orbits at tip Mach 0.3 (a moving source). Its exact
far field (analytic, retarded-time) is reproduced by the FW-H prediction on a
co-rotating surface (the moving-surface path) and, as a cross-check, on a
stationary surface enclosing the orbit.

Usage:
  python plot_rotating.py --npz rotating_signals.npz --out rotating_source.pdf
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", default="rotating_source.pdf")
    args = ap.parse_args()
    d = np.load(args.npz)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    cases = [("corotating", "Co-rotating surface (moving-surface path)"),
             ("stationary", "Stationary surface (cross-check)")]
    for ax, (tag, title) in zip(axes, cases):
        t = d[f"{tag}_t"] * 1e3
        ax.plot(t, d[f"{tag}_exact"] * 1e3, "k-", lw=2.2,
                label="analytic rotating source")
        ax.plot(t, d[f"{tag}_pred"] * 1e3, "--", color="#1f77b4", lw=1.3,
                label="fwhFoam")
        err = float(d[f"{tag}_err"])
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("observer time [ms]")
        ax.set_ylabel("acoustic pressure [mPa]")
        ax.grid(True, ls=":", alpha=0.5)
        pk = np.max(np.abs(d[f"{tag}_exact"] * 1e3))
        ax.set_ylim(-1.35 * pk, 1.7 * pk)         # headroom for the legend
        ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
        ax.text(0.03, 0.04, f"rel. $L_2$ = {err:.1e}",
                transform=ax.transAxes, fontsize=9, va="bottom")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
