#!/usr/bin/env python3
"""Plot the moving-surface invariance test -> moving_surface.pdf.

A static monopole enclosed by a rigidly rotating icosphere (tip Mach 0.3)
and by an oscillating icosphere (velocity amplitude 0.2 c0). The predicted
observer pressure must reproduce the analytic static-source far field
despite the surface motion, exercising the n-dot, M-dot and surface-velocity
terms.

Usage:
  python plot_moving.py --npz moving_signals.npz --out moving_surface.pdf
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
    ap.add_argument("--out", default="moving_surface.pdf")
    args = ap.parse_args()
    d = np.load(args.npz)

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.6))
    cases = [("rotating", "Rotating surface (tip $M=0.3$)"),
             ("oscillating", "Oscillating surface ($v=0.2\\,c_0$)")]
    for ax, (tag, title) in zip(axes, cases):
        t = d[f"{tag}_t"] * 1e3
        ax.plot(t, d[f"{tag}_exact"] * 1e3, "k-", lw=2.2,
                label="analytic (static source)")
        ax.plot(t, d[f"{tag}_pred"] * 1e3, "--", color="#d62728", lw=1.4,
                label="fwhFoam (moving surface)")
        err = float(d[f"{tag}_err"])
        ax.set_title(title)
        ax.set_xlabel("observer time [ms]")
        ax.set_ylabel("acoustic pressure [mPa]")
        ax.grid(True, ls=":", alpha=0.5)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        ax.text(0.03, 0.05, f"rel. $L_2$ = {err:.1e}",
                transform=ax.transAxes, fontsize=9, va="bottom")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
