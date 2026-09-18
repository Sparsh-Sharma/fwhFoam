#!/usr/bin/env python3
"""Plot the spatial and temporal convergence study -> convergence.pdf.

Usage:
  python plot_convergence.py --csv _conv_work/convergence.csv --out figs/convergence.pdf
"""
from __future__ import annotations
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="convergence.pdf")
    args = ap.parse_args()

    spatial, temporal = [], []
    with open(args.csv) as f:
        for row in csv.DictReader(f):
            rec = (float(row["nFaces_or_dt"]), float(row["relL2"]))
            (spatial if row["kind"] == "spatial" else temporal).append(rec)

    spatial.sort(); temporal.sort()
    sN = np.array([r[0] for r in spatial]); sE = np.array([r[1] for r in spatial])
    h = 1.0 / np.sqrt(sN)
    tD = np.array([r[0] for r in temporal]); tE = np.array([r[1] for r in temporal])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))

    ax1.loglog(h, sE, "o-", color="#1f77b4", label="fwhFoam")
    ref = sE[-1] * (h / h[-1]) ** 2
    ax1.loglog(h, ref, "k--", lw=1, label=r"2nd order")
    ax1.set_xlabel(r"$h \sim 1/\sqrt{N_\mathrm{faces}}$")
    ax1.set_ylabel(r"relative $L_2$ error")
    ax1.set_title("Spatial convergence")
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend(frameon=False)

    ax2.loglog(tD, tE, "s-", color="#d62728", label="fwhFoam")
    ref2 = tE[-1] * (tD / tD[-1]) ** 2
    ax2.loglog(tD, ref2, "k--", lw=1, label=r"2nd order")
    ax2.set_xlabel(r"time step $\Delta t$ [s]")
    ax2.set_ylabel(r"relative $L_2$ error")
    ax2.set_title("Temporal convergence")
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
