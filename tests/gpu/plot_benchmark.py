#!/usr/bin/env python3
"""Plot the GPU benchmark -> gpu_benchmark.pdf.

Left: wall-clock time per frequency line of the sigma density and the
diffraction filter, CPU (NumPy) vs GPU (CuPy), against face count.
Right: end-to-end FW-H solver time (faces x observers x time) and the
achieved GPU speedup.

Usage:
  python plot_benchmark.py --json benchmark_gpu.json --out gpu_benchmark.pdf
"""
from __future__ import annotations
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def col(rows, key):
    return np.array([r[key] for r in rows if key in r], float)


def col_N(rows, key):
    return np.array([r["N"] for r in rows if key in r], float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--out", default="gpu_benchmark.pdf")
    args = ap.parse_args()

    d = json.load(open(args.json))
    rows, fwh = d["rows"], d.get("fwh_rows", [])
    gname = d.get("gpu") or "GPU"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.8))

    # --- kernels: time vs N -------------------------------------------
    for key, c, m, lab in [
        ("sigma_cpu_s", "#1f77b4", "o", r"$\sigma$ CPU"),
        ("sigma_gpu_s", "#1f77b4", "s", r"$\sigma$ GPU"),
        ("filter_cpu_s", "#d62728", "o", "filter CPU"),
        ("filter_gpu_s", "#d62728", "s", "filter GPU"),
    ]:
        y, x = col(rows, key), col_N(rows, key)
        ls = "--" if "cpu" in key else "-"
        fill = "none" if "cpu" in key else c
        ax1.loglog(x, y, m + ls, color=c, mfc=fill, label=lab, lw=1.4)
    ax1.set_xlabel(r"surface faces $N$")
    ax1.set_ylabel("time per frequency line [s]")
    ax1.set_title("Localization kernels")
    ax1.grid(True, which="both", ls=":", alpha=0.5)
    ax1.legend(frameon=False, fontsize=8, ncol=2)

    # --- FW-H solver: time + speedup ----------------------------------
    xc = col_N(fwh, "fwh_cpu_s")
    yc = col(fwh, "fwh_cpu_s")
    xg = col_N(fwh, "fwh_gpu_s")
    yg = col(fwh, "fwh_gpu_s")
    ax2.loglog(xc, yc, "o--", color="#2ca02c", mfc="none", label="FW-H CPU", lw=1.4)
    ax2.loglog(xg, yg, "s-", color="#2ca02c", label="FW-H GPU", lw=1.4)
    ax2.set_xlabel(r"surface faces $N$")
    ax2.set_ylabel("solver wall-clock [s]")
    ax2.set_title(f"FW-H solver ({fwh[0]['nObs']} obs, {fwh[0]['nT']} steps)")
    ax2.grid(True, which="both", ls=":", alpha=0.5)
    ax2.legend(frameon=False, fontsize=8, loc="upper left")

    # annotate the top speedups
    sp = [(r["N"], r.get("sigma_speedup")) for r in rows if "sigma_speedup" in r]
    fsp = [(r["N"], r.get("filter_speedup")) for r in rows if "filter_speedup" in r]
    best_sig = max(s for _, s in sp)
    best_fil = max(s for _, s in fsp)
    ax1.text(0.03, 0.03,
             f"up to {best_sig:.0f}$\\times$ ($\\sigma$), "
             f"{best_fil:.0f}$\\times$ (filter)",
             transform=ax1.transAxes, fontsize=8, va="bottom")

    fig.suptitle(f"CPU (NumPy) vs GPU (CuPy, {gname})", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
