#!/usr/bin/env python3
r"""Plot the sigma source-localization map on the cylinder control surface.

Reads sigma_map.csv (written by tools/sigma_localize.py) for the permeable
cylindrical control surface and produces:
  (left)  sigma vs azimuthal angle around the shell -- where, around the
          cylinder, the radiated power at the shedding tone originates;
  (right) a plan-view scatter of the shell coloured by sigma.

Usage:
  python plot_sigma_cylinder.py --csv sigma_map.csv --out sigma_cylinder.pdf
"""
from __future__ import annotations
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="sigma_cylinder.pdf")
    args = ap.parse_args()

    d = np.genfromtxt(args.csv, delimiter=",", names=True)
    x, y, sigma = d["x"], d["y"], d["sigma"]
    theta = np.degrees(np.arctan2(y, x))

    # bin sigma by azimuth (average over the small z-extent)
    nb = 72
    edges = np.linspace(-180, 180, nb + 1)
    idx = np.clip(np.digitize(theta, edges) - 1, 0, nb - 1)
    prof = np.array([sigma[idx == b].mean() if np.any(idx == b) else 0.0
                     for b in range(nb)])
    centres = 0.5 * (edges[:-1] + edges[1:])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    ax1.plot(centres, prof, color="#1f77b4")
    ax1.axhline(0, color="k", lw=0.6)
    for a in (90, -90):
        ax1.axvline(a, color="#d62728", ls="--", lw=1, alpha=0.7)
    ax1.set_xlim(-180, 180)
    ax1.set_xticks([-180, -90, 0, 90, 180])
    ax1.set_xlabel(r"azimuth around control surface $\theta$ [deg]")
    ax1.set_ylabel(r"$\sigma$ [W/m$^2$]")
    ax1.set_title(r"Source-power density vs azimuth")
    ax1.grid(True, ls=":", alpha=0.5)

    sc = ax2.scatter(x, y, c=sigma, cmap="RdBu_r",
                     vmin=-np.abs(sigma).max(), vmax=np.abs(sigma).max(), s=14)
    ax2.set_aspect("equal")
    ax2.set_xlabel("x"); ax2.set_ylabel("y")
    ax2.set_title(r"$\sigma$ on the control surface (flow $\rightarrow$)")
    fig.colorbar(sc, ax=ax2, label=r"$\sigma$ [W/m$^2$]", shrink=0.8)

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
