#!/usr/bin/env python3
"""Post-process the cylinder Aeolian-tone tutorial -> cylinder.pdf.

Reads the fwh observer signals and (optionally) the forceCoeffs history,
and produces:
  (left)  far-field pressure spectrum at the cross-flow observer, with the
          vortex-shedding Strouhal number marked from the lift spectrum;
  (right) directivity of the RMS acoustic pressure over the observers.

Usage:
  python plot_cylinder.py --case /path/to/case --out figs/cylinder.pdf
      [--U 1.0 --D 1.0]
"""
from __future__ import annotations
import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 "..", "..", "tools")))
from pyfwh import io, spectra  # noqa: E402


def load_coeff_lift(case):
    """Return (t, Cl) from postProcessing/forceCoeffs if present."""
    pats = glob.glob(os.path.join(case, "postProcessing", "forceCoeffs",
                                  "*", "coefficient.dat"))
    if not pats:
        return None, None
    t, cl = [], []
    with open(sorted(pats)[0]) as f:
        header = []
        for line in f:
            if line.startswith("#"):
                header = line[1:].split()
                continue
            v = line.split()
            if not v:
                continue
            t.append(float(v[0]))
            # column named 'Cl' (fallback: 4th col)
            idx = header.index("Cl") if "Cl" in header else 3
            cl.append(float(v[idx]))
    return np.array(t), np.array(cl)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--out", default="cylinder.pdf")
    ap.add_argument("--U", type=float, default=1.0)
    ap.add_argument("--D", type=float, default=1.0)
    ap.add_argument("--acdir", default=None,
                    help="observer directory (default: runtime acousticData; "
                         "point at an fwhSolve-output for a dense arc)")
    args = ap.parse_args()

    acdir = args.acdir or os.path.join(
        args.case, "postProcessing", "fwhWall", "acousticData")
    obs_files = sorted(glob.glob(os.path.join(acdir, "observer_*.dat")))
    if not obs_files:
        sys.exit(f"no observer files in {acdir}")

    sigs = {}
    for fpath in obs_files:
        name = os.path.basename(fpath)[len("observer_"):-len(".dat")]
        s = io.read_observer(fpath)
        tw = s["meta"].get("validWindow", (s["t"][0], s["t"][-1]))
        m = (s["t"] >= tw[0]) & (s["t"] <= tw[1])
        if m.sum() < 16:
            m = np.ones_like(s["t"], dtype=bool)
        sigs[name] = {
            "t": s["t"][m], "p": s["p"][m],
            "pos": s["meta"].get("position", np.zeros(3)),
        }

    # Reference shedding Strouhal from the lift spectrum
    St_shed = None
    tcl, cl = load_coeff_lift(args.case)
    if tcl is not None and len(tcl) > 32:
        dtl = np.mean(np.diff(tcl))
        # use last 70 % (developed shedding)
        i0 = int(0.3 * len(cl))
        f, g = spectra.psd(cl[i0:], dtl)
        fpk = f[1 + int(np.argmax(g[1:]))]
        St_shed = fpk * args.D / args.U

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 3.8))

    # Spectrum at the loudest observer (the cross-flow lift-dipole peak)
    name_cf = max(sigs, key=lambda k:
                  np.sqrt(np.mean((sigs[k]["p"] - sigs[k]["p"].mean())**2)))
    s = sigs[name_cf]
    dt = np.mean(np.diff(s["t"]))
    f, Lp = spectra.spl(s["p"], dt)
    St = f * args.D / args.U
    ax1.plot(St, Lp, color="#1f77b4")
    if St_shed:
        ax1.axvline(St_shed, color="k", ls="--", lw=1,
                    label=f"shedding St={St_shed:.3f}")
        ax1.legend(frameon=False)
    ax1.set_xlim(0, 1.0)
    ax1.set_xlabel(r"Strouhal number $St=fD/U$")
    ax1.set_ylabel(r"SPL [dB re 20 $\mu$Pa]")
    ax1.set_title(f"Far-field spectrum ({name_cf})")
    ax1.grid(True, ls=":", alpha=0.5)

    # Directivity: RMS pressure vs observer angle
    angs, rms = [], []
    for name, s in sigs.items():
        pos = np.asarray(s["pos"], dtype=float)
        ang = np.degrees(np.arctan2(pos[1], pos[0]))
        angs.append(ang)
        rms.append(np.sqrt(np.mean((s["p"] - s["p"].mean()) ** 2)))
    order = np.argsort(angs)
    angs = np.array(angs)[order]; rms = np.array(rms)[order]
    ax2 = plt.subplot(1, 2, 2, projection="polar")
    th = np.radians(angs)
    ax2.plot(th, rms, "o-", color="#d62728")
    ax2.set_theta_zero_location("E")
    ax2.set_title("RMS acoustic pressure directivity", va="bottom")

    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out, "| shedding St =", St_shed)


if __name__ == "__main__":
    main()
