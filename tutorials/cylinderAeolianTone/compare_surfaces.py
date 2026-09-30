#!/usr/bin/env python3
r"""Compare impermeable (wall) and permeable FW-H surfaces at several radii.

Reads the fwhFoam observer signals produced by the wall and the permeable
surfaces (r = 4, 6, 8 D) from one cylinder run, overlays the far-field
spectra at a chosen observer, and reports the tonal level of each surface
and the sensitivity of the permeable prediction to the surface radius.

Usage:
  python compare_surfaces.py --case /path/to/cyl_multi --observer mic_090 \
      --out compare_surfaces.pdf
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tools")))
from pyfwh import io, spectra  # noqa: E402

SURFS = [("fwhWall", "impermeable wall", "#111111", "-"),
         ("fwhPerm4", "permeable r=4D", "#1f77b4", "--"),
         ("fwhPerm6", "permeable r=6D", "#2ca02c", "-."),
         ("fwhPerm8", "permeable r=8D", "#d62728", ":")]


def spec(case, func, obs, tmin, U, D, tone_hz=None):
    """Return (St_axis, PSD, dt, series) and, if tone_hz given, the tonal
    amplitude evaluated at that fixed frequency (the acoustic tone)."""
    p = os.path.join(case, "postProcessing", func, "acousticData",
                     f"observer_{obs}.dat")
    if not os.path.exists(p):
        return None
    s = io.read_observer(p)
    m = s["t"] >= tmin
    t, pr = s["t"][m], s["p"][m]
    pr = pr - pr.mean()
    dt = np.mean(np.diff(t))
    f, g = spectra.psd(pr, dt)
    if tone_hz is None:
        k = 1 + int(np.argmax(g[1:]))
        tone_hz = f[k]
    amp = spectra.amplitude_at(pr, dt, tone_hz)          # at the tone
    amp2f = spectra.amplitude_at(pr, dt, 2 * tone_hz)     # wake pseudo-sound
    return dict(St=f * D / U, psd=g, tone_hz=tone_hz,
                amp=amp, amp2f=amp2f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--observer", default="mic_090")
    ap.add_argument("--U", type=float, default=1.0)
    ap.add_argument("--D", type=float, default=1.0)
    ap.add_argument("--tmin", type=float, default=60.0)
    ap.add_argument("--out", default="compare_surfaces.pdf")
    args = ap.parse_args()

    # the acoustic tone frequency is fixed by the impermeable wall (clean)
    wall = spec(args.case, "fwhWall", args.observer, args.tmin, args.U, args.D)
    tone_hz = wall["tone_hz"]
    tone_St = tone_hz * args.D / args.U

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    print(f"acoustic tone (from wall): St={tone_St:.4f}")
    print(f"{'surface':18} {'|p|@tone[Pa]':>14} {'|p|@2f[Pa]':>14}")
    amps = {}
    for func, label, c, ls in SURFS:
        r = spec(args.case, func, args.observer, args.tmin, args.U, args.D,
                 tone_hz=tone_hz)
        if r is None:
            print(f"{label:18} (missing)"); continue
        amps[func] = r["amp"]
        print(f"{label:18} {r['amp']:14.4e} {r['amp2f']:14.4e}")
        ax.semilogy(r["St"], r["psd"], color=c, ls=ls, label=label)
    ax.axvline(tone_St, color="0.5", lw=0.8, ls="-", alpha=0.6)

    if all(k in amps for k in ("fwhPerm4", "fwhPerm6", "fwhPerm8")):
        vals = [amps[k] for k in ("fwhPerm4", "fwhPerm6", "fwhPerm8")]
        spread = (max(vals) - min(vals)) / np.mean(vals)
        print(f"permeable-radius sensitivity at the tone = {spread*100:.1f} %")
    if "fwhWall" in amps and "fwhPerm8" in amps:
        d = abs(amps["fwhWall"] - amps["fwhPerm8"]) / amps["fwhWall"]
        print(f"impermeable vs permeable(r=8D) at the tone = {d*100:.1f} %")

    ax.set_xlim(0, 1.0)
    ax.set_xlabel(r"Strouhal number $St=fD/U$")
    ax.set_ylabel(r"PSD of $p'$ [Pa$^2$/Hz]")
    ax.set_title(f"Impermeable vs permeable surfaces ({args.observer})")
    ax.legend(frameon=False); ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
