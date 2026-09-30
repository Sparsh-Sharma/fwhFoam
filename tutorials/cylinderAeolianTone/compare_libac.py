#!/usr/bin/env python3
r"""Cross-validate fwhFoam against libAcoustics on the same cylinder case.

Reads the fwhFoam observer signal and the libAcoustics '<name>-time.dat'
output at the same observer, overlays the far-field pressure spectra, and
reports the agreement in tonal frequency and amplitude.

Usage:
  python compare_libac.py --case /path/to/cyl_libac --observer mic_090 \
      --libac acousticData/libac-time.dat --out compare_libac.pdf
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "tools")))
from pyfwh import io, spectra  # noqa: E402


def read_libac(path, observer):
    """Read libAcoustics <name>-time.dat -> (t, p) for the named observer."""
    with open(path) as f:
        header = f.readline().split()
    col = None
    for i, h in enumerate(header):
        if h.startswith(observer):
            col = i; break
    if col is None:
        sys.exit(f"observer {observer} not in libAcoustics header {header}")
    a = np.genfromtxt(path, skip_header=1)
    return a[:, 0], a[:, col]


def tone(t, p, U=1.0, D=1.0):
    p = p - np.mean(p)
    dt = np.mean(np.diff(t))
    f, g = spectra.psd(p, dt)
    k = 1 + int(np.argmax(g[1:]))
    return f, g, f[k], spectra.amplitude_at(p, dt, f[k])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", required=True)
    ap.add_argument("--observer", default="mic_090")
    ap.add_argument("--libac", default="acousticData/libac-time.dat")
    ap.add_argument("--fwh", default=None)
    ap.add_argument("--U", type=float, default=1.0)
    ap.add_argument("--D", type=float, default=1.0)
    ap.add_argument("--tmin", type=float, default=50.0,
                    help="start of the developed-shedding analysis window [s]")
    ap.add_argument("--out", default="compare_libac.pdf")
    args = ap.parse_args()

    fwh_path = args.fwh or os.path.join(
        args.case, "postProcessing", "fwhWall", "acousticData",
        f"observer_{args.observer}.dat")
    s = io.read_observer(fwh_path)
    m = s["t"] >= args.tmin
    tf, pf = s["t"][m], s["p"][m]

    tl, pl = read_libac(os.path.join(args.case, args.libac), args.observer)
    ml = tl >= args.tmin
    tl, pl = tl[ml], pl[ml]

    ff, gf, f0f, af = tone(tf, pf, args.U, args.D)
    fl, gl, f0l, al = tone(tl, pl, args.U, args.D)

    Stf, Stl = f0f * args.D / args.U, f0l * args.D / args.U
    amp_rel = abs(af - al) / max(al, 1e-30)
    print(f"fwhFoam:      St={Stf:.4f}  tone |p|={af:.4e} Pa")
    print(f"libAcoustics: St={Stl:.4f}  tone |p|={al:.4e} Pa")
    print(f"tonal-amplitude relative difference = {amp_rel*100:.2f} %")

    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.semilogy(ff * args.D / args.U, gf, color="#1f77b4", label="fwhFoam")
    ax.semilogy(fl * args.D / args.U, gl, color="#d62728", ls="--",
                label="libAcoustics")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel(r"Strouhal number $St=fD/U$")
    ax.set_ylabel(r"PSD of $p'$ [Pa$^2$/Hz]")
    ax.set_title(f"fwhFoam vs libAcoustics ({args.observer})")
    ax.legend(frameon=False); ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout(); fig.savefig(args.out, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
