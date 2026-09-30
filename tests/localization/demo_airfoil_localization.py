#!/usr/bin/env python3
r"""Demonstrate on-surface source localization by the diffraction filter.

A closed 3D airfoil surface carries a wall-pressure field that is the sum of
(i) a large-amplitude convecting hydrodynamic wave (non-radiating) and
(ii) a weak, compact acoustic source near the trailing edge. The raw
pressure magnitude |p| is dominated by the hydrodynamic wave and does not
reveal the source; the diffraction filter p_S = D[p] suppresses the
non-radiating content and localizes the radiating source at the trailing
edge --- the same behaviour the method shows on scale-resolved airfoil data.

This is a demonstration of the localization capability with a
physically-motivated field; the real-CFD application produces the same maps
from a simulation's wall pressure.

Writes a two-panel figure |p| vs |p_S| viewed on the airfoil surface.
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                 "..", "..", "tools")))
from pyfwh import filter as flt  # noqa: E402


def naca0012(x):
    t = 0.12
    return 5 * t * (0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2
                    + 0.2843*x**3 - 0.1015*x**4)


def airfoil_surface(nchord=80, nspan=24, span=0.6):
    """Closed NACA0012 surface (upper+lower + end caps) as triangles."""
    beta = np.linspace(0, np.pi, nchord)
    xc = 0.5 * (1 - np.cos(beta))                 # cosine clustering
    yt = naca0012(xc)
    zc = np.linspace(-span/2, span/2, nspan)
    verts, faces = [], []

    def vid(cache, key, pt):
        if key in cache:
            return cache[key]
        cache[key] = len(verts); verts.append(pt); return cache[key]

    cache = {}
    # upper (s=+1) and lower (s=-1) as two grids
    grid = {}
    for si, s in enumerate((+1, -1)):
        for i in range(nchord):
            for j in range(nspan):
                p = (xc[i], s*yt[i], zc[j])
                grid[(si, i, j)] = vid(cache, (round(p[0],6),round(p[1],6),round(p[2],6)), p)
    for si, out in ((0, True), (1, False)):
        for i in range(nchord-1):
            for j in range(nspan-1):
                a = grid[(si,i,j)]; b = grid[(si,i+1,j)]
                c = grid[(si,i+1,j+1)]; d = grid[(si,i,j+1)]
                if out: faces += [[a,b,c],[a,c,d]]
                else:   faces += [[a,c,b],[a,d,c]]
    # end caps at j=0 and j=nspan-1 (join upper/lower)
    for j in (0, nspan-1):
        for i in range(nchord-1):
            u0=grid[(0,i,j)]; u1=grid[(0,i+1,j)]
            l0=grid[(1,i,j)]; l1=grid[(1,i+1,j)]
            if j==0: faces += [[u0,l0,l1],[u0,l1,u1]]
            else:    faces += [[u0,l1,l0],[u0,u1,l1]]
    return np.array(verts), np.array(faces)


def face_geom(v, f):
    v0,v1,v2 = v[f[:,0]], v[f[:,1]], v[f[:,2]]
    cr = np.cross(v1-v0, v2-v0)
    a = 0.5*np.linalg.norm(cr, axis=1)
    keep = a > 1e-12                               # drop degenerate triangles
    f = f[keep]; cr = cr[keep]; a = a[keep]
    v0,v1,v2 = v[f[:,0]], v[f[:,1]], v[f[:,2]]
    c = (v0+v1+v2)/3
    n = cr/(2*a[:,None])
    return c, n, a, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="airfoil_localization.pdf")
    ap.add_argument("--freq", type=float, default=1100.0)
    ap.add_argument("--c0", type=float, default=340.0)
    ap.add_argument("--Mc", type=float, default=0.25,
                    help="convection Mach of the hydrodynamic wave (Uc/c0)")
    args = ap.parse_args()

    v, f = airfoil_surface(nchord=120, nspan=16)
    c, n, a, f = face_geom(v, f)
    k = 2*np.pi*args.freq/args.c0
    kh = k/args.Mc                      # subsonic convecting wavenumber >> k

    xchord = c[:,0]
    # (i) hydrodynamic convecting wave: large amplitude, subsonic phase
    #     speed Uc = Mc*c0  ->  surface wavenumber kh = k/Mc >> k
    p_hydro = 1.0*np.exp(1j*kh*xchord)*np.exp(-3*(xchord-0.5)**2)
    # (ii) compact radiating source at the trailing edge, 50x weaker
    src = 0.02*np.exp(-((xchord-0.97)/0.04)**2)
    p = p_hydro + src

    pS = flt.acoustic_surface_pressure(p, c, n, a, k)

    # rejection diagnostics: mid-chord (hydro) vs trailing edge (source)
    mid = (xchord > 0.35) & (xchord < 0.65)
    te = xchord > 0.9
    print(f"faces={len(c)}  kc={k:.1f}  kh/k={kh/k:.1f}")
    print(f"|p|  : midchord max={np.abs(p[mid]).max():.3e}  TE max={np.abs(p[te]).max():.3e}")
    print(f"|p_S|: midchord max={np.abs(pS[mid]).max():.3e}  TE max={np.abs(pS[te]).max():.3e}")
    print(f"hydrodynamic rejection at midchord: "
          f"{np.abs(p[mid]).max()/max(np.abs(pS[mid]).max(),1e-30):.1f}x")

    # ---- chord-span maps of the upper surface ----
    up = n[:,1] > 0.15                      # upper side, excluding caps
    X, Z = c[up,0], c[up,2]
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
    for ax, field, title in [
            (axs[0], np.abs(p[up]),  r"raw wall pressure $|\hat p|$"),
            (axs[1], np.abs(pS[up]), r"acoustic surface pressure $|\hat p_S|$")]:
        sc = ax.tripcolor(X, Z, field, cmap="inferno", shading="gouraud")
        cb = fig.colorbar(sc, ax=ax, shrink=0.9)
        cb.set_label("[arb.]")
        ax.set_xlabel("chordwise $x/c$")
        ax.set_title(title)
        ax.set_xlim(0, 1)
    axs[0].set_ylabel("span $z/c$")
    fig.suptitle("Diffraction-filter source localization (upper surface): the"
                 " filter rejects the convecting wave and reveals the"
                 " trailing-edge source")
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight", dpi=150)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
