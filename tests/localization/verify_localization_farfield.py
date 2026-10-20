#!/usr/bin/env python3
r"""Far-field confirmation that the localised regions are the true radiators.

Addresses the objection that regions of strong surface pressure need not be
regions of efficient radiation (a convecting boundary-layer footprint is loud
but radiates weakly). The test uses exact fields on a flat plate, for which a
convecting wall-pressure wavepacket is a genuinely non-radiating (evanescent)
field, and computes the radiation diagnostics with the released code
(``pyfwh.localization``).

On the plate are superposed, at one frequency:
  * a point monopole just below the plate near x = x_s --- an efficient
    radiator with exact Cauchy data (p and v_n);
  * an evanescent wall-pressure wavepacket convecting at Mach ``Mc`` (surface
    wavenumber k_c = k / Mc > k, normal decay kappa = sqrt(k_c^2 - k^2)), whose
    Cauchy data (p, v_n) are those of an exact evanescent field, so it radiates
    zero far field by construction. Its surface pressure is set ``gap`` dB
    above the monopole's.

Reported:
  1. the surface sound-power density sigma localises the radiated power; its
     fraction in the monopole region is compared with that of the raw |p|^2;
  2. the far-field directivity from the full surface matches the monopole;
  3. the wavepacket is +gap dB on the surface but radiates far less, quantified
     as a level difference in the far field.

Usage:
  python verify_localization_farfield.py [--freq 2000] [--gap 18] [--out f.pdf]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "tools")))
from pyfwh import localization, analytic, gpu, plotstyle  # noqa: E402


def flat_plate(Lx, Ly, nx, ny):
    xs = np.linspace(-Lx / 2, Lx / 2, nx)
    ys = np.linspace(-Ly / 2, Ly / 2, ny)
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    cen = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], axis=1)
    nrm = np.tile([0.0, 0.0, 1.0], (cen.shape[0], 1))
    area = np.full(cen.shape[0], (Lx / nx) * (Ly / ny))
    return cen, nrm, area


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freq", type=float, default=2000.0)
    ap.add_argument("--Mc", type=float, default=0.08)
    ap.add_argument("--gap", type=float, default=18.0)
    ap.add_argument("--Lx", type=float, default=0.6)
    ap.add_argument("--Ly", type=float, default=0.24)
    ap.add_argument("--nx", type=int, default=170)
    ap.add_argument("--ny", type=int, default=44)
    ap.add_argument("--xs", type=float, default=0.20, help="monopole x [m]")
    ap.add_argument("--out", default=os.path.join(HERE, "localization_farfield.pdf"))
    args = ap.parse_args()

    c0, rho0 = 340.0, 1.225
    omega = 2 * np.pi * args.freq
    k = omega / c0
    kc = k / args.Mc
    kappa = np.sqrt(kc**2 - k**2)

    cen, nrm, area = flat_plate(args.Lx, args.Ly, args.nx, args.ny)
    x = cen[:, 0]

    # monopole just below the plate: exact radiating Cauchy data
    s = np.array([args.xs, 0.0, -0.03])
    mono = analytic.MonopoleField(1e-3, omega, c0=c0, rho0=rho0, x0=s)
    p_ac = mono.pressure(cen, 0.0)
    vn_ac = np.einsum("ni,ni->n", mono.velocity(cen, 0.0), nrm)

    # evanescent convecting wavepacket: exact non-radiating Cauchy data
    env = np.exp(-((x) / (0.6 * args.Lx / 2)) ** 2)
    p_hy = env * np.exp(1j * kc * x)
    vn_hy = (kappa / (1j * omega * rho0)) * p_hy

    sc = (np.sqrt(np.mean(np.abs(p_ac) ** 2))
          / np.sqrt(np.mean(np.abs(p_hy) ** 2))) * 10 ** (args.gap / 20)
    p_hy, vn_hy = sc * p_hy, sc * vn_hy
    phat, vnhat = p_ac + p_hy, vn_ac + vn_hy

    # --- (1) sigma power localisation (chunked evaluator for the O(N^2) sum) --
    sig, P_sig = gpu.sigma_density(cen, nrm, area, phat, vnhat,
                                   omega, rho0, c0, chunk=512, xp=np)
    src = np.abs(x - args.xs) < 0.08              # monopole region
    loud = np.abs(x) < 0.08                       # wavepacket region
    raw2 = np.abs(phat) ** 2
    f_sig_src = float(np.sum(area[src] * sig[src]) / P_sig)
    f_sig_loud = float(np.sum(area[loud] * sig[loud]) / P_sig)
    f_raw_src = float(np.sum(area[src] * raw2[src]) / np.sum(area * raw2))
    f_raw_loud = float(np.sum(area[loud] * raw2[loud]) / np.sum(area * raw2))

    # --- (2,3) far-field directivity and radiated level ----------------------
    P_all, xh, F_all = localization.farfield_power(cen, nrm, area, phat, vnhat,
                                                   omega, rho0, c0)
    P_ac, _, F_ac = localization.farfield_power(cen, nrm, area, p_ac, vn_ac,
                                                omega, rho0, c0)
    P_hy, _, F_hy = localization.farfield_power(cen, nrm, area, p_hy, vn_hy,
                                                omega, rho0, c0)
    upper = xh[:, 2] > 0
    d_match = float(np.linalg.norm(np.abs(F_all[upper]) - np.abs(F_ac[upper]))
                    / np.linalg.norm(np.abs(F_ac[upper])))
    hy_far = 10 * np.log10(P_hy / P_ac)

    print(f"flat plate N={len(cen)}, f={args.freq:.0f} Hz, kc/k={kc/k:.1f}, "
          f"wavepacket surface pressure +{args.gap:.0f} dB over monopole")
    print("--- (1) radiated-power localisation (sigma) ---")
    print(f"  monopole region : sigma-power {100*f_sig_src:5.1f} %   "
          f"raw |p|^2 {100*f_raw_src:5.1f} %")
    print(f"  wavepacket region: sigma-power {100*f_sig_loud:5.1f} %   "
          f"raw |p|^2 {100*f_raw_loud:5.1f} %")
    print("--- (2) far-field directivity (upper half-space) ---")
    print(f"  |F_full| vs |F_monopole| rel. L2 difference: {d_match:.3f}")
    print("--- (3) wavepacket radiation efficiency ---")
    print(f"  wavepacket is +{args.gap:.0f} dB on the surface but "
          f"{hy_far:+.1f} dB in the radiated far field")

    ok = (f_sig_src > 0.6 and f_sig_src > 3 * f_sig_loud
          and d_match < 0.2 and hy_far < -8.0)
    print("\nLOCALISATION FAR-FIELD CHECK " + ("PASSED" if ok else "FAILED"))

    _render(args, cen, sig, raw2, s, xh, F_all, F_ac, F_hy)
    sys.exit(0 if ok else 1)


def _render(args, cen, sig, raw2, s, xh, F_all, F_ac, F_hy):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plotstyle.use_paper()
    fig = plt.figure(figsize=(9.2, 3.5), layout="constrained")

    axL = fig.add_subplot(1, 2, 1)
    xa = cen[:, 0]
    nb = 50
    edges = np.linspace(xa.min(), xa.max(), nb + 1)
    ctr = 0.5 * (edges[:-1] + edges[1:])
    idx = np.clip(np.digitize(xa, edges) - 1, 0, nb - 1)
    br = np.array([raw2[idx == i].sum() for i in range(nb)])
    ss = np.maximum(sig, 0.0)
    bs = np.array([ss[idx == i].sum() for i in range(nb)])
    axL.plot(ctr, br / br.max(), "-", color="#d62728", label=r"raw $|\hat p|^2$")
    axL.plot(ctr, bs / bs.max(), "-", color="#1f77b4",
             label=r"$\sigma$ power density")
    axL.axvline(s[0], color="k", ls="--", lw=1)
    axL.set_xlabel(r"$x$ [m] (dashed: monopole)")
    axL.set_ylabel("normalised chordwise distribution")
    axL.set_title("Where the pressure is vs where the power radiates")
    axL.legend(frameon=False, fontsize=8)

    axR = fig.add_subplot(1, 2, 2, projection="polar")
    th = np.arctan2(xh[:, 0], xh[:, 2])
    pl = (np.abs(xh[:, 1]) < 0.2) & (xh[:, 2] > -1e-9)
    o = np.argsort(th[pl])
    ref = np.abs(F_ac[pl]).max()
    axR.plot(th[pl][o], np.abs(F_all[pl][o]) / ref, color="#1f77b4",
             label="full surface")
    axR.plot(th[pl][o], np.abs(F_ac[pl][o]) / ref, "--", color="k",
             label="monopole only")
    axR.plot(th[pl][o], np.abs(F_hy[pl][o]) / ref, color="#d62728",
             label="wavepacket only")
    axR.set_thetamin(-90); axR.set_thetamax(90)
    axR.set_title("Far-field directivity (upper half-space)", pad=10, fontsize=9)
    axR.legend(frameon=False, fontsize=7, loc="lower center",
               bbox_to_anchor=(0.5, -0.33), ncol=3)
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
