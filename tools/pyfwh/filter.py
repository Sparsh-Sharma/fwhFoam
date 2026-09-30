r"""Acoustic diffraction filter and on-surface source localization.

This module implements the diffraction filter of Delfs & Ruck --- the
acoustic double-layer boundary operator --- which converts a (hydrodynamic)
wall-pressure field on a closed surface into the *acoustic surface pressure*
``p_S``. Where the raw wall pressure ``p`` is dominated by non-radiating
hydrodynamic content, ``p_S`` is sharply localized on the parts of the
surface that actually radiate, and is the quantity mapped to produce
on-surface source-localization images.

Per angular frequency ``omega`` (wavenumber ``k = omega/c0``) the filter is

    p_S(x) = (1/2pi) PV int_S K(x, xi) p(xi) dS(xi),
    K(x, xi) = exp(-ikr)(ikr + 1) (e_r . n(xi)) / r^2,   e_r = (x - xi)/r,

discretized on the face centroids as a dense operator ``D`` (``p_S = D @ p``).
The weakly singular self term is absorbed by the discrete solid-angle
(row-sum) rule, calibrated so that the operator's sphere eigenvalues are
``g_l = -2i x^2 j_l'(x) h_l^(2)(x) - 1`` (Delfs & Ruck); the incompressible
limit gives ``g_l -> -1/(2l+1)``.

The construction needs only face centroids, outward unit normals and areas
--- the same geometry fwhFoam's surface data already carry --- so it applies
to any closed triangulated surface without connectivity information.

References
----------
J. W. Delfs, B. Ruck, A quantity to identify turbulence related sound
generation on surfaces, J. Sound Vib. 586 (2024) 118490.
"""

from __future__ import annotations

import numpy as np


def double_layer_matrix(centroids, normals, areas, k):
    r"""Discrete diffraction-filter operator ``D`` with ``p_S = D @ p``.

    Parameters
    ----------
    centroids : (N,3) face centres
    normals   : (N,3) outward unit normals
    areas     : (N,) face areas
    k         : wavenumber ``omega/c0``

    Returns
    -------
    D : (N,N) complex operator; ``p_S = D @ p_hat`` at this frequency.
    """
    x = np.asarray(centroids, float)
    n = np.asarray(normals, float)
    a = np.asarray(areas, float)
    N = x.shape[0]

    diff = x[:, None, :] - x[None, :, :]          # r_ij = x_i - xi_j
    r = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(r, 1.0)                        # avoid /0; diag reset below
    er_dot_n = np.einsum("ijk,jk->ij", diff, n) / r

    K = np.exp(-1j * k * r) * (1j * k * r + 1.0) * er_dot_n / r**2
    D = (1.0 / (2.0 * np.pi)) * K * a[None, :]

    # static (k=0) kernel for the diagonal solid-angle calibration
    D0 = (1.0 / (2.0 * np.pi)) * (er_dot_n / r**2) * a[None, :]
    np.fill_diagonal(D, 0.0)
    np.fill_diagonal(D0, 0.0)
    D[np.diag_indices(N)] = -1.0 - D0.sum(axis=1)
    return D


def acoustic_surface_pressure(p_hat, centroids, normals, areas, k):
    r"""Acoustic surface pressure ``p_S = D[p_hat]`` at one frequency.

    ``p_hat`` is the complex surface-pressure spectrum at wavenumber ``k``.
    Returns the complex ``p_S`` on the faces --- the source-localization
    field. ``|p_S|`` is the map plotted for on-surface localization.
    """
    D = double_layer_matrix(centroids, normals, areas, k)
    return D @ np.asarray(p_hat, complex)


def sphere_eigenvalue(ell, kR):
    r"""Analytic diffraction-filter eigenvalue on a sphere of radius ``R``.

    ``g_l(kR) = -2i (kR)^2 j_l'(kR) h_l^(2)(kR) - 1`` (Delfs & Ruck spectral
    theory); used to verify the discrete operator. ``h_l^(2) = j_l - i y_l``.
    """
    from scipy.special import spherical_jn, spherical_yn
    x = float(kR)
    jl = spherical_jn(ell, x)
    jlp = spherical_jn(ell, x, derivative=True)
    yl = spherical_yn(ell, x)
    hl = jl - 1j * yl                              # outgoing h_l^(2)
    return -2j * x**2 * jlp * hl - 1.0


def filter_spectra(freqs, phat, centroids, normals, areas, c0,
                   fmin=0.0, fmax=np.inf):
    r"""Apply the filter across a band of frequencies.

    Parameters
    ----------
    freqs : (F,) frequencies [Hz]
    phat  : (F,N) complex surface-pressure spectra
    c0    : sound speed
    fmin, fmax : band limits [Hz]

    Returns
    -------
    sel   : indices of the selected lines
    pS    : (len(sel), N) complex acoustic surface pressure per line
    """
    sel = np.where((freqs >= fmin) & (freqs <= fmax) & (freqs > 0))[0]
    out = np.empty((sel.size, centroids.shape[0]), complex)
    for i, idx in enumerate(sel):
        k = 2 * np.pi * freqs[idx] / c0
        out[i] = acoustic_surface_pressure(phat[idx], centroids, normals,
                                           areas, k)
    return sel, out
