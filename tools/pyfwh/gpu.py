r"""GPU-accelerated localization kernels (CuPy backend, NumPy fallback).

The two O(N^2)-per-frequency operations of the localization pipeline --- the
sigma surface sound-power density (:mod:`pyfwh.localization`) and the
diffraction-filter double-layer operator (:mod:`pyfwh.filter`) --- are
re-implemented here in a backend-agnostic form: every function takes an
``xp`` array module that is either ``numpy`` or ``cupy``, selected
automatically by :func:`get_xp`. On an NVIDIA GPU the same code runs
unmodified under CuPy.

Two changes relative to the reference implementations make the kernels
GPU-friendly (and also faster and lighter on the CPU):

* the angular kernels are algebraically contracted before evaluation,

      n_a . W(d) . n_b = 4pi [ (j1(z)/z) (n_a.n_b) - j2(z) (n_a.dh)(n_b.dh) ]
      n . V(d)         = 4pi i j1(z) (n.dh)

  so no (A,B,3,3) tensor is ever materialized;
* the spherical Bessel functions j0, j1, j2 are evaluated from their
  elementary closed forms (with series near z = 0), since SciPy's
  ``spherical_jn`` is CPU-only.

Numerical agreement with the reference implementations is machine-precision
and is asserted by ``tests/gpu/benchmark_gpu.py``, which also produces the
CPU-vs-GPU benchmark.
"""

from __future__ import annotations

import numpy as np

try:
    import cupy as _cp
    try:
        _cp.cuda.runtime.getDeviceCount()
        _HAVE_GPU = True
    except Exception:
        _cp = None
        _HAVE_GPU = False
except Exception:
    _cp = None
    _HAVE_GPU = False


def available():
    """True if CuPy is importable and a CUDA device is present."""
    return _HAVE_GPU


def get_xp(use_gpu=None):
    """Return the array module: cupy if requested/available, else numpy.

    ``use_gpu=None`` auto-selects (GPU when available); ``use_gpu=True``
    raises if no GPU is present.
    """
    if use_gpu is None:
        return _cp if _HAVE_GPU else np
    if use_gpu:
        if not _HAVE_GPU:
            raise RuntimeError("GPU requested but CuPy/CUDA is unavailable")
        return _cp
    return np


def asnumpy(x):
    """Copy an array to host memory (no-op for NumPy arrays)."""
    if _cp is not None and isinstance(x, _cp.ndarray):
        return _cp.asnumpy(x)
    return np.asarray(x)


def synchronize(xp):
    """Block until pending GPU work completes (no-op on NumPy)."""
    if _cp is not None and xp is _cp:
        _cp.cuda.Device().synchronize()


# --------------------------------------------------------------------------
# Elementary spherical Bessel functions (backend-agnostic)
# --------------------------------------------------------------------------

def _sph_j012(z, xp):
    """j0(z), j1(z)/z, j2(z) from elementary forms; Taylor series near 0.

    Returns (j0, j1z, j2) where j1z = j1(z)/z (finite at z = 0: 1/3).
    """
    z = xp.asarray(z)
    small = z < 1e-3
    zs = xp.where(small, 1.0, z)          # safe divisor
    s, c = xp.sin(zs), xp.cos(zs)
    z2 = zs * zs

    j0 = s / zs
    j1 = s / z2 - c / zs
    j1z = j1 / zs
    j2 = (3.0 / z2 - 1.0) * s / zs - 3.0 * c / z2

    t2 = z * z                             # true z^2 for the series
    j0 = xp.where(small, 1.0 - t2 / 6.0, j0)
    j1z = xp.where(small, 1.0 / 3.0 - t2 / 30.0, j1z)
    j2 = xp.where(small, t2 / 15.0, j2)
    return j0, j1z, j2


# --------------------------------------------------------------------------
# sigma surface sound-power density
# --------------------------------------------------------------------------

def sigma_density(pts, n, area, phat, vnhat, omega, rho, c,
                  chunk=1024, xp=None):
    r"""Two-point surface power density sigma at one frequency (GPU-capable).

    Same contract as :func:`pyfwh.localization.sigma_density`; inputs may be
    NumPy or ``xp`` arrays, outputs are host (NumPy) arrays.
    """
    xp = xp if xp is not None else get_xp()
    pts = xp.asarray(pts, dtype=xp.float64)
    n = xp.asarray(n, dtype=xp.float64)
    area = xp.asarray(area, dtype=xp.float64)
    phi = xp.asarray(phat, dtype=xp.complex128)
    psi = -1j * omega * rho * xp.asarray(vnhat, dtype=xp.complex128)
    k = omega / c
    N = int(pts.shape[0])
    pref = 4.0 * np.pi / (32.0 * np.pi**2 * rho * c)   # 4pi from the kernels

    aphi = area * xp.conj(phi)
    apsi = area * xp.conj(psi)
    sigma = xp.zeros(N, dtype=xp.float64)
    chunk = N if not chunk else int(chunk)

    for a0 in range(0, N, chunk):
        a1 = min(a0 + chunk, N)
        d = pts[a0:a1, None, :] - pts[None, :, :]       # (A,B,3)
        dn = xp.sqrt(xp.sum(d * d, axis=-1))            # (A,B)
        j0, j1z, j2 = _sph_j012(k * dn, xp)

        safe = xp.where(dn > 0, dn, 1.0)
        nad = xp.sum(n[a0:a1, None, :] * d, axis=-1) / safe   # n_a . dh
        nbd = xp.sum(n[None, :, :] * d, axis=-1) / safe       # n_b . dh
        nanb = n[a0:a1] @ n.T                                  # (A,B)

        nWn = j1z * nanb - j2 * nad * nbd
        j1 = j1z * (k * dn)
        # sum_b area_b * [ k^2 nWn phi_a phi_b^* - i k (i j1 nad) phi_a psi_b^*
        #                 + i k (i j1 nbd) psi_a phi_b^* + j0 psi_a psi_b^* ]
        acc = (
            (k * k) * phi[a0:a1] * (nWn @ aphi)
            + k * phi[a0:a1] * ((j1 * nad) @ apsi)
            - k * psi[a0:a1] * ((j1 * nbd) @ aphi)
            + psi[a0:a1] * (j0 @ apsi)
        )
        sigma[a0:a1] = pref * xp.real(acc)

    P = float(asnumpy(xp.sum(area * sigma)))
    return asnumpy(sigma), P


# --------------------------------------------------------------------------
# Diffraction filter (double-layer operator)
# --------------------------------------------------------------------------

def acoustic_surface_pressure(p_hat, centroids, normals, areas, k,
                              chunk=1024, xp=None):
    r"""Acoustic surface pressure ``p_S = D @ p_hat`` (GPU-capable).

    Same contract as :func:`pyfwh.filter.acoustic_surface_pressure`, but the
    operator rows are formed in chunks and applied on the fly, so the dense
    (N,N) matrix is never stored. Returns a host (NumPy) complex array.
    """
    xp = xp if xp is not None else get_xp()
    x = xp.asarray(centroids, dtype=xp.float64)
    n = xp.asarray(normals, dtype=xp.float64)
    a = xp.asarray(areas, dtype=xp.float64)
    p = xp.asarray(p_hat, dtype=xp.complex128)
    N = int(x.shape[0])
    chunk = N if not chunk else int(chunk)

    pS = xp.empty(N, dtype=xp.complex128)
    idx = xp.arange(N)

    for a0 in range(0, N, chunk):
        a1 = min(a0 + chunk, N)
        diff = x[a0:a1, None, :] - x[None, :, :]
        r = xp.sqrt(xp.sum(diff * diff, axis=-1))
        rows = idx[a0:a1] - a0
        r[rows, idx[a0:a1]] = 1.0                       # avoid /0 on the diag
        ern = xp.sum(diff * n[None, :, :], axis=-1) / r

        G0 = ern / (r * r) * a[None, :] / (2.0 * np.pi)  # static kernel rows
        G0[rows, idx[a0:a1]] = 0.0
        K = xp.exp(-1j * k * r) * (1j * k * r + 1.0) * G0
        K[rows, idx[a0:a1]] = 0.0
        diag = -1.0 - G0.sum(axis=1)                     # solid-angle rule
        pS[a0:a1] = K @ p + diag * p[a0:a1]

    return asnumpy(pS)


def _scatter_add(xp, a, idx, v):
    if xp is np:
        np.add.at(a, idx, v)
    else:
        import cupyx
        cupyx.scatter_add(a, idx, v)


def farassat_1a(data, observers, c0, rho0, U0=(0.0, 0.0, 0.0), xp=None):
    r"""GPU-capable Farassat formulation-1A solver (advanced time).

    A vectorized port of the C++ core (``fwhFormulation1A``) for **static**
    integration surfaces in a uniform mean flow: the Garrick-triangle
    emission delay, the mid-level central-difference source derivatives and
    the segment-wise linear scatter onto the uniform observer time grid are
    identical, so results match ``fwhSolve`` on the same FWH-DATA input.
    Faces x observers are evaluated in one array operation per source time
    level, which is what the GPU accelerates.

    Parameters
    ----------
    data : pyfwh.io.FWHData
        Static-surface data (motion records are not supported here; use the
        C++ ``fwhSolve`` for moving surfaces).
    observers : dict name -> (3,) observer position
    c0, rho0 : ambient sound speed and density
    U0 : uniform mean-flow velocity (wind-tunnel frame)
    xp : numpy or cupy module (default: :func:`get_xp`)

    Returns
    -------
    dict name -> (times, p) host arrays, trimmed to the valid window in
    which every face has begun and no face has stopped contributing.
    """
    if getattr(data, "has_motion", False):
        raise NotImplementedError(
            "farassat_1a (GPU) supports static surfaces only; "
            "use fwhSolve for moving-surface data")
    xp = xp if xp is not None else get_xp()

    times = np.asarray(data.times, float)
    dt = float(np.mean(np.diff(times)))
    T = times.size
    if T < 3:
        raise ValueError("need at least 3 time levels")

    y = xp.asarray(data.centres, dtype=xp.float64)        # (N,3)
    nf = xp.asarray(data.normals, dtype=xp.float64)
    dA = xp.asarray(data.areas, dtype=xp.float64)
    N = int(y.shape[0])
    names = list(observers.keys())
    xo = xp.asarray(np.array([observers[k] for k in names], float))  # (O,3)
    O = int(xo.shape[0])
    U0 = np.asarray(U0, float)
    U0x = xp.asarray(U0)

    # --- static per-pair geometry: Garrick delay, rHat, coefficients -----
    d = xo[:, None, :] - y[None, :, :]                    # (O,N,3)
    U0sq = float(U0 @ U0)
    if U0sq < 1e-300:
        magd = xp.sqrt(xp.sum(d * d, axis=-1))
        Tdel = magd / c0
        rHat = d / xp.maximum(magd, 1e-300)[..., None]
    else:
        a = c0 * c0 - U0sq
        dU = xp.sum(d * U0x, axis=-1)
        Tdel = (-dU + xp.sqrt(dU * dU + a * xp.sum(d * d, axis=-1))) / a
        rvec = d - U0x * Tdel[..., None]
        rHat = rvec / xp.maximum(
            xp.sqrt(xp.sum(rvec * rvec, axis=-1)), 1e-300)[..., None]
    r = c0 * Tdel                                          # (O,N)

    M = -U0x / c0                                          # static: vB = -U0
    Mr = xp.sum(rHat * M, axis=-1)                         # (O,N)
    omr = 1.0 - Mr
    ok = (r > 1e-300) & (omr >= 0.02)
    invR = xp.where(ok, 1.0 / xp.maximum(r, 1e-300), 0.0)
    invOmr2 = xp.where(ok, 1.0 / xp.maximum(omr, 1e-6)**2, 0.0)
    magSqrM = float(U0sq) / (c0 * c0)
    K = c0 * (Mr - magSqrM)
    A1 = invR * invOmr2
    A2K = K * invR * invR * invOmr2 / xp.maximum(omr, 1e-6)
    A3 = invR * invR * invOmr2
    w = dA / (4.0 * np.pi)                                 # (N,)

    # --- observer accumulator grid --------------------------------------
    Td_np = asnumpy(Tdel)
    ok_np = asnumpy(ok)
    Tmin = float(Td_np[ok_np].min())
    Tmax = float(Td_np[ok_np].max())
    nmin = int(np.floor((times[1] + Tmin) / dt))
    nmax = int(np.floor((times[T - 2] + Tmax) / dt)) + 2
    nbins = nmax - nmin + 1
    acc = xp.zeros(O * nbins, dtype=xp.float64)
    obase = xp.arange(O, dtype=xp.int64)[:, None] * nbins  # (O,1)

    tArr0 = Tdel                                           # + tau, per level
    LM_coef = M                                            # (3,)

    p_all = np.asarray(data.p, float)
    u_all = np.asarray(data.u, float)
    rho_all = (np.asarray(data.rho, float) if data.rho is not None
               else np.full((T, N), rho0))

    prevP = xp.zeros((O, N), dtype=xp.float64)
    first = True
    firstArrMax = None
    lastArrMin = None

    def level_UL(m):
        um = xp.asarray(u_all[m])
        rm = xp.asarray(rho_all[m])
        uB = um - U0x
        unRel = xp.sum(um * nf, axis=-1)
        U = -U0x + (rm / rho0)[:, None] * um
        L = xp.asarray(p_all[m])[:, None] * nf + rm[:, None] * uB * unRel[:, None]
        return U, L

    Uprev, Lprev = level_UL(0)
    Ucur, Lcur = level_UL(1)
    inv2dt = 1.0 / (2.0 * dt)

    for m in range(1, T - 1):
        Unext, Lnext = level_UL(m + 1)
        Udot = (Unext - Uprev) * inv2dt
        Ldot = (Lnext - Lprev) * inv2dt

        Un = xp.sum(Ucur * nf, axis=-1)                    # (N,)
        Undot = xp.sum(Udot * nf, axis=-1)
        Lr = xp.einsum("ni,oni->on", Lcur, rHat)
        Ldotr = xp.einsum("ni,oni->on", Ldot, rHat)
        LM = xp.sum(Lcur * LM_coef, axis=-1)               # (N,)

        pT = (w * rho0) * (Undot[None, :] * A1 + Un[None, :] * A2K)
        pL = w * (Ldotr * A1 / c0 + (Lr - LM[None, :]) * A3 + Lr * A2K / c0)
        p = xp.where(ok, pT + pL, 0.0)

        tArr = times[m] + tArr0                            # (O,N)
        if first:
            firstArrMax = float(asnumpy(xp.max(xp.where(ok, tArr, -np.inf))))
            first = False
        else:
            t1 = tArr
            t0 = tArr - dt
            nLo = xp.floor(t0 / dt).astype(xp.int64) + 1
            nHi = xp.floor(t1 / dt + 1e-9).astype(xp.int64)
            for j in range(2):
                n = nLo + j
                m_ok = ok & (n <= nHi) & (n >= nmin) & (n <= nmax)
                wseg = (n * dt - t0) / dt
                val = xp.where(m_ok, prevP + wseg * (p - prevP), 0.0)
                idx = xp.where(m_ok, obase + (n - nmin), 0)
                _scatter_add(xp, acc, idx.ravel(), val.ravel())
        prevP = p
        lastArrMin = float(asnumpy(xp.min(xp.where(ok, tArr, np.inf))))
        Uprev, Lprev, Ucur, Lcur = Ucur, Lcur, Unext, Lnext

    acc = asnumpy(acc).reshape(O, nbins)
    tgrid = (nmin + np.arange(nbins)) * dt
    sel = (tgrid >= firstArrMax - 1e-9 * dt) & (tgrid <= lastArrMin + 1e-9 * dt)
    return {name: (tgrid[sel], acc[i, sel]) for i, name in enumerate(names)}


def filter_spectra(freqs, phat, centroids, normals, areas, c0,
                   fmin=0.0, fmax=np.inf, chunk=1024, xp=None):
    r"""GPU-capable analogue of :func:`pyfwh.filter.filter_spectra`."""
    xp = xp if xp is not None else get_xp()
    freqs = np.asarray(freqs, float)
    sel = np.where((freqs >= fmin) & (freqs <= fmax) & (freqs > 0))[0]
    out = np.empty((sel.size, np.asarray(centroids).shape[0]), complex)
    cen = xp.asarray(centroids, dtype=xp.float64)
    nrm = xp.asarray(normals, dtype=xp.float64)
    ar = xp.asarray(areas, dtype=xp.float64)
    for i, j in enumerate(sel):
        k = 2 * np.pi * freqs[j] / c0
        out[i] = acoustic_surface_pressure(phat[j], cen, nrm, ar, k,
                                           chunk=chunk, xp=xp)
    return sel, out
