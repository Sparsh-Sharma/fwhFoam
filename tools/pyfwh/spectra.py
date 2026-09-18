"""Spectral post-processing helpers (PSD, SPL, OASPL)."""

from __future__ import annotations

import numpy as np

PREF = 2e-5  # reference acoustic pressure [Pa]


def psd(p, dt, window="hann", detrend=True):
    """One-sided power spectral density of a real pressure signal.

    Returns (freqs, Gxx) with Gxx in Pa^2/Hz.
    """
    p = np.asarray(p, dtype=np.float64)
    n = p.size
    if detrend:
        p = p - p.mean()
    if window == "hann":
        w = np.hanning(n)
    elif window in (None, "boxcar", "none"):
        w = np.ones(n)
    else:
        raise ValueError(f"unknown window {window}")
    wp = p * w
    P = np.fft.rfft(wp)
    freqs = np.fft.rfftfreq(n, dt)
    scale = 2.0 * dt / (np.sum(w**2))
    Gxx = scale * np.abs(P) ** 2
    Gxx[0] /= 2.0
    if n % 2 == 0:
        Gxx[-1] /= 2.0
    return freqs, Gxx


def spl(p, dt, **kw):
    """Sound pressure level spectrum [dB re 20 uPa] with 1 Hz bandwidth."""
    f, g = psd(p, dt, **kw)
    return f, 10.0 * np.log10(np.maximum(g, 1e-300) / PREF**2)


def oaspl(p):
    """Overall SPL [dB] from the rms of a pressure time series."""
    p = np.asarray(p, dtype=np.float64)
    rms = np.sqrt(np.mean((p - p.mean()) ** 2))
    return 20.0 * np.log10(max(rms, 1e-300) / PREF)


def amplitude_at(p, dt, f0):
    """Single-sided amplitude of the spectral line nearest f0 [Pa]."""
    n = p.size
    P = np.fft.rfft((p - p.mean()) * np.hanning(n))
    freqs = np.fft.rfftfreq(n, dt)
    k = int(np.argmin(np.abs(freqs - f0)))
    # Hann coherent gain = 0.5
    return 2.0 * np.abs(P[k]) / (n * 0.5)
