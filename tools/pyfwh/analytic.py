"""Analytic acoustic reference fields for verifying fwhFoam.

All fields are exact solutions of the wave / convected-wave equation, so
they simultaneously provide

  * the source data (p', u, rho) to feed onto an FW-H surface, and
  * the exact far-field observer pressure to compare against.

Because the FW-H surface encloses only these analytic sources and the
data are placed on the surface exactly, a correct Farassat-1A
implementation must reproduce the analytic observer signal to
discretisation accuracy (surface resolution and time step).

Conventions: SI units, medium at rest unless a convection velocity U0 is
given, ambient density rho0 and sound speed c0. Pressure p' is the
acoustic perturbation; density perturbation rho' = p'/c0^2; particle
velocity u is irrotational (u = grad(phi) with p' = -rho0 d phi/dt).
"""

from __future__ import annotations

import numpy as np


class MonopoleField:
    r"""Stationary point monopole of angular frequency omega in a medium
    at rest.

    Velocity potential (outgoing):
        phi(r,t) = -(A / (4 pi r)) exp(i (omega t - k r))
    Acoustic pressure:
        p' = -rho0 d phi/dt = i omega rho0 A/(4 pi r) exp(i(omega t - kr))
    Particle velocity:
        u = grad phi = A/(4 pi) (1/r^2 + i k/r) rHat exp(i(omega t - kr))

    The physical field is the real part. ``A`` is the source strength
    (volume acceleration amplitude, units m^3/s^2 convention absorbed).
    """

    #- Uniform mean-flow velocity of the medium (zero: medium at rest)
    U_mean = np.zeros(3)

    def __init__(self, amplitude, omega, c0=340.29, rho0=1.225, x0=(0, 0, 0)):
        self.A = float(amplitude)
        self.omega = float(omega)
        self.c0 = float(c0)
        self.rho0 = float(rho0)
        self.k = self.omega / self.c0
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.U_mean = np.zeros(3)

    def _rvec(self, x):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        d = x - self.x0
        r = np.linalg.norm(d, axis=1)
        rhat = d / r[:, None]
        return d, r, rhat

    def pressure(self, x, t):
        """Complex acoustic pressure p'(x,t) (take .real for physical)."""
        _, r, _ = self._rvec(x)
        phase = np.exp(1j * (self.omega * t - self.k * r))
        return (1j * self.omega * self.rho0 * self.A / (4 * np.pi * r)) * phase

    def velocity(self, x, t):
        """Complex particle velocity u(x,t), shape (...,3)."""
        _, r, rhat = self._rvec(x)
        phase = np.exp(1j * (self.omega * t - self.k * r))
        radial = (self.A / (4 * np.pi)) * (1.0 / r**2 + 1j * self.k / r) * phase
        return radial[:, None] * rhat

    def density(self, x, t):
        return self.pressure(x, t) / self.c0**2


class DipoleField:
    r"""Point dipole = derivative of a monopole along axis ``d`` (unit).

    Constructed from a monopole potential differentiated in space:
        phi_dip = (d . grad) phi_mono / (i k)   [normalised]
    Here we build it directly from the monopole via finite directional
    derivative in closed form using the analytic gradient, so the result
    is exact. The dipole models a compact oscillating force (loading
    noise), giving a figure-of-eight directivity ~ cos(theta).
    """

    def __init__(self, amplitude, omega, axis=(0, 1, 0),
                 c0=340.29, rho0=1.225, x0=(0, 0, 0)):
        self.mono = MonopoleField(amplitude, omega, c0, rho0, x0)
        ax = np.asarray(axis, dtype=np.float64)
        self.axis = ax / np.linalg.norm(ax)
        self.c0 = c0
        self.rho0 = rho0
        self.omega = omega
        self.k = self.mono.k
        self.x0 = self.mono.x0

    def pressure(self, x, t):
        # p'_dip = d . grad p'_mono
        # grad of g(r)=exp(-ikr)/r term; do it analytically:
        d, r, rhat = self.mono._rvec(x)
        cos = rhat @ self.axis
        phase = np.exp(1j * (self.omega * t - self.k * r))
        pref = 1j * self.omega * self.rho0 * self.mono.A / (4 * np.pi)
        # d/dn [phase/r] = (-1/r^2 - i k/r) cos * phase
        return pref * (-1.0 / r**2 - 1j * self.k / r) * cos * phase

    def velocity(self, x, t):
        # Numerically exact directional derivative of monopole velocity
        # along axis; use analytic second-derivative-free finite diff on
        # the closed-form monopole velocity (small eps, double precision).
        eps = 1e-6
        xp = np.atleast_2d(x) + eps * self.axis
        xm = np.atleast_2d(x) - eps * self.axis
        return (self.mono.velocity(xp, t) - self.mono.velocity(xm, t)) / (2 * eps)

    def density(self, x, t):
        return self.pressure(x, t) / self.c0**2


class ConvectedMonopoleField:
    r"""Stationary monopole in a uniform mean flow U0 (wind-tunnel frame).

    The field is generated from a single velocity potential phi that is
    the exact time-harmonic Green's function of the convected wave
    equation for a source at ``x0`` in a uniform flow U0 = M c0:

        phi(x,t) = -(A/(4 pi sigma)) exp(i(omega t - k R*)),
        sigma    = sqrt(xs^2 + beta^2 rperp^2),   beta^2 = 1 - M^2,
        R*       = (sigma - M xs)/beta^2,

    where xs is the streamwise coordinate (along U0) and rperp the
    transverse distance. All observables are derived from this single
    phi, so they are mutually consistent to machine precision:

        u   = grad phi                         (particle velocity)
        p'  = -rho0 (d/dt + U0 . grad) phi      (acoustic pressure)
        rho'= p'/c0^2

    Consistency (u, p', rho from one phi that solves the convected wave
    equation) is exactly what a permeable FW-H surface requires, so a
    correct convective implementation reproduces p' at the observer.

    Spatial derivatives are taken by central differences on the smooth
    phi with eps scaled to the wavelength (double precision => ~1e-9
    relative error), which is far below the surface-discretisation error.
    """

    def __init__(self, amplitude, omega, U0=(0, 0, 0),
                 c0=340.29, rho0=1.225, x0=(0, 0, 0)):
        self.A = float(amplitude)
        self.omega = float(omega)
        self.c0 = float(c0)
        self.rho0 = float(rho0)
        self.k = self.omega / self.c0
        self.U0 = np.asarray(U0, dtype=np.float64)
        self.M = self.U0 / self.c0
        self.Mmag = float(np.linalg.norm(self.M))
        self.x0 = np.asarray(x0, dtype=np.float64)
        self.ehat = (self.M / self.Mmag if self.Mmag > 0
                     else np.array([1.0, 0.0, 0.0]))
        self._eps = 1e-6 * (self.c0 / self.omega)  # ~1e-6 wavelength/2pi
        # Total fluid velocity on a surface in the tunnel frame is the
        # mean flow plus the acoustic perturbation
        self.U_mean = self.U0.copy()

    def phi(self, x, t):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        d = x - self.x0
        beta2 = 1.0 - self.Mmag**2
        xs = d @ self.ehat
        rperp2 = np.maximum(np.sum(d**2, axis=1) - xs**2, 0.0)
        sigma = np.sqrt(xs**2 + beta2 * rperp2)
        Rstar = (sigma - self.Mmag * xs) / beta2
        phase = np.exp(1j * (self.omega * t - self.k * Rstar))
        return -(self.A / (4 * np.pi * sigma)) * phase

    def _grad_phi(self, x, t):
        x = np.atleast_2d(np.asarray(x, dtype=np.float64))
        eps = self._eps
        g = np.zeros((x.shape[0], 3), dtype=complex)
        for j in range(3):
            e = np.zeros(3); e[j] = 1.0
            g[:, j] = (self.phi(x + eps * e, t)
                       - self.phi(x - eps * e, t)) / (2 * eps)
        return g

    def velocity(self, x, t):
        return self._grad_phi(x, t)

    def pressure(self, x, t):
        # p' = -rho0 (d/dt + U0 . grad) phi
        dphidt = 1j * self.omega * self.phi(x, t)
        conv = self._grad_phi(x, t) @ self.U0
        return -self.rho0 * (dphidt + conv)

    def density(self, x, t):
        return self.pressure(x, t) / self.c0**2
