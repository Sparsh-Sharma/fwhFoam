"""Unit tests for pyfwh (run with pytest)."""
import numpy as np
import pytest

from pyfwh import io, analytic, geometry, spectra, localization


def test_fwhdata_roundtrip(tmp_path):
    cen, nrm, area, _, _ = geometry.icosphere(radius=1.0, subdivisions=1)
    n = cen.shape[0]
    times = np.linspace(0, 1e-3, 5)
    p = np.random.default_rng(0).standard_normal((5, n))
    u = np.random.default_rng(1).standard_normal((5, n, 3))
    rho = 1.225 + 0.01 * np.random.default_rng(2).standard_normal((5, n))

    f = tmp_path / "rt.fwh"
    io.FWHData(cen, nrm, area, times, p, u, rho).write(f)
    d = io.read(f)

    assert d.n_faces == n
    np.testing.assert_allclose(d.centres, cen)
    np.testing.assert_allclose(d.normals, nrm)
    np.testing.assert_allclose(d.areas, area)
    np.testing.assert_allclose(d.times, times)
    np.testing.assert_allclose(d.p, p)
    np.testing.assert_allclose(d.u, u)
    np.testing.assert_allclose(d.rho, rho)


def test_icosphere_closed_surface():
    # A closed surface has sum(area * normal) ~ 0 and total area ~ 4 pi R^2
    for sub in (0, 1, 2, 3):
        cen, nrm, area, _, _ = geometry.icosphere(radius=2.0, subdivisions=sub)
        assert np.allclose(np.sum(area[:, None] * nrm, axis=0), 0, atol=1e-9)
        # converges to 4 pi R^2 = 16 pi from below
        assert np.sum(area) <= 4 * np.pi * 4 + 1e-9
        if sub >= 2:
            assert np.sum(area) > 0.98 * 4 * np.pi * 4


def test_monopole_farfield_decay():
    # |p| ~ 1/r in the far field
    field = analytic.MonopoleField(1e-3, 2 * np.pi * 340.29)
    r1, r2 = 10.0, 20.0
    p1 = abs(field.pressure(np.array([[0, r1, 0]]), 0.0)[0])
    p2 = abs(field.pressure(np.array([[0, r2, 0]]), 0.0)[0])
    assert np.isclose(p1 / p2, r2 / r1, rtol=1e-6)


def test_dipole_null_axis():
    # A y-axis dipole is silent on the x-axis
    field = analytic.DipoleField(1e-3, 2 * np.pi * 340.29, axis=(0, 1, 0))
    p_axis = abs(field.pressure(np.array([[10.0, 0, 0]]), 0.0)[0])
    p_peak = abs(field.pressure(np.array([[0, 10.0, 0]]), 0.0)[0])
    assert p_axis < 1e-6 * p_peak


def test_psd_parseval():
    # Parseval: integral of PSD ~ variance of the signal
    rng = np.random.default_rng(3)
    n, dt = 4096, 1e-3
    x = rng.standard_normal(n)
    f, g = spectra.psd(x, dt, window=None)
    trapezoid = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    integral = trapezoid(g, f)
    assert np.isclose(integral, np.var(x), rtol=0.05)


def test_convected_reduces_to_monopole():
    # At U0 = 0 the convected field equals the plain monopole
    om = 2 * np.pi * 200.0
    m = analytic.MonopoleField(1e-3, om)
    c = analytic.ConvectedMonopoleField(1e-3, om, U0=(0, 0, 0))
    x = np.array([[3.0, 4.0, 0.0]])
    assert np.isclose(m.pressure(x, 0.0)[0], c.pressure(x, 0.0)[0], rtol=1e-4)


def test_sigma_power_identity_monopole():
    # int sigma dS == far-field power == analytic power, for a monopole
    om = 2 * np.pi * 200.0
    c0, rho0 = 340.29, 1.225
    field = analytic.MonopoleField(1e-3, om, c0=c0, rho0=rho0)
    cen, nrm, area, _, _ = geometry.icosphere(radius=1.0, subdivisions=3)
    phat = field.pressure(cen, 0.0)
    vnhat = np.einsum("ni,ni->n", field.velocity(cen, 0.0), nrm)

    P_true = localization.pointwise_power(nrm, area, phat, vnhat)
    sigma, P_sig = localization.sigma_density(cen, nrm, area, phat, vnhat,
                                              om, rho0, c0)
    P_far, _, _ = localization.farfield_power(cen, nrm, area, phat, vnhat,
                                              om, rho0, c0)
    assert abs(P_sig - P_true) / abs(P_true) < 0.01
    assert abs(P_far - P_true) / abs(P_true) < 0.01
    # a monopole radiates isotropically: sigma is (nearly) uniform
    assert np.std(sigma) / np.mean(sigma) < 0.02


def test_sigma_rigid_limit_reduces_to_pressure_block():
    # With v_n = 0 (rigid wall) sigma uses only the pressure-pressure block
    om = 2 * np.pi * 150.0
    c0, rho0 = 340.29, 1.225
    cen, nrm, area, _, _ = geometry.icosphere(radius=1.0, subdivisions=2)
    rng = np.random.default_rng(0)
    phat = rng.standard_normal(len(cen)) + 1j * rng.standard_normal(len(cen))
    zero = np.zeros(len(cen), complex)
    s_full, P_full = localization.sigma_density(cen, nrm, area, phat, zero,
                                                om, rho0, c0)
    # rigid density must be finite and integrate to the same via far field
    P_far, _, _ = localization.farfield_power(cen, nrm, area, phat, zero,
                                              om, rho0, c0)
    assert np.isfinite(P_full)
    assert abs(P_full - P_far) / max(abs(P_far), 1e-30) < 0.05


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
