import numpy as np

from blowup.adaptive import integrate, psi_power, psi_rhs, rescaled
from blowup.parabolic import HeatFlux1D, flux_power


class Scalar:
    def rhs(self, t, u):
        return u**2

    def jac(self, t, u):
        return np.array([[2 * u[0]]])


def test_explicit_euler_blowup_time_is_one_plus_tau():
    # u' = u^2, u(0) = 1, tau_n = tau / u_n: u_{n+1} = u_n + tau u_n y T_tau = 1 + tau.
    for tau in [0.1, 0.01]:
        r = integrate(Scalar(), np.array([1.0]), tau, psi_power(2.0), method="explicit", u_stop=1e100)
        assert r.exploded and abs(r.T - (1 + tau)) < 1e-9


def test_fixed_step_has_no_finite_blowup_time():
    r = integrate(Scalar(), np.array([1.0]), 0.05, None, method="explicit", u_stop=1e100)
    # Con paso fijo t_n = n tau: cuando u_n llega al umbral ya pasó el verdadero T = 1.
    assert r.t[-1] > 1.5


def test_pde_imex_and_implicit_converge_to_reference():
    N = 10
    P = HeatFlux1D(N, *flux_power(2.0))
    u0 = np.ones(N + 1)
    ref = rescaled(P, u0, psi_power(2.0), u_stop=1e100).T
    psi = psi_rhs(P)
    e1 = abs(integrate(P, u0, 0.02, psi, method="imex").T - ref)
    e2 = abs(integrate(P, u0, 0.01, psi, method="imex").T - ref)
    assert 1.6 < e1 / e2 < 2.4  # orden ~1
    ei = abs(integrate(P, u0, 0.01, psi, method="implicit").T - ref)
    assert ei < 5e-3


def test_remaining_time_consistent():
    P = HeatFlux1D(10, *flux_power(2.0))
    r = integrate(P, np.ones(11), 0.02, psi_rhs(P), method="imex", u_stop=1e60)
    np.testing.assert_allclose(r.t[:5] + r.remaining[:5], r.T, rtol=1e-10)
    assert r.remaining[-1] > 0 and r.remaining[-1] < 1e-50
