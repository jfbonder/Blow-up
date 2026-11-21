import numpy as np

from blowup.parabolic import (HeatFlux1D, classify, critical_lambda, flux_power,
                              flux_power_minus_linear, unstable_eigenvalue_continuous)


def test_rhs_is_lumped_fem():
    F, dF = flux_power(2.0)
    P = HeatFlux1D(8, F, dF)
    u = np.random.default_rng(0).random(9)
    B = np.zeros(9)
    B[-1] = F(u[-1])
    expected = (-(P.stiffness @ u) + B) / P.mass
    np.testing.assert_allclose(P.rhs(0, u), expected, rtol=1e-12)


def test_jacobian_matches_finite_differences():
    F, dF = flux_power_minus_linear(3.0)
    P = HeatFlux1D(6, F, dF, reaction=lambda u: -u**2, dreaction=lambda u: -2 * u)
    u = 1 + np.random.default_rng(1).random(7)
    J = P.jac_dense(u)
    eps = 1e-6
    Jfd = np.column_stack([(P.rhs(0, u + eps * e) - P.rhs(0, u - eps * e)) / (2 * eps) for e in np.eye(7)])
    np.testing.assert_allclose(J, Jfd, rtol=1e-6, atol=1e-4)


def test_constant_steady_states_and_classification():
    F, dF = flux_power_minus_linear(2.0)
    P = HeatFlux1D(10, F, dF)
    np.testing.assert_allclose(P.rhs(0, np.ones(11)), 0, atol=1e-12)
    assert classify(P, 0.9 * np.ones(11))[0] == "decay"
    assert classify(P, 1.1 * np.ones(11))[0] == "blowup"


def test_critical_lambda_table1_large_N():
    F, dF = flux_power_minus_linear(2.0)
    P = HeatFlux1D(50, F, dF)
    lam = critical_lambda(P, 0.5 * np.cos(2 * np.pi * P.x) + 1, 0.9, 1.05, tol=1e-6)
    assert abs(lam - 0.9770) < 5e-5  # Tabla 1 del paper (N = 50)


def test_unstable_eigenvalue_converges():
    F, dF = flux_power_minus_linear(2.0)
    P = HeatFlux1D(100, F, dF)
    mu = np.linalg.eigvals(P.jac_dense(np.ones(101))).real
    assert (mu > 0).sum() == 1
    assert abs(mu.max() - unstable_eigenvalue_continuous(2.0)) < 1e-3


def test_dirichlet_steady_state_and_threshold():
    from blowup.parabolic import ReactionDirichlet1D, power_steady_max, power_steady_profile

    p = 2.0
    assert abs(power_steady_max(p) - 11.79669) < 1e-4
    P = ReactionDirichlet1D(50, lambda u: np.abs(u) ** p, lambda u: p * np.abs(u) ** (p - 1) * np.sign(u))
    V = P.steady_state(power_steady_profile(p, P.xi))
    np.testing.assert_allclose(P.rhs(0, V), 0, atol=1e-8)
    assert abs(V.max() - power_steady_max(p)) < 5e-3
    mu = np.linalg.eigvalsh(P.jac_dense(V))
    assert (mu > 0).sum() == 1  # variedad estable de codimensión uno
    u0 = np.sin(np.pi * P.xi)
    lam = critical_lambda(P, u0, 10.0, 13.0, tol=1e-6, equilibrium=V)
    assert abs(lam - 11.51048) < 1e-4


def test_solve_to_blowup_power_rate_and_nodes():
    from blowup.parabolic import solve_to_blowup

    p, N = 2.0, 10
    P = HeatFlux1D(N, *flux_power(p))
    sol = solve_to_blowup(P, np.ones(N + 1), lambda m: m ** (p - 1), u_stop=1e100)
    R = sol.remaining
    # u_N ~ C (T_h - t)^{-1}: pendiente -1 en log-log.
    i0 = int(np.searchsorted(-R, -R[-1] * 1e30))
    slope = np.polyfit(np.log(R[i0:]), np.log(sol.U[i0:, N]), 1)[0]
    assert abs(slope + 1) < 1e-3
    # Con u_x(1)=u^2 explotan exactamente x_N y x_{N-1} (este último como -ln(T_h - t)).
    j = int(np.searchsorted(-R, -R[-1] * 1e10))
    growing = sol.U[-1] - sol.U[j] > 1e-6 * (1 + np.abs(sol.U[-1]))
    assert growing.sum() == 2 and growing[-1] and growing[-2]
    # Remaining time consistent with the physical time.
    np.testing.assert_allclose(sol.t[:10] + R[:10], sol.T, rtol=1e-8)
