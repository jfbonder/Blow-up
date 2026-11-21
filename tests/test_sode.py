import math

import numpy as np
from scipy.integrate import quad

from blowup.sode import (
    Z0_PAPER,
    b_paper,
    deterministic_tail,
    sigma_paper,
    simulate_path,
    stopping_times,
)


def zero(x):
    return 0.0 * np.asarray(x, dtype=float)


def test_drift_increment_is_exactly_h():
    # Sin ruido: X_k = z + h k exactamente y t_k = sum h / b(X_j).
    h, z, M = 0.5, 10.0, 100.0
    p = simulate_path(b_paper, zero, z, h, M, np.random.default_rng(0))
    k = np.arange(len(p.X))
    np.testing.assert_allclose(p.X, z + h * k)
    np.testing.assert_allclose(np.diff(p.t), h / b_paper(p.X[:-1]))
    assert p.exploded and p.X[-1] >= M and p.X[-2] < M


def test_deterministic_limit_is_ode_time():
    # Sin ruido y h -> 0, R_h^M -> int_z^M ds / b(s).
    z, M = 10.0, 1e3
    exact, _ = quad(lambda s: 1 / b_paper(s), z, M)
    R = stopping_times(b_paper, zero, z, 1e-3, M, 1, np.random.default_rng(0))[0]
    assert abs(R - exact) < 1e-3


def test_reproducible_with_seed():
    a = stopping_times(b_paper, sigma_paper, Z0_PAPER, 1.0, 1e3, 50, np.random.default_rng(7))
    b = stopping_times(b_paper, sigma_paper, Z0_PAPER, 1.0, 1e3, 50, np.random.default_rng(7))
    np.testing.assert_array_equal(a, b)


def test_path_and_vectorized_agree_in_distribution():
    h, M, n = 1.0, 1e3, 400
    rng = np.random.default_rng(1)
    R_path = np.array(
        [simulate_path(b_paper, sigma_paper, Z0_PAPER, h, M, rng).stopping_time for _ in range(n)]
    )
    R_vec = stopping_times(b_paper, sigma_paper, Z0_PAPER, h, M, n, np.random.default_rng(2))
    se = math.sqrt(R_path.var() / n + R_vec.var() / n)
    assert abs(R_path.mean() - R_vec.mean()) < 4 * se


def test_max_steps_returns_nan():
    R = stopping_times(b_paper, sigma_paper, Z0_PAPER, 1.0, 1e6, 5, np.random.default_rng(0), max_steps=10)
    assert np.isnan(R).all()


def test_deterministic_tail_power_law():
    # Para b = s^1.1 la cola es 10 M^{-0.1}; con +0.1 debe estar muy cerca.
    M = 1e4
    assert abs(deterministic_tail(b_paper, M) - 10 * M**-0.1) < 1e-3
