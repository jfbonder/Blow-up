import numpy as np

from blowup.spde import Scheme, explosion_times, simulate_path, stationary_max


def zero_f(u):
    return 0.0 * u


def test_stationary_max_matches_bvp():
    # Resolver v'' + v^2 = 0, v(0)=v(1)=0 (rama positiva) y comparar el máximo.
    from scipy.integrate import solve_bvp

    x = np.linspace(0, 1, 201)
    y0 = np.vstack([12 * 4 * x * (1 - x), 12 * 4 * (1 - 2 * x)])
    sol = solve_bvp(lambda x, y: np.vstack([y[1], -y[0] ** 2]),
                    lambda a, b: np.array([a[0], b[0]]), x, y0, tol=1e-8)
    assert sol.success
    assert abs(sol.sol(0.5)[0] - stationary_max()) < 1e-3
    assert abs(stationary_max() - 11.7967) < 1e-3


def test_deterministic_small_data_decays_large_blows_up():
    s = Scheme()
    x = np.linspace(0, 1, s.N + 1)
    small = simulate_path(0.0, np.random.default_rng(0), s, u0=5 * np.sin(np.pi * x), t_max=2.0)
    assert np.isnan(small.T) and small.umax[-1] < 1e-2
    big = simulate_path(0.0, np.random.default_rng(0), s, u0=30 * np.sin(np.pi * x), t_max=2.0)
    assert np.isfinite(big.T) and big.T < 1.0


def test_noise_scaling_space_time_white_noise():
    # Sin reacción, la varianza estacionaria de u(1/2) para ruido blanco
    # espacio-temporal es G(1/2,1/2)/2 = x(1-x)/2 = 1/8 (sigma=1).
    s = Scheme(f=zero_f, tau_max=0.2 * 0.02**2, u_max=np.inf)
    rng = np.random.default_rng(1)
    U = np.zeros((1000, s.N - 1))
    t = 0.0
    while t < 0.6:
        U, dt = s.step(U, 1.0, rng)
        t += dt[0]
    var = U[:, s.N // 2 - 1].var()
    assert abs(var - 1 / 8) < 0.15 / 8 * 1.5


def test_explosion_times_vectorized_sigma_and_tmax():
    T = explosion_times(np.array([20.0, 20.0, 0.0]), 3, np.random.default_rng(2), t_max=5.0)
    assert np.isfinite(T[:2]).all() and np.isnan(T[2])
