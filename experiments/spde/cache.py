"""Cache en disco de los tiempos de explosión (las corridas tardan minutos)."""

from __future__ import annotations

import numpy as np

from blowup.spde import Scheme, explosion_times
from experiments.common import ROOT

RESULTS = ROOT / "results" / "spde"


def get_times(sigma: float, n: int, seed: int, scheme: Scheme, t_max: float = np.inf,
              recompute: bool = False) -> np.ndarray:
    """Tiempos de explosión para ``n`` trayectorias; se guardan en ``results/spde``."""
    name = (f"T_sigma{sigma:g}_n{n}_seed{seed}_h{scheme.h:g}_tau{scheme.tau:g}"
            f"_taumax{scheme.tau_max:g}_tmax{t_max:g}.npy")
    path = RESULTS / name
    if path.exists() and not recompute:
        print(f"cargado de {path.relative_to(ROOT)}")
        return np.load(path)
    print(f"simulando {n} trayectorias con sigma={sigma:g} ...", flush=True)
    T = explosion_times(sigma, n, np.random.default_rng(seed), scheme, t_max=t_max, progress=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    np.save(path, T)
    print(f"guardado: {path.relative_to(ROOT)}")
    return T
