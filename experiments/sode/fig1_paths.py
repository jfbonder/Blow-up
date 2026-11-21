r"""Figura 1 de Dávila et al.: tres trayectorias que explotan.

Esquema adaptativo con :math:`b(\xi)=|\xi|^{1.1}+0.1`, :math:`\sigma=\sqrt b`,
:math:`z=10`, detenido cuando la solución calculada alcanza :math:`M=10^5`
(valor dado en el paper).

Elección propia (el paper no da :math:`h`): :math:`h=0.25`.  Con ese valor
cada trayectoria necesita :math:`\approx M/h = 4\times10^5` pasos, que es el
rango del eje :math:`k` de la Figura 2; así las Figuras 1 y 2 muestran las
mismas trayectorias (misma semilla).

Además de la figura principal genera dos ampliaciones (Figs. 1.1 y 1.2 del
paper) que muestran la rugosidad browniana de las trayectorias.

Uso::

    python -m experiments.sode.fig1_paths [--seed N]
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from blowup.sode import Z0_PAPER, b_paper, sigma_paper, simulate_path
from experiments.common import parser, save

H = 0.25
M = 1e5
N_PATHS = 3
COLORS = ["tab:green", "tab:blue", "tab:red"]


def compute(seed: int, h: float = H, M: float = M, n_paths: int = N_PATHS):
    rng = np.random.default_rng(seed)
    return [simulate_path(b_paper, sigma_paper, Z0_PAPER, h, M, rng) for _ in range(n_paths)]


def main() -> None:
    args = parser(__doc__.splitlines()[0], seed=1).parse_args()
    paths = compute(args.seed)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p, c in zip(paths, COLORS):
        ax.plot(p.t, p.X, color=c, lw=0.8, label=f"$R_h^M = {p.stopping_time:.3f}$")
        print(f"R_h^M = {p.stopping_time:.4f}   pasos = {p.n_steps}")
    ax.set_xlabel("$t$")
    ax.set_ylabel("$X(t)$")
    ax.set_title(rf"Tres trayectorias con explosión ($h={H}$, $M=10^5$, $z=10$)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    save(fig, "sode", "fig1_paths")

    # Fig. 1.1: comienzo de las tres trayectorias.
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for p, c in zip(paths, COLORS):
        sel = p.X <= 4500
        k_end = np.argmax(~sel) if (~sel).any() else len(sel)
        ax.plot(p.t[:k_end], p.X[:k_end], color=c, lw=0.6)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$X(t)$")
    ax.set_title(r"Ampliación: comienzo ($X \leq 4500$)")
    ax.grid(alpha=0.3)
    save(fig, "sode", "fig1_1_zoom_start")

    # Fig. 1.2: ampliación de la segunda trayectoria cerca de X ~ 1.3e4 - 2e4.
    p = paths[1]
    sel = (p.X >= 1.3e4) & (p.X <= 2e4)
    i0, i1 = np.argmax(sel), len(sel) - np.argmax(sel[::-1])
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(p.t[i0:i1], p.X[i0:i1], color=COLORS[1], lw=0.6)
    ax.set_xlabel("$t$")
    ax.set_ylabel("$X(t)$")
    ax.set_title(r"Ampliación: trayectoria azul, $1.3\cdot10^4 \leq X \leq 2\cdot10^4$")
    ax.grid(alpha=0.3)
    save(fig, "sode", "fig1_2_zoom_middle")


if __name__ == "__main__":
    main()
