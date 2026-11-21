r"""Figura 2 de Dávila et al.: :math:`X_k/(hk) \to 1` casi seguramente.

Ilustra el Teorema 1.3(2).  Usa las mismas trayectorias que la Figura 1
(:math:`h=0.25`, :math:`M=10^5`, misma semilla); el paper no da :math:`h` ni
:math:`M` para esta figura, pero el eje :math:`k` llega a :math:`4\times10^5
\approx M/h`.

Uso::

    python -m experiments.sode.fig2_ratio [--seed N]
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from experiments.common import parser, save
from experiments.sode.fig1_paths import COLORS, H, compute


def main() -> None:
    args = parser(__doc__.splitlines()[0], seed=1).parse_args()
    paths = compute(args.seed)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p, c in zip(paths, COLORS):
        k = np.arange(1, len(p.X))
        ratio = p.X[1:] / (H * k)
        ax.plot(k, ratio, color=c, lw=0.8)
        print(f"X_k/(hk) en el último paso (k={k[-1]}): {ratio[-1]:.4f}")
    ax.axhline(1.0, color="k", ls="--", lw=1)
    ax.set_ylim(0.8, 1.6)
    ax.set_xlabel("$k$")
    ax.set_ylabel("$X_k / (hk)$")
    ax.set_title(rf"$X_k/(hk) \to 1$ c.s. ($h={H}$, $z=10$)")
    ax.grid(alpha=0.3)
    save(fig, "sode", "fig2_ratio")


if __name__ == "__main__":
    main()
