"""Utilidades compartidas por los scripts de experimentos."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"


def parser(description: str, seed: int) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--seed", type=int, default=seed, help="semilla (default: %(default)s)")
    return p


def save(fig, subdir: str, name: str) -> Path:
    out = FIGURES / subdir
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"guardado: {path.relative_to(ROOT)}")
    return path
