"""
Helpers to synthesise market-price and demand series for the battery
simulation.  Keeps the generation logic in one place so notebooks or
tests can import it as well.

Usage
-----
from src.data.series_model import generate_price_series, constant_demand

prices  = generate_price_series(horizon=150, seed=42)
demand  = constant_demand(horizon=150, level=5.0)
"""
from __future__ import annotations
import numpy as np


def generate_price_series(
    horizon: int,
    *,
    seed: int | None = 42,
    lo: float = 0.05,
    hi: float = 1.00,
) -> list[float]:
    """
    Produce a noisy upward-drifting price curve that resembles Figure 3
    of the paper.

    Parameters
    ----------
    horizon : int
        Number of *simulation steps* (not points) – returns horizon+1 floats.
    seed : int | None
        Seed for reproducible randomness.
    lo, hi : float
        Lower / upper clipping bounds.

    Returns
    -------
    list[float]  length == horizon + 1
    """
    rng = np.random.default_rng(seed)
    t   = np.arange(horizon + 1)

    mean_trend = 0.45 + 0.30 * (t / horizon)                 # slow rise
    noise      = rng.normal(0.0, 0.15 + 0.10 * (t / horizon))  # growing σ

    prices = np.clip(mean_trend + noise, lo, hi)
    return prices.tolist()


def constant_demand(horizon: int, *, level: float = 5.0) -> list[float]:
    """
    Flat demand line used in the paper.
    """
    return [level] * (horizon + 1)
