#/Users/nashe/nested_policy_pipeline/src/main.py
"""
Entry-point script.
"""
from __future__ import annotations
import argparse
import logging
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt

from src.algorithm.nested_algorithm import run_nested_algorithm
from src.environment.battery_env import BatteryEnvironment
from config import HORIZON


# ------------------------------------------------------------------
def run_baseline() -> float:
    env = BatteryEnvironment()
    state = env.reset()
    for _ in range(HORIZON):
        state = env.step(0.0)
    return float(state[3])


def plot_savings(savings: List[float], out: Path | None = None) -> None:
    iters = list(range(1, len(savings) + 1))
    plt.figure(figsize=(6, 3))
    plt.plot(iters, savings, marker="o", markerfacecolor="white")
    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Cost savings [%]", fontsize=12)
    plt.title("Development of cost savings over iterations", fontsize=13)
    plt.grid(True, alpha=0.4)
    if out:
        plt.tight_layout()
        plt.savefig(out, dpi=300)
        logging.info("Figure written to %s", out.resolve())
    else:
        plt.show()


# ------------------------------------------------------------------
def main(save_only: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    logging.info("▶ Baseline run (battery off)…")
    baseline_cost = run_baseline()
    logging.info("Baseline cost  : %.3f", baseline_cost)

    logging.info("▶ Nested-policy pipeline…")
    results = run_nested_algorithm()
    seg_costs = [seg["segment_cost"] for seg in results["per_segment"]]

    # ---- FIXED line (removed stray backslashes) ----
    savings_pct = [(baseline_cost - c) / baseline_cost * 100 for c in seg_costs]
    logging.info("Segment savings (%%): %s",
                 [f"{s:.1f}" for s in savings_pct])
    # -----------------------------------------------

    fig_path = Path("fig_cost_savings.png")
    plot_savings(savings_pct, fig_path if save_only else None)


# ------------------------------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--save-only", action="store_true")
    args = p.parse_args()
    main(save_only=args.save_only)
