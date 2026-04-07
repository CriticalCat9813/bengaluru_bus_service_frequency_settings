"""
Plot simple comparison charts for baseline full-core vs k-core extension results.

Expected input:
  C:\game_heory\results\summary\extB_summary_bengaluru_india_n1430.csv
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

from src.config import *

N_PLAYERS = 1430
SUMMARY_CSV = f"{RELPATH}/results/summary/extB_summary_{FILENAME}_n{N_PLAYERS}.csv"
OUT_DIR = Path(f"{RELPATH}/results/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _k_numeric(x):
    if x == "full":
        return None
    try:
        return int(x)
    except Exception:
        return None


def main():
    df = pd.read_csv(SUMMARY_CSV)
    df["k_numeric"] = df["k_cap"].apply(_k_numeric)

    for objective in sorted(df["objective"].unique()):
        sub = df[df["objective"] == objective].copy()

        baseline = sub[sub["k_cap"] == "full"].copy()
        capped = sub[sub["k_cap"] != "full"].copy().sort_values("k_numeric")

        plt.figure(figsize=(8, 5))
        if not baseline.empty:
            y0 = float(baseline["utilitarian_welfare"].iloc[0])
            plt.axhline(y0, linestyle="--", label="full-core baseline")
        if not capped.empty:
            plt.plot(capped["k_numeric"], capped["utilitarian_welfare"], marker="o", label="k-core")
        plt.xlabel("k")
        plt.ylabel("Utilitarian welfare")
        plt.title(f"{objective}: utilitarian welfare vs k")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{objective}_util_welfare_vs_k.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        if not baseline.empty:
            y0 = float(baseline["maximin_welfare"].iloc[0])
            plt.axhline(y0, linestyle="--", label="full-core baseline")
        if not capped.empty:
            plt.plot(capped["k_numeric"], capped["maximin_welfare"], marker="o", label="k-core")
        plt.xlabel("k")
        plt.ylabel("Maximin welfare")
        plt.title(f"{objective}: maximin welfare vs k")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{objective}_maximin_welfare_vs_k.png", dpi=200)
        plt.close()

        plt.figure(figsize=(8, 5))
        if not baseline.empty:
            y0 = float(baseline["eps_final"].iloc[0])
            plt.axhline(y0, linestyle="--", label="full-core baseline")
        if not capped.empty:
            plt.plot(capped["k_numeric"], capped["eps_final"], marker="o", label="k-core")
        plt.xlabel("k")
        plt.ylabel("Final multiplicative least objection")
        plt.title(f"{objective}: final least objection vs k")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"{objective}_eps_vs_k.png", dpi=200)
        plt.close()

    print(f"Saved figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
