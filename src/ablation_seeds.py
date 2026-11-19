"""Rerun the thin-file ablation over several random splits.

The single-split ablation in analysis.py shows gaps of 0.01-0.05 AUC between steps.
This script repeats it over 5 different train/valid/test splits (and model seeds) and reports
the mean and spread, so we can tell which differences are real and which are split noise.

Writes results/ablation_seeds.csv (one row per seed x block x segment)
and results/ablation_seeds_summary.csv (mean, sd, min, max).
"""
import pandas as pd

from analysis import RES, ablation, load, split

SEEDS = [42, 7, 123, 2024, 31]


def main():
    df, blocks = load()
    runs = []
    for seed in SEEDS:
        print(f"seed {seed}")
        train, valid, test = split(df, seed)
        runs.append(ablation(train, valid, test, blocks, seed).assign(seed=seed))
    runs = pd.concat(runs)
    runs.to_csv(RES / "ablation_seeds.csv", index=False)

    # The thin-vs-thick gap within each seed is less noisy than either AUC alone.
    wide = runs.pivot_table(index=["seed", "added_block"], columns="segment", values="auc").reset_index()
    wide["gap_thin_minus_thick"] = wide["thin file"] - wide["has bureau record"]
    summary = (
        wide.groupby("added_block", sort=False)[["has bureau record", "thin file", "gap_thin_minus_thick"]]
        .agg(["mean", "std", "min", "max"])
        .round(4)
    )
    summary.to_csv(RES / "ablation_seeds_summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
