"""Figures for the README, built from the CSVs in results/."""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RES = Path("results")
BLUE, ORANGE, GREY = "#2b6cb0", "#dd6b20", "#a0aec0"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

LABELS = {"core": "Loan terms, income,\nemployment, family", "region_housing": "+ region & housing",
          "alternative_data": "+ phone, documents,\nsocial circle", "previous_apps": "+ previous Home Credit\napplications",
          "external_scores": "+ external scores", "bureau": "+ credit bureau\nhistory"}


def fig_audit():
    a = pd.read_csv(RES / "group_audit.csv")
    a = a[(a["model"] == "xgboost") & (a["group"] != "all")].reset_index(drop=True)
    labels = a["level"].tolist()
    y = range(len(a))[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5.2), sharey=True)
    ax1.errorbar(a["auc"], y, xerr=[a["auc"] - a["auc_lo"], a["auc_hi"] - a["auc"]], fmt="o", color=BLUE, capsize=3)
    overall = pd.read_csv(RES / "group_audit.csv").query("model=='xgboost' and group=='all'")["auc"].item()
    ax1.axvline(overall, color=GREY, ls="--", lw=1)
    ax1.set_yticks(list(y), labels)
    ax1.set_xlabel("AUC (95% bootstrap CI)")
    ax1.set_title("How well the model ranks risk")

    ax2.barh(list(y), a["good_borrowers_rejected"], color=[ORANGE if l == "thin file" else BLUE for l in labels])
    for yi, v in zip(y, a["good_borrowers_rejected"]):
        ax2.text(v + 0.005, yi, f"{v:.0%}", va="center", fontsize=9)
    ax2.set_xlabel("Share of borrowers who repaid but would be rejected")
    ax2.set_title("Who pays for a single 80%-approval cutoff")
    ax2.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    # dividers between group types
    starts = a.groupby("group", sort=False).head(1).index.tolist()[1:]
    for ax in (ax1, ax2):
        for s in starts:
            ax.axhline(len(a) - s - 0.5, color="#e2e8f0", lw=1)
    fig.tight_layout()
    fig.savefig(RES / "fig1_group_audit.png", dpi=160)


def fig_ablation():
    order = list(LABELS)
    seeds_file = RES / "ablation_seeds.csv"
    runs = pd.read_csv(seeds_file if seeds_file.exists() else RES / "ablation.csv")
    stats = runs.groupby(["added_block", "segment"])["auc"].agg(["mean", "min", "max"]).reset_index()
    n_seeds = runs["seed"].nunique() if "seed" in runs else 1

    fig, ax = plt.subplots(figsize=(9, 4))
    x = range(len(order))
    for segment, color, label in [("has bureau record", BLUE, "has bureau record"),
                                  ("thin file", ORANGE, "thin file (no bureau record)")]:
        s = stats[stats["segment"] == segment].set_index("added_block").loc[order]
        ax.plot(x, s["mean"], "o-", color=color, label=label)
        if n_seeds > 1:
            ax.fill_between(x, s["min"], s["max"], color=color, alpha=0.15, lw=0)
    if n_seeds > 1:
        ax.text(0.99, 0.02, f"mean of {n_seeds} random splits; band = min to max",
                transform=ax.transAxes, ha="right", fontsize=8, color="#4a5568")
    ax.set_xticks(list(x), [LABELS[b] for b in order], fontsize=8)
    ax.set_ylabel("Test AUC")
    ax.set_title("What predicts default without a credit history? (blocks added left to right)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RES / "fig2_ablation.png", dpi=160)


if __name__ == "__main__":
    fig_audit()
    fig_ablation()
    print("figures written to results/")
