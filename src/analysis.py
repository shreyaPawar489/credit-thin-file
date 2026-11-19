"""Train the default models and audit them by borrower group.

Steps
  1. Split 60/20/20 (train / validation / test), stratified on default and thin-file status.
  2. Baseline: logistic regression. Main model: XGBoost (validation set used only for early stopping).
  3. Audit on the test set: AUC with bootstrap CIs, calibration, and a simple approval policy, by group.
  4. Thin-file ablation: add feature blocks one at a time and see what predicts default
     for people with no bureau record.
  5. SHAP-style contributions (XGBoost's built-in pred_contribs) for thin vs thick files.

Writes everything to results/.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
APPROVAL_RATE = 0.80  # the lender approves the 80% of applicants with the lowest predicted risk
N_BOOT = 300
RES = Path("results")
GROUPS = ["thin_file", "age_band", "gender", "income_type"]
ABLATION = ["core", "region_housing", "alternative_data", "previous_apps", "external_scores", "bureau"]


# ---------- data ----------

def load():
    df = pd.read_parquet("data/features.parquet")
    with open("data/feature_blocks.json") as f:
        blocks = json.load(f)
    for c in df.columns:
        if df[c].dtype == object and c in blocks:
            df[c] = df[c].astype("category")
    df["thin_file"] = df["thin_file"].map({1: "thin file", 0: "has bureau record"})
    return df, blocks


def split(df):
    strata = df["TARGET"].astype(str) + "_" + df["thin_file"]
    train, rest = train_test_split(df, test_size=0.4, stratify=strata, random_state=SEED)
    valid, test = train_test_split(rest, test_size=0.5, stratify=strata.loc[rest.index], random_state=SEED)
    return train, valid, test


# ---------- models ----------

def fit_xgb(train, valid, cols):
    model = xgb.XGBClassifier(
        n_estimators=3000, learning_rate=0.05, max_depth=5, min_child_weight=20,
        subsample=0.8, colsample_bytree=0.6, reg_lambda=5,
        tree_method="hist", enable_categorical=True, eval_metric="auc",
        early_stopping_rounds=100, random_state=SEED, n_jobs=-1,
    )
    model.fit(train[cols], train["TARGET"], eval_set=[(valid[cols], valid["TARGET"])], verbose=False)
    return model


def fit_logit(train, cols):
    cat = [c for c in cols if str(train[c].dtype) == "category"]
    num = [c for c in cols if c not in cat]
    prep = ColumnTransformer([
        ("num", make_pipeline(SimpleImputer(strategy="median", add_indicator=True), StandardScaler()), num),
        ("cat", make_pipeline(SimpleImputer(strategy="constant", fill_value="missing"),
                              OneHotEncoder(handle_unknown="ignore", min_frequency=50)), cat),
    ])
    model = make_pipeline(prep, LogisticRegression(C=0.1, max_iter=2000))
    # sklearn wants plain object columns, not pandas categories
    X = train[cols].copy()
    X[cat] = X[cat].astype(object)
    model.fit(X, train["TARGET"])
    return model, cat


def predict_logit(model, cat, df, cols):
    X = df[cols].copy()
    X[cat] = X[cat].astype(object)
    return model.predict_proba(X)[:, 1]


# ---------- metrics ----------

def auc_ci(y, p, rng):
    """AUC with a percentile bootstrap 95% interval."""
    y, p = np.asarray(y), np.asarray(p)
    point = roc_auc_score(y, p)
    boots = []
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            boots.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def group_audit(test, pred_col, cutoff, rng):
    """Discrimination, calibration and approval outcomes for every group."""
    rows = []
    for g in ["all"] + GROUPS:
        groups = [("all", test)] if g == "all" else test.groupby(g, observed=True)
        for level, d in groups:
            if len(d) < 200:
                continue
            auc, lo, hi = auc_ci(d["TARGET"], d[pred_col], rng)
            approved = d[pred_col] <= cutoff
            good = d["TARGET"] == 0
            rows.append({
                "group": g, "level": str(level), "n": len(d),
                "default_rate": d["TARGET"].mean(),
                "mean_predicted": d[pred_col].mean(),
                "auc": auc, "auc_lo": lo, "auc_hi": hi,
                "approval_rate": approved.mean(),
                "default_rate_if_approved": d.loc[approved, "TARGET"].mean(),
                # share of people who would have repaid but were turned down
                "good_borrowers_rejected": (~approved & good).sum() / good.sum(),
            })
    return pd.DataFrame(rows)


def ablation(train, valid, test, blocks):
    """Add feature blocks cumulatively; score thin-file and thick-file test borrowers separately."""
    rows, cols = [], []
    for block in ABLATION:
        cols = cols + [c for c, b in blocks.items() if b == block]
        model = fit_xgb(train, valid, cols)
        p = model.predict_proba(test[cols])[:, 1]
        for level, d in test.assign(p=p).groupby("thin_file"):
            rows.append({"added_block": block, "n_features": len(cols), "segment": level,
                         "auc": roc_auc_score(d["TARGET"], d["p"])})
        print(f"  + {block:17s} ({len(cols):3d} features) done")
    return pd.DataFrame(rows)


def contributions(model, test, cols, n=8000):
    """Mean |contribution| per feature (TreeSHAP, computed by XGBoost itself), thin vs thick."""
    out = {}
    for level, d in test.groupby("thin_file"):
        d = d.sample(min(n, len(d)), random_state=SEED)
        contrib = model.get_booster().predict(xgb.DMatrix(d[cols], enable_categorical=True), pred_contribs=True)
        out[level] = pd.Series(np.abs(contrib[:, :-1]).mean(axis=0), index=cols)  # last column is the bias term
    return pd.DataFrame(out)


# ---------- main ----------

def main():
    RES.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    df, blocks = load()
    cols = list(blocks)
    train, valid, test = split(df)
    test = test.copy()
    print(f"train {len(train):,} | valid {len(valid):,} | test {len(test):,}")

    print("fitting logistic regression...")
    logit, cat = fit_logit(train, cols)
    test["p_logit"] = predict_logit(logit, cat, test, cols)

    print("fitting XGBoost...")
    model = fit_xgb(train, valid, cols)
    test["p_xgb"] = model.predict_proba(test[cols])[:, 1]
    print(f"  best iteration: {model.best_iteration}")

    audits = []
    for name, col in [("logistic", "p_logit"), ("xgboost", "p_xgb")]:
        # One cutoff for everyone, set on the validation set so the test set stays untouched.
        p_valid = predict_logit(logit, cat, valid, cols) if name == "logistic" else model.predict_proba(valid[cols])[:, 1]
        cutoff = np.quantile(p_valid, APPROVAL_RATE)
        audits.append(group_audit(test, col, cutoff, rng).assign(model=name))
    audit = pd.concat(audits)
    audit.to_csv(RES / "group_audit.csv", index=False)

    # Thin-file borrowers skew young. Is the approval gap just an age effect?
    # Compare thin vs thick files inside each age band (XGBoost, same cutoff).
    cutoff = np.quantile(model.predict_proba(valid[cols])[:, 1], APPROVAL_RATE)
    test["approved"] = test["p_xgb"] <= cutoff
    within_age = (
        test[test["TARGET"] == 0]
        .groupby(["age_band", "thin_file"], observed=True)["approved"]
        .agg(good_borrowers_rejected=lambda a: 1 - a.mean(), n_good="size")
        .reset_index()
    )
    within_age.to_csv(RES / "rejection_within_age.csv", index=False)

    print("thin-file ablation...")
    abl = ablation(train, valid, test, blocks)
    abl.to_csv(RES / "ablation.csv", index=False)

    # Does a model trained only on thin-file borrowers beat the pooled model for them?
    thin = lambda d: d[d["thin_file"] == "thin file"]
    thin_model = fit_xgb(thin(train), thin(valid), cols)
    t = thin(test)
    separate = {
        "pooled_model_auc": roc_auc_score(t["TARGET"], t["p_xgb"]),
        "thin_only_model_auc": roc_auc_score(t["TARGET"], thin_model.predict_proba(t[cols])[:, 1]),
    }

    contrib = contributions(model, test, cols)
    contrib.to_csv(RES / "contributions.csv")

    # Missingness of the strongest external score, by segment (explains a lot of the gap).
    missing = test.groupby("thin_file")[["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]].apply(lambda d: d.isna().mean())

    summary = {
        "n_train": len(train), "n_valid": len(valid), "n_test": len(test),
        "thin_file_share": float((df["thin_file"] == "thin file").mean()),
        "xgb_best_iteration": int(model.best_iteration),
        "separate_thin_model": separate,
        "ext_source_missing_share": missing.round(3).to_dict(orient="index"),
    }
    with open(RES / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
