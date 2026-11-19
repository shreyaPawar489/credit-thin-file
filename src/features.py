"""Build one row per applicant from the Home Credit tables.

Output: data/features.parquet with
  - application features (cleaned)
  - credit bureau aggregates (empty for thin-file applicants)
  - previous Home Credit application aggregates
  - group columns used only for auditing (thin_file, age_band, gender, income_type)
and a JSON file mapping each feature to a "feature block" used in the ablation.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data")


def load_application():
    app = pd.read_csv(RAW / "application_train.csv")

    # DAYS_EMPLOYED = 365243 is a placeholder (mostly pensioners), not 1,000 years of work.
    app["EMPLOYED_PLACEHOLDER"] = (app["DAYS_EMPLOYED"] == 365243).astype(int)
    app.loc[app["DAYS_EMPLOYED"] == 365243, "DAYS_EMPLOYED"] = np.nan

    # A few simple affordability ratios.
    app["CREDIT_TO_INCOME"] = app["AMT_CREDIT"] / app["AMT_INCOME_TOTAL"]
    app["ANNUITY_TO_INCOME"] = app["AMT_ANNUITY"] / app["AMT_INCOME_TOTAL"]
    app["CREDIT_TO_GOODS"] = app["AMT_CREDIT"] / app["AMT_GOODS_PRICE"]
    app["AGE_YEARS"] = -app["DAYS_BIRTH"] / 365.25
    return app


def bureau_aggregates():
    b = pd.read_csv(RAW / "bureau.csv")
    b["IS_ACTIVE"] = (b["CREDIT_ACTIVE"] == "Active").astype(int)
    b["HAS_OVERDUE"] = (b["CREDIT_DAY_OVERDUE"] > 0).astype(int)
    agg = b.groupby("SK_ID_CURR").agg(
        BUR_N_LOANS=("SK_ID_BUREAU", "count"),
        BUR_N_ACTIVE=("IS_ACTIVE", "sum"),
        BUR_ANY_OVERDUE=("HAS_OVERDUE", "max"),
        BUR_MAX_OVERDUE_AMT=("AMT_CREDIT_MAX_OVERDUE", "max"),
        BUR_TOTAL_DEBT=("AMT_CREDIT_SUM_DEBT", "sum"),
        BUR_TOTAL_CREDIT=("AMT_CREDIT_SUM", "sum"),
        BUR_N_PROLONGED=("CNT_CREDIT_PROLONG", "sum"),
        BUR_DAYS_FIRST_CREDIT=("DAYS_CREDIT", "min"),
        BUR_DAYS_LAST_CREDIT=("DAYS_CREDIT", "max"),
    )
    agg["BUR_DEBT_SHARE"] = agg["BUR_TOTAL_DEBT"] / agg["BUR_TOTAL_CREDIT"].replace(0, np.nan)
    return agg


def previous_app_aggregates():
    p = pd.read_csv(
        RAW / "previous_application.csv",
        usecols=["SK_ID_CURR", "SK_ID_PREV", "NAME_CONTRACT_STATUS", "AMT_CREDIT", "DAYS_DECISION", "CNT_PAYMENT"],
    )
    p["APPROVED"] = (p["NAME_CONTRACT_STATUS"] == "Approved").astype(int)
    p["REFUSED"] = (p["NAME_CONTRACT_STATUS"] == "Refused").astype(int)
    agg = p.groupby("SK_ID_CURR").agg(
        PREV_N_APPS=("SK_ID_PREV", "count"),
        PREV_SHARE_APPROVED=("APPROVED", "mean"),
        PREV_SHARE_REFUSED=("REFUSED", "mean"),
        PREV_MEAN_CREDIT=("AMT_CREDIT", "mean"),
        PREV_DAYS_LAST_DECISION=("DAYS_DECISION", "max"),
        PREV_MEAN_TERM=("CNT_PAYMENT", "mean"),
    )
    return agg


def feature_blocks(columns):
    """Assign every model feature to one block, for the thin-file ablation."""
    ext = {"EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"}
    blocks = {}
    for c in columns:
        if c.startswith("BUR_") or c.startswith("AMT_REQ_CREDIT_BUREAU"):
            blocks[c] = "bureau"
        elif c in ext:
            blocks[c] = "external_scores"
        elif c.startswith("PREV_"):
            blocks[c] = "previous_apps"
        elif (
            c.startswith("FLAG_DOCUMENT")
            or c in {"FLAG_MOBIL", "FLAG_EMP_PHONE", "FLAG_WORK_PHONE", "FLAG_CONT_MOBILE", "FLAG_PHONE", "FLAG_EMAIL"}
            or "SOCIAL_CIRCLE" in c
            or c in {"DAYS_LAST_PHONE_CHANGE", "DAYS_ID_PUBLISH", "DAYS_REGISTRATION",
                     "HOUR_APPR_PROCESS_START", "WEEKDAY_APPR_PROCESS_START"}
        ):
            blocks[c] = "alternative_data"
        elif (
            c.startswith("REGION_") or c.startswith("REG_") or c.startswith("LIVE_")
            or c.endswith(("_AVG", "_MODE", "_MEDI"))
        ):
            blocks[c] = "region_housing"
        else:
            blocks[c] = "core"  # loan terms, income, employment, education, family
    return blocks


def main():
    app = load_application()
    bur = bureau_aggregates()
    prev = previous_app_aggregates()

    df = app.merge(bur, left_on="SK_ID_CURR", right_index=True, how="left")
    df = df.merge(prev, left_on="SK_ID_CURR", right_index=True, how="left")

    # Thin file = no record at the credit bureau at all.
    df["thin_file"] = df["BUR_N_LOANS"].isna().astype(int)
    df["PREV_N_APPS"] = df["PREV_N_APPS"].fillna(0)

    # Audit-only group columns (not model features).
    df["age_band"] = pd.cut(df["AGE_YEARS"], [0, 30, 40, 50, 60, 100], labels=["<30", "30-39", "40-49", "50-59", "60+"])
    df["gender"] = df["CODE_GENDER"].replace("XNA", np.nan)
    df["income_type"] = df["NAME_INCOME_TYPE"].where(
        df["NAME_INCOME_TYPE"].isin(["Working", "Commercial associate", "Pensioner", "State servant"]), "Other"
    )

    # Gender is excluded from the model (as fair-lending rules usually require) but kept for the audit.
    # DAYS_BIRTH is replaced by AGE_YEARS.
    drop = {"SK_ID_CURR", "TARGET", "CODE_GENDER", "DAYS_BIRTH", "thin_file", "age_band", "gender", "income_type"}
    features = [c for c in df.columns if c not in drop]

    blocks = feature_blocks(features)
    OUT.mkdir(exist_ok=True)
    df.to_parquet(OUT / "features.parquet", index=False)
    with open(OUT / "feature_blocks.json", "w") as f:
        json.dump(blocks, f, indent=1)

    print(f"{len(df):,} applicants, {len(features)} features")
    print(f"thin-file share: {df['thin_file'].mean():.1%}")
    print(pd.Series(blocks).value_counts())


if __name__ == "__main__":
    main()
