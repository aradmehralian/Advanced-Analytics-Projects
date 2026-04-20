# TabPFN local run (VS Code / macOS)

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr

import os

from tabpfn import TabPFNRegressor

# 1. LOAD DATA

transactions = pd.read_csv(
    "transactions_2016_2017.csv",
    parse_dates=["order_date", "pack_date"],
    low_memory=False
)

train_customers = pd.read_csv("customer_clv_train.csv")
test_customers  = pd.read_csv("customer_clv_test.csv")


# 2. STRATIFIED SPLIT

def stratified_revenue_split(
    df,
    target_col="revenue_2018_2019",
    test_size=0.2,
    random_state=42
):
    df = df.copy()
    df["strat_col"] = 0

    mask_pos = df[target_col] > 0
    if mask_pos.any():
        df.loc[mask_pos, "strat_col"] = pd.qcut(
            df.loc[mask_pos, target_col],
            q=4,
            labels=[1, 2, 3, 4]
        ).astype(int)

    train_idx, val_idx = train_test_split(
        df.index,
        test_size=test_size,
        random_state=random_state,
        stratify=df["strat_col"]
    )

    train_df = df.loc[train_idx].drop(columns="strat_col")
    val_df   = df.loc[val_idx].drop(columns="strat_col")

    return train_df, val_df


train_data, val_data = stratified_revenue_split(train_customers)

# 3. FEATURE ENGINEERING (minimal placeholder)

def build_features(df):
    max_date = pd.Timestamp("2017-12-31")

    df = df.copy()
    df["days_old"] = (max_date - df["order_date"]).dt.days
    df["decay_weight"] = np.exp(-df["days_old"] / 365)
    df["weighted_rev"] = df["sale_revenue"] * df["decay_weight"]

    agg = df.groupby("cust_id").agg(
        total_revenue=("sale_revenue", "sum"),
        weighted_revenue=("weighted_rev", "sum"),
        avg_revenue=("sale_revenue", "mean"),
        n_transactions=("sale_id", "nunique"),
    )

    return agg.fillna(0)


X_all = build_features(transactions)

# 4. MERGE + PREPARE
X_train_raw = train_data[["cust_id", "revenue_2018_2019"]].merge(
    X_all, on="cust_id", how="left"
)

X_val_raw = val_data[["cust_id", "revenue_2018_2019"]].merge(
    X_all, on="cust_id", how="left"
)

y_train = X_train_raw["revenue_2018_2019"]
y_val   = X_val_raw["revenue_2018_2019"]

X_train = X_train_raw.drop(columns=["cust_id", "revenue_2018_2019"]).fillna(0)
X_val   = X_val_raw.drop(columns=["cust_id", "revenue_2018_2019"]).fillna(0)

# 5. TABPFN (LOCAL)
print(f"Training customers: {len(X_train)}")

model = TabPFNRegressor(
    device="cpu",
    ignore_pretraining_limits=True
)

model.fit(X_train, np.log1p(y_train))

# 6. EVALUATION

val_preds_log = model.predict(X_val)
val_preds     = np.expm1(val_preds_log)

mae = mean_absolute_error(y_val, val_preds)
spearman, _ = spearmanr(y_val, val_preds)

print(f"Validation MAE: {mae:.4f}")
print(f"Validation Spearman: {spearman:.4f}")

# I'm still struggling w Git & Python but these are the results from my terminal
# Training customers: 93272
# Validation MAE: 65.1241
# Validation Spearman: 0.3786