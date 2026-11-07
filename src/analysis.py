"""Cleaning, feature preparation and model code for the juice sales project.

The notebook (notebooks/juice_analysis.ipynb) walks through the same steps with
explanations and charts. Run this file to reproduce the cleaning + model results:

    python src/analysis.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "juice_dataset.csv"
RANDOM_STATE = 42
NUMERIC_FEATURES = ["price_per_100ml", "pack_size_ml", "discount_pct", "rating", "sugar_g_per_100ml"]


def load_data(path=DATA_PATH):
    return pd.read_csv(path)


def clean_data(df):
    df = df.copy()

    # inconsistent text: " orange", "ORANGE" and "Orange" are the same category
    df["category"] = df["category"].str.strip().str.title()
    df["is_organic"] = df["is_organic"].str.strip().str.lower().map(
        {"yes": True, "y": True, "no": False, "n": False}
    )

    df = df.drop_duplicates()

    # ratings must be between 1 and 5
    df.loc[~df["rating"].between(1, 5), "rating"] = np.nan

    # quantity is the thing we want to predict, so rows without a valid value are dropped
    df = df[df["quantity_sold"] > 0].copy()

    # a price far above the usual price for its pack size is a typing error
    typical_price = df.groupby("pack_size_ml")["price"].transform("median")
    df.loc[df["price"] > 3 * typical_price, "price"] = np.nan

    # fill remaining gaps with the median of a sensible group (robust to outliers)
    df["price"] = df["price"].fillna(df.groupby("pack_size_ml")["price"].transform("median"))
    for col in ["rating", "sugar_g_per_100ml"]:
        df[col] = df[col].fillna(df.groupby("category")[col].transform("median"))

    df["quantity_sold"] = df["quantity_sold"].astype(int)
    # revenue was partly missing and depends on price, so recompute it for every row
    df["revenue"] = (df["price"] * (1 - df["discount_pct"] / 100) * df["quantity_sold"]).round(2)
    return df.reset_index(drop=True)


def prepare_features(df):
    """Returns X, y. Target is quantity_sold; revenue is NOT used as a feature (it contains the target)."""
    df = df.copy()
    df["price_per_100ml"] = df["price"] / df["pack_size_ml"] * 100
    df["is_summer"] = df["month"].between(5, 8)
    X = df[NUMERIC_FEATURES + ["is_organic", "is_summer", "category"]]
    X = pd.get_dummies(X, columns=["category"], drop_first=True, dtype=int)
    X[["is_organic", "is_summer"]] = X[["is_organic", "is_summer"]].astype(int)
    return X, df["quantity_sold"]


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)

    # scale only the numeric columns, fitting on training data only
    scaler = StandardScaler().fit(X_train[NUMERIC_FEATURES])
    X_train, X_test = X_train.copy(), X_test.copy()
    X_train[NUMERIC_FEATURES] = scaler.transform(X_train[NUMERIC_FEATURES])
    X_test[NUMERIC_FEATURES] = scaler.transform(X_test[NUMERIC_FEATURES])

    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)

    mse = mean_squared_error(y_test, pred)
    metrics = {
        "MAE": mean_absolute_error(y_test, pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_test, pred),
    }
    return model, metrics, y_test, pred, X_train.columns


if __name__ == "__main__":
    raw = load_data()
    df = clean_data(raw)
    print(f"Rows: {len(raw)} raw -> {len(df)} after cleaning")
    X, y = prepare_features(df)
    model, metrics, y_test, pred, cols = train_and_evaluate(X, y)
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")
    print("\nCoefficients (numeric features are standardised):")
    print(pd.Series(model.coef_, index=cols).sort_values(key=abs, ascending=False).round(2))
