"""Generates the SIMULATED juice sales dataset used in this project.

The data is synthetic (not real store data), so any patterns in it are ones
built in here. Run once from the project root: python src/generate_dataset.py
"""
import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)
N = 650

brands = {"PureSip": 1.00, "FreshSqueeze": 1.10, "TropiCo": 0.90, "GreenLeaf": 1.20, "DailyGlow": 0.85}
# flavor: (base sugar g/100ml, popularity effect on sales, base rating)
flavors = {
    "Orange": (8.5, 60, 4.2), "Apple": (9.5, 45, 4.0), "Mango": (13.0, 35, 4.3),
    "Mixed Fruit": (10.5, 20, 3.8), "Pineapple": (11.5, 0, 3.9),
    "Berry": (9.0, -15, 4.1), "Grape": (14.5, -10, 3.7),
}
regions = ["North", "South", "East", "West"]

brand_names = rng.choice(list(brands), N)
flavor_names = rng.choice(list(flavors), N, p=[0.2, 0.18, 0.15, 0.12, 0.12, 0.12, 0.11])
pack = rng.choice([200, 250, 500, 1000], N, p=[0.2, 0.25, 0.35, 0.2])
month = rng.integers(1, 13, N)
region = rng.choice(regions, N)
organic = rng.random(N) < 0.25
discount = rng.choice([0, 5, 10, 15, 20], N, p=[0.4, 0.2, 0.2, 0.12, 0.08])

sugar = np.array([flavors[f][0] for f in flavor_names]) - 1.5 * organic + rng.normal(0, 1.0, N)
ppm = (0.55 * np.array([brands[b] for b in brand_names]) * (1 + 0.35 * organic)
       * (1 - 0.0001 * (pack - 200)) * rng.normal(1, 0.06, N))      # price per 100 ml
price = np.round(ppm * pack / 100 * 10, 2)

rating = (np.array([flavors[f][2] for f in flavor_names]) + 0.25 * organic
          - 0.02 * (sugar - 10) + rng.normal(0, 0.35, N))
rating = np.clip(np.round(rating, 1), 1.0, 5.0)

summer = np.isin(month, [5, 6, 7, 8])
quantity = (180 + np.array([flavors[f][1] for f in flavor_names]) - 140 * (ppm - 0.6)
            + 2.2 * discount + 35 * (rating - 4) + 45 * summer - 25 * organic
            + rng.normal(0, 30, N))
quantity = np.clip(np.round(quantity), 5, None).astype(int)

df = pd.DataFrame({
    "product_name": [f"{b} {f}" for b, f in zip(brand_names, flavor_names)],
    "category": flavor_names,
    "pack_size_ml": pack,
    "price": price,
    "discount_pct": discount,
    "quantity_sold": quantity,
    "rating": rating,
    "sugar_g_per_100ml": np.round(sugar, 1),
    "is_organic": np.where(organic, "Yes", "No"),
    "month": month,
    "region": region,
})
df["revenue"] = np.round(df["price"] * (1 - df["discount_pct"] / 100) * df["quantity_sold"], 2)

# --- add the kind of problems real data has, so the cleaning step has work to do ---
def pick(k):
    return rng.choice(len(df), k, replace=False)

df.loc[pick(30), "rating"] = np.nan
df.loc[pick(22), "sugar_g_per_100ml"] = np.nan
df.loc[pick(12), "price"] = np.nan
df.loc[pick(8), "quantity_sold"] = np.nan
df.loc[pick(15), "revenue"] = np.nan
df.loc[pick(4), "rating"] = [9.5, 7.0, 0.0, 45.0]              # impossible ratings
df.loc[pick(4), "price"] = df["price"].median() * 10            # decimal-point typos
df.loc[pick(2), "quantity_sold"] = -20                          # negative quantities
for idx in pick(25):
    c = df.loc[idx, "category"]
    df.loc[idx, "category"] = rng.choice([c.lower(), c.upper(), " " + c + " "])
for idx in pick(30):
    df.loc[idx, "is_organic"] = rng.choice(["yes", "Y", "N", "no"])

df = pd.concat([df, df.sample(15, random_state=1)], ignore_index=True)   # duplicates
df = df.sample(frac=1, random_state=7).reset_index(drop=True)

out = Path(__file__).resolve().parents[1] / "data" / "juice_dataset.csv"
df.to_csv(out, index=False)
print(f"Saved {len(df)} rows to {out}")
