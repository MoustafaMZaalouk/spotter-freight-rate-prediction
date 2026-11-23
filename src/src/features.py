"""Feature engineering. Everything here is computed from a single row (no leakage)."""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC = [
    "distance", "log_distance", "haversine", "distance_ratio", "weight",
    "market_index", "quote_signal", "quote_x_distance",
    "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon",
    "dow", "is_weekend", "weight_was_negative", "weight_at_cap",
    "market_index_missing", "weight_missing", "short_haul",
]
CATEGORICAL = ["equipment", "pickup", "delivery", "lane"]


def haversine(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(a))


def build(df: pd.DataFrame, categories: dict | None = None):
    """Return (X, categories). Pass the train `categories` to encode other sets."""
    X = pd.DataFrame(index=df.index)
    X["distance"] = df["distance"]
    X["log_distance"] = np.log(df["distance"])
    X["haversine"] = haversine(df.pickup_lat, df.pickup_lon, df.delivery_lat, df.delivery_lon)
    X["distance_ratio"] = df["distance"] / X["haversine"].clip(lower=1)
    X["short_haul"] = (df["distance"] < 300).astype(int)
    for c in ["weight", "market_index", "quote_signal", "pickup_lat", "pickup_lon",
              "delivery_lat", "delivery_lon", "weight_was_negative", "weight_at_cap",
              "market_index_missing", "weight_missing"]:
        X[c] = df[c]
    X["quote_x_distance"] = df["quote_signal"] * df["distance"]
    # Day of week only. Month / day-of-year are deliberately NOT used: the test period
    # (Nov-Dec) lies after all training dates, so a calendar feature cannot be learned.
    X["dow"] = df["date"].dt.dayofweek
    X["is_weekend"] = (X["dow"] >= 5).astype(int)
    X["equipment"] = df["equipment"]
    X["pickup"] = df["pickup"]
    X["delivery"] = df["delivery"]
    X["lane"] = df["pickup"] + ">" + df["delivery"]
    if categories is None:
        categories = {c: sorted(X[c].unique()) for c in CATEGORICAL}
    for c in CATEGORICAL:
        X[c] = pd.Categorical(X[c], categories=categories[c])
    return X[NUMERIC + CATEGORICAL], categories
