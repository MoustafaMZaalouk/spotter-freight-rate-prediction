"""Model definitions. Target is log(rate per mile); predictions are mapped back to dollars."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from . import config as C


def to_target(y, distance):
    return np.log(y / distance)


def from_target(z, distance):
    return np.exp(z) * distance


class RateMedianBaseline:
    """Naive baseline: median $/mile by equipment and distance band times distance."""

    def fit(self, X, y, distance):
        band = pd.cut(distance, [0, 150, 300, 600, 1000, 2000, 5000])
        self.global_median = float((y / distance).median())
        self.tab = (y / distance).groupby([X["equipment"].astype(str), band], observed=True).median()
        return self

    def predict(self, X, distance):
        band = pd.cut(distance, [0, 150, 300, 600, 1000, 2000, 5000])
        key = list(zip(X["equipment"].astype(str), band))
        rpm = np.array([self.tab.get(k, np.nan) for k in key])
        return np.nan_to_num(rpm, nan=self.global_median) * distance


class RidgeModel:
    """Linear benchmark on log($/mile) with simple one-hot equipment and log-distance."""

    def fit(self, X, z):
        self.fill = X[["weight", "market_index", "quote_signal"]].median()
        A = self._design(X, fit=True)
        self.m = Ridge(alpha=1.0).fit(A, z)
        return self

    def _design(self, X, fit=False):
        A = pd.DataFrame({
            "log_d": X["log_distance"], "w": X["weight"].fillna(self.fill["weight"]) / 1e4,
            "mi": X["market_index"].fillna(self.fill["market_index"]), "q": X["quote_signal"],
            "q2": (X["quote_signal"] - 2.05) ** 2,
        })
        for e in ["Reefer", "Flatbed"]:
            A[e] = (X["equipment"].astype(str) == e).astype(int)
        return A

    def predict_z(self, X):
        return self.m.predict(self._design(X))


LGB_PARAMS = dict(
    objective="huber", alpha=0.3,   # robust on the log scale
    learning_rate=0.03, num_leaves=63, min_child_samples=30,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=5.0, cat_smooth=20, min_data_per_group=50, cat_l2=10,
    n_estimators=4000, random_state=C.SEED, deterministic=True, force_row_wise=True,
    verbose=-1, n_jobs=4,
)


def fit_lgb(Xtr, ztr, Xva=None, zva=None, n_estimators=None):
    params = dict(LGB_PARAMS)
    if n_estimators:
        params["n_estimators"] = n_estimators
    m = lgb.LGBMRegressor(**params)
    if Xva is not None:
        m.fit(Xtr, ztr, eval_set=[(Xva, zva)], eval_metric="l1",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    else:
        m.fit(Xtr, ztr)
    return m
