from __future__ import annotations

import numpy as np


def report(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = p - y
    ape = np.abs(err) / y
    ss_res, ss_tot = (err ** 2).sum(), ((y - y.mean()) ** 2).sum()
    return {
        "MAE": float(np.abs(err).mean()),
        "RMSE": float(np.sqrt((err ** 2).mean())),
        "MAPE_%": float(100 * ape.mean()),
        "MedianAPE_%": float(100 * np.median(ape)),
        "R2": float(1 - ss_res / ss_tot),
    }
