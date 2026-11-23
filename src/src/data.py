"""Validated inputs and training-fitted relative label diagnostics."""
from __future__ import annotations
import numpy as np
import pandas as pd
from . import config as C

BINS = [0, 150, 300, 600, 1000, 2000, np.inf]
BAND_NAMES = ['<150', '150-300', '300-600', '600-1000', '1000-2000', '2000+']
REQUIRED = ['load_id','pickup','delivery','pickup_lat','pickup_lon','delivery_lat','delivery_lon','distance','equipment','weight','date','market_index','quote_signal']

def validate(df, labeled=False):
    required = REQUIRED + ([C.TARGET] if labeled else [])
    missing = set(required) - set(df.columns)
    if missing: raise ValueError(f'Missing columns: {sorted(missing)}')
    if df.load_id.isna().any() or df.load_id.duplicated().any(): raise ValueError('IDs must be present and unique')
    if df.date.isna().any(): raise ValueError('Invalid dates')
    for c in ['distance','pickup_lat','pickup_lon','delivery_lat','delivery_lon','quote_signal'] + ([C.TARGET] if labeled else []):
        if not np.isfinite(pd.to_numeric(df[c], errors='coerce')).all(): raise ValueError(f'Invalid {c}')
    if (df.distance <= 0).any(): raise ValueError('Distance must be positive')
    if labeled and (df[C.TARGET] <= 0).any(): raise ValueError('Target must be positive')
    for c in ['weight','market_index']:
        v = pd.to_numeric(df[c], errors='raise')
        if np.isinf(v).any(): raise ValueError(f'Infinite {c}')
    for c in ['pickup','delivery','equipment']:
        if df[c].isna().any(): raise ValueError(f'Missing {c}')
    for c in ['pickup_lat','delivery_lat']:
        if not df[c].between(-90,90).all(): raise ValueError(f'Invalid {c}')
    for c in ['pickup_lon','delivery_lon']:
        if not df[c].between(-180,180).all(): raise ValueError(f'Invalid {c}')

def load(path, labeled=False):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    validate(df, labeled)
    return df

def clean_features(df):
    df = df.copy()
    # Assumption, not a verified correction: preserve magnitude and sign flag.
    df['weight_was_negative'] = (df.weight < 0).astype(int)
    df['weight'] = df.weight.abs()
    df['weight_at_cap'] = (df.weight >= C.WEIGHT_CAP).astype(int)
    df['market_index_missing'] = df.market_index.isna().astype(int)
    df['weight_missing'] = df.weight.isna().astype(int)
    return df

def bands(df):
    return pd.cut(df.distance, BINS, labels=BAND_NAMES).astype(str)

class RelativeLabelFilter:
    """Fit expected RPM medians on the training slice only; never repair labels.

    Thresholds 0.5 and 2.5 are broad, fixed diagnostic choices, not optimized.
    A flagged row is suspicious, not proven corrupt. Evaluation retains every row.
    """
    def fit(self, df):
        rpm = df[C.TARGET] / df.distance
        self.global_median = float(rpm.median())
        self.table = rpm.groupby([df.equipment, bands(df)], observed=True).median().to_dict()
        return self
    def ratio(self, df):
        expected = np.array([self.table.get(k, self.global_median) for k in zip(df.equipment, bands(df))])
        return (df[C.TARGET] / df.distance) / expected
    def flag(self, df):
        r = self.ratio(df)
        return (r < 0.5) | (r > 2.5)

def time_split(df):
    return (df[df.date <= C.TRAIN_END], df[(df.date > C.TRAIN_END) & (df.date <= C.VALID_END)], df[df.date > C.VALID_END])
