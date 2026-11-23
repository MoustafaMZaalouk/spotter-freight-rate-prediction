"""Paths and constants shared by the whole pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
REPORTS = ROOT / "reports"

SEED = 42

# Time-aware split of the labeled data (dates are inclusive, ISO strings).
TRAIN_END = "2025-07-31"   # fit on Jan-Jul
VALID_END = "2025-09-15"   # early stopping on Aug to mid-Sep
# Sep 16-Oct 31 is a retrospective holdout, already inspected in prior work.
# A 1-week gap is not needed because there are no rolling features, but we keep the
# holdout the last ~6 weeks because the real validation set is Nov-Dec 2025.

TARGET = "posted_rate"
WEIGHT_CAP = 47_500.0        # the weight column is clipped at +-47,500 in the raw data
