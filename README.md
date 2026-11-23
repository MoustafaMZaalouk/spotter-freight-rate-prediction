# Freight rate prediction

Predict `posted_rate` in USD for freight loads. The model uses the **Without market_index** LightGBM variant.

**Input data is not included.** Full reproduction requires the original assessment files from an authorized source. See the [report PDF](reports/report.pdf) for results, assumptions and limits. Published predictions are assessment artifacts, not evidence of a production deployment.

## Results, all rows first

Retrospective Sep 16-Oct 31 holdout: **MAE $87.24, MAPE 3.93%, median APE 1.08%**.
Full-feature model: MAE $117.37, MAPE 5.23%.

Given variant selection on these windows, a cautious working expectation is closer to the full-feature level, about 5% MAPE, than the selected 3.93% holdout result. This is a planning judgment, not a measured hidden score or guarantee.

The main reason to drop market_index is input shift: PSI 5.10, with mean 1.08 in training versus 0.93 in future inputs. Trees do not extrapolate outside learned ranges. Rolling comparisons support the choice, improving 2 months and worsening 1, but the overlapping windows are not independent evidence.
Selected three-fold rolling mean MAPE: 5.17% (population SD 2.08 percentage points).
These windows informed model selection. They are not an untouched final test or Spotter's hidden validation score.

## Reproduce locally

Tested Python 3.10.12; see pinned `requirements.txt` and `outputs/metrics.json`.

Input datasets are NOT included. Copy the four supplied files into `data/`:
- `train_test.csv` (if supplied as train-test.csv, rename it locally)
- `validation.csv`
- `validation_predictions_template.csv`
- `december_chart_inputs.csv`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m src.pipeline
python -m src.make_report
python score.py --predictions outputs/validation_predictions.csv --december-predictions outputs/december_chart_inputs.csv --output-dir outputs/scorer_results
python -m src.verify
```

`src.verify` runs tests and the unchanged real scorer, checks exact IDs/column order, and writes `outputs/verification.json`. It does not evaluate hidden labels. Seeds and four LightGBM threads are fixed. Same-environment repeatability is checked; byte identity across machines is not guaranteed.

## Method and honesty notes

- Log(rate/mile) target, Huber loss, LightGBM with fixed hyperparameters. No hyperparameter grid search. Early stopping monitors validation L1 and Huber on the log-rate-per-mile target, using all development rows and patience 100; either metric can trigger stopping. Final refit uses the selected early-stopped tree count without a multiplier.
- Relative label filter: training-only medians by equipment and distance band; flag RPM/median below 0.5 or above 2.5. Remove only from training, never headline evaluation. Suspicious is not proven corrupt. Label-filtered diagnostics are secondary and cannot be used at prediction time.
- Ablations remove quote and its interaction, market and missing flag, geography, weight and flags, city/lane categories, and weekday. The main reason to drop market_index is input shift: PSI 5.10, with mean 1.08 in training versus 0.93 in future inputs. Trees do not extrapolate outside learned ranges. Rolling comparisons support the choice, improving 2 months and worsening 1, but the overlapping windows are not independent evidence. No further combinations were selected.
- quote_signal remains important but its definition is unknown. Its presence in validation does not prove it is available before quoting or free of leakage. Ask the data owner.
- Train/future input quantiles, PSI, equipment mix, missingness, unseen categories, flag shares and per-distance errors are generated in metrics.json. Gain importance alone is not proof of feature value. Geography and weight are not called useless.
- Negative weights use an absolute-value assumption and a sign flag. Missing values stay missing with flags. Unseen city/lane categories become missing for LightGBM.
- Median RPM by equipment/distance is the main baseline. Ridge is a simple linear reference with training-fitted fill values, not a strong tuned competitor.
- December chart compares daily validation input averages against fixed December averages. It is an input-sensitivity illustration, not a demand forecast. The scorer's own chart has one line; use reports/december_comparison.png for the two-line analysis.

## Deliverables

- `reports/report.pdf`: generated three-page report.
- `reports/december_comparison.png`: daily vs fixed inputs.
- `outputs/validation_predictions.csv`: exactly load_id,predicted_rate, 12,000 rows.
- `outputs/december_chart_inputs.csv`: original seven-column format.
- `outputs/december_sensitivity.csv`: both chart lines.
- `outputs/metrics.json`: complete machine-generated evidence.

Input files are excluded and ignored by git. Prediction outputs remain included intentionally as the required submission artifact.
