"""Generate the standalone report and README from metrics.json."""
import json
import numpy as np
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from . import config as C

m=json.loads((C.OUT/'metrics.json').read_text())
selected=m['selected_model']; r=m['results']; s=r[selected]['all_rows']; f=r['Full model']['all_rows']
folds=m['no_market_rolling_folds'] if selected=='Without market_index' else m['rolling_folds']
avg=lambda key: float(np.mean([x['all_rows'][key] for x in folds]))
sd=lambda key: float(np.std([x['all_rows'][key] for x in folds]))
shift=m['distribution']['market_index']
improved=sum(y['all_rows']['MAPE_%'] < x['all_rows']['MAPE_%'] for x,y in zip(m['rolling_folds'],m['no_market_rolling_folds']))
worsened=sum(y['all_rows']['MAPE_%'] > x['all_rows']['MAPE_%'] for x,y in zip(m['rolling_folds'],m['no_market_rolling_folds']))
expectation=f"Given variant selection on these windows, a cautious working expectation is closer to the full-feature level, about {f['MAPE_%']:.0f}% MAPE, than the selected {s['MAPE_%']:.2f}% holdout result. This is a planning judgment, not a measured hidden score or guarantee."
selection=f"The main reason to drop market_index is input shift: PSI {shift['PSI_nonmissing']:.2f}, with mean {shift['train']['mean']:.2f} in training versus {shift['validation']['mean']:.2f} in future inputs. Trees do not extrapolate outside learned ranges. Rolling comparisons support the choice, improving {improved} months and worsening {worsened}, but the overlapping windows are not independent evidence."
category_effect='; '.join(f"{name}: MAPE {r[name]['all_rows']['MAPE_%']:.2f}% versus full {f['MAPE_%']:.2f}%" for name in ['Without city/lane categories','Without weekday'])
ss=getSampleStyleSheet()
ss.add(ParagraphStyle(name='BodySmall',fontName='Helvetica',fontSize=9,leading=12,spaceAfter=5,textColor=colors.HexColor('#251f21')))
ss.add(ParagraphStyle(name='Section',fontName='Helvetica-Bold',fontSize=12,leading=15,spaceBefore=9,spaceAfter=5,textColor=colors.HexColor('#28766c')))
ss.add(ParagraphStyle(name='MainTitle',fontName='Helvetica-Bold',fontSize=20,leading=24,spaceAfter=9,textColor=colors.HexColor('#251f21')))
ss.add(ParagraphStyle(name='Cell',parent=ss['BodySmall'],fontSize=8,leading=10,spaceAfter=0))
story=[]
def p(text): story.append(Paragraph(text,ss['BodySmall']))
def h(text): story.append(Paragraph(text,ss['Section']))
def title(text): story.append(Paragraph(text,ss['MainTitle']))
def table(rows,widths=None):
    cells=[[Paragraph(str(x),ss['Cell']) for x in row] for row in rows]
    t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5efeb')),('VALIGN',(0,0),(-1,-1),'TOP'),('BOTTOMPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),5),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#28766c')),('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#eae9ea'))]))
    story.append(t); story.append(Spacer(1,6))
def metrics(v): return [f"{v['MAE']:.2f}",f"{v['MAPE_%']:.2f}",f"{v['MedianAPE_%']:.2f}"]
def footer(canvas,doc):
    canvas.setFont('Helvetica',8); canvas.setFillColor(colors.HexColor('#585254')); canvas.drawString(40,25,'Spotter | Freight rate prediction | Retrospective evaluation'); canvas.drawRightString(A4[0]-40,25,str(doc.page))

title('Freight rate prediction')
p(f"<b>All-row retrospective holdout:</b> MAE <b>${s['MAE']:.2f}</b>, MAPE <b>{s['MAPE_%']:.2f}%</b>, median APE <b>{s['MedianAPE_%']:.2f}%</b>. Selected variant: <b>{selected}</b>. These are not Spotter's hidden validation metrics.")
p(expectation)
p(f"Task: predict posted_rate in USD for {m['validation_rows']:,} future loads, using {m['rows']:,} labeled loads. The supplied scorer checks file structure and fixed-lane inputs; it does not measure predictive accuracy.")
h('1. Evaluation design and model')
p('Chronological split: fit through July 31, 2025; early stop on August 1 to September 15; evaluate every row from September 16 to October 31. These evaluation windows are also used to compare variants, so the holdout is not untouched. Rolling windows provide stability checks, not an independent final test. No random shuffle or claims of an unbiased final score.')
p(f"LightGBM predicts log(rate per mile), then exp(prediction) x distance. Huber loss, seed 42, fixed hyperparameters; no grid search. Early stopping monitors validation L1 and Huber on the log-rate-per-mile target, using <b>all</b> development rows and patience 100; either metric can trigger stopping, not dollar MAE. Each ablation gets its own early stopping. The final refit uses all relatively unflagged labeled rows and {m['final_trees']} trees. The tree count comes from the chronological early-stopping window and is kept fixed for the final refit.")
table([['All-row comparison','MAE $','MAPE %','MedAPE %'],['Full-feature model',*metrics(f)],['Selected variant',*metrics(s)],['Median RPM baseline',*metrics(r['Median RPM baseline']['all_rows'])],['Simple linear reference',*metrics(r['Simple linear reference']['all_rows'])]],[235,80,90,100])
h('2. Feature ablation: same dates and cleaning')
table([['Removed group','MAE $','MAPE %','MedAPE %']]+[[name,*metrics(r[name]['all_rows'])] for name in ['Full model','Without quote_signal','Without market_index','Without geography','Without weight','Without city/lane categories','Without weekday']],[235,80,90,100])
p('quote_signal removal also removes its distance interaction. Geography removal includes coordinates, cities, lane, haversine and road/straight ratio, but retains supplied road distance. Weight removal includes its flags. market_index removal includes its missing flag. These are refit experiments, not causal effects or proof of future value.')
p(selection)
p(category_effect+'. These groups were not combined or tuned further. Haversine is a correlated distance/geography proxy; gain ranking does not prove feature value.')
diag=r[selected]['label_filtered_diagnostic']
p(f"Secondary label-filtered diagnostic only: MAE ${diag['MAE']:.2f}, MAPE {diag['MAPE_%']:.2f}%, R2 {diag['R2']:.4f}. It removes {r[selected]['test_flagged']} holdout rows using actual labels and is not deployable or the headline. All-row RMSE is ${s['RMSE']:.2f}; R2 is {s['R2']:.4f}. Dollar R2 is strongly affected by trip length and extreme labels.")
story.append(PageBreak())
title('Stability and data checks')
h('3. Three rolling-origin windows, all rows')
table([['Fit through / stop month / test','Full MAPE %','No-market MAPE %','Selected MAE $']]+[[x['train_end']+' / '+x['early_stop_end'][5:7]+' / '+x['test_end'][5:7],f"{x['all_rows']['MAPE_%']:.2f}",f"{y['all_rows']['MAPE_%']:.2f}",f"{z['all_rows']['MAE']:.2f}"] for x,y,z in zip(m['rolling_folds'],m['no_market_rolling_folds'],folds)],[245,85,95,80])
p(f"Selected mean +/- population SD: MAE ${avg('MAE'):.2f} +/- ${sd('MAE'):.2f}; MAPE {avg('MAPE_%'):.2f}% +/- {sd('MAPE_%'):.2f} percentage points. The months vary materially. Later folds reuse earlier evaluation dates in training, and September/October overlap the retrospective holdout. These are stability checks, not independent confidence intervals.")
p(f"Selected-model rolling best-iteration range: {min(x['best_iteration'] for x in folds)}-{max(x['best_iteration'] for x in folds)} trees; final refit uses {m['final_trees']}. This variation and the single seed limit precision; no seed ensemble was evaluated.")
h('4. Train vs validation inputs')
rows=[['Input','Mean train / val','P05-P95 train','P05-P95 val','PSI']]
for c in ['quote_signal','market_index','distance']:
    q=m['distribution'][c]; a=q['train']; v=q['validation']
    rows.append([c,f"{a['mean']:.3f} / {v['mean']:.3f}",f"{a['p05']:.3f}-{a['p95']:.3f}",f"{v['p05']:.3f}-{v['p95']:.3f}",f"{q['PSI_nonmissing']:.3f}"])
table(rows,[90,115,110,110,80])
p('PSI uses ten training-quantile bins, excludes missing values, and floors empty-bin shares at 1e-6. It is descriptive and bin-sensitive, not a formal guarantee. The measured shifts and quantile ranges are shown above; similar marginal values do not rule out leakage or a changed relationship to the target. The meaning of quote_signal and its availability before a quote must be confirmed with the data owner.')
p('Equipment shares (train / validation): '+ '; '.join(f"{k} {v['train']:.2f}% / {v['validation']:.2f}%" for k,v in m['distribution']['equipment_mix_%'].items())+'.')
p('Missing values (train / validation): '+ '; '.join(f"{k}: {v['train']:,} / {v['validation']:,}" for k,v in m['quality']['missing'].items())+'. Unseen future categories against final fitting rows: '+', '.join(f"{k} {v:,}" for k,v in m['unseen_validation_categories'].items())+'. Unseen values become missing categories handled by LightGBM.')
h('5. Relative label filter and assumptions')
lf=m['label_filter']
p(f"Fit median rate per mile by equipment and distance band on the current training slice only; flag ratios below 0.5 or above 2.5, with training-global median fallback. Drop flagged <b>training</b> labels only; development and headline evaluation keep all rows. Final refit flags {lf['flagged_all_labeled']:,} of {m['rows']:,} labeled rows. Thresholds are fixed broad choices, not optimized or proof of corruption.")
table([['Distance band (mi)','Rows','Flagged','Flag share %']]+[[x['group'],f"{x['n']:,}",x['flagged'],f"{x['flagged_%']:.2f}"] for x in sorted(lf['groups']['distance_band'],key=lambda x:__import__('src.data',fromlist=['BAND_NAMES']).BAND_NAMES.index(x['group']))],[235,90,90,90])
e=lf['groups']['equipment']; months=lf['groups']['month']; city=lf['groups']['pickup'][0]
p(f"Low/high flagged median ratios are {lf['low_median_ratio']:.3f}x / {lf['high_median_ratio']:.3f}x. Flag share ranges from {min(x['flagged_%'] for x in e):.2f}% to {max(x['flagged_%'] for x in e):.2f}% across equipment and {min(x['flagged_%'] for x in months):.2f}% to {max(x['flagged_%'] for x in months):.2f}% across months. Highest pickup share among cities with at least 100 rows: {city['group']}, {city['flagged_%']:.2f}% ({city['flagged']}/{city['n']}). These descriptive differences do not establish whether flags are predictable. No claim that they are impossible to predict.")
p(f"Negative weight: {m['quality']['negative_weight']:,} rows. Taking absolute values is an unverified sign-error assumption; the original sign is flagged. Missing numeric values remain missing with flags. Weight at or above 47,500 lb is flagged ({m['quality']['weight_at_cap']:,} rows), not repaired.")
story.append(PageBreak())
title('December sensitivity and delivery')
h('6. Fixed-lane chart')
story.append(Image(str(C.REPORTS/'december_comparison.png'),width=505,height=183))
d=m['december']
p(f"Lexington to Fort Wayne, 360 miles, Dry Van, 32,000 lb. Solid line uses each December day's mean quote_signal and market_index from validation inputs only. Dashed line holds both at December validation means (quote_signal {d['quote_signal_fixed']:.4f}, market_index {d['market_index_fixed']:.4f}) while date varies. The selected model excludes market_index, so that column has no effect. Coordinates are looked up from labeled input features, not targets.")
p(f"Daily-input range: ${d['daily_min']:.2f}-${d['daily_max']:.2f}; fixed-input range: ${d['fixed_min']:.2f}-${d['fixed_max']:.2f}. The fixed-input line still varies with weekday on this lane; its repeating shape is a fitted calendar effect, not validated seasonality. Differences between lines are sensitivity to supplied daily input averages, not evidence of seasonal demand. The vertical axis is zoomed. Neither line is a validated December demand forecast.")
h('7. Selected model error by distance, every holdout row')
table([['Distance band (mi)','Rows','MAE $','MAPE %','MedAPE %']]+[[x['band'],x['n'],*metrics(x)] for x in m['holdout_distance_bands']],[160,75,90,90,90])
h('8. Reproducibility and limitations')
p('Metric tables and data-dependent comparisons are generated from outputs/metrics.json. Same-environment repeatability is recorded in outputs/repeatability.json. Requirements pin the versions used here; identical output is expected only in the same environment, not promised across machines or future libraries. The pipeline uses fixed seeds and four LightGBM threads.')
p(f"Final predictions contain {m['validation_rows']:,} unique load IDs and exactly load_id,predicted_rate. The unchanged supplied score.py validates predictions and the separate seven-column December output. It cannot validate the hidden score. The shareable archive excludes all files under data/ and includes .gitignore rules to prevent committing inputs.")
p('Remaining risks: quote_signal provenance and prediction-time availability; label-filter thresholds; clipped or missing weights; future market changes; unseen lanes; uncertainty is not modeled. Feature selection uses historical evaluation evidence. A fresh forward evaluation is needed before production use. The linear model is only a simple reference; the median-by-equipment/distance model is the main baseline.')
p('Evidence files: outputs/metrics.json (all diagnostics), outputs/december_sensitivity.csv (both chart lines), outputs/validation_predictions.csv, outputs/verification.json and tests/test_pipeline.py. Input datasets are excluded; README documents local reproduction.')
SimpleDocTemplate(str(C.REPORTS/'report.pdf'),pagesize=A4,leftMargin=40,rightMargin=40,topMargin=35,bottomMargin=38,title='Spotter freight rate prediction',author='').build(story,onFirstPage=footer,onLaterPages=footer)

readme=f'''# Freight rate prediction

Predict `posted_rate` in USD for freight loads. The model uses the **{selected}** LightGBM variant.

## Results, all rows first

Retrospective Sep 16-Oct 31 holdout: **MAE ${s['MAE']:.2f}, MAPE {s['MAPE_%']:.2f}%, median APE {s['MedianAPE_%']:.2f}%**.
Full-feature model: MAE ${f['MAE']:.2f}, MAPE {f['MAPE_%']:.2f}%.

{expectation}

{selection}
Selected three-fold rolling mean MAPE: {avg('MAPE_%'):.2f}% (population SD {sd('MAPE_%'):.2f} percentage points).
These windows informed model selection. They are not an untouched final test or Spotter's hidden validation score.

## Reproduce locally

Tested Python {m['environment']['python']}; see pinned `requirements.txt` and `outputs/metrics.json`.

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
- Ablations remove quote and its interaction, market and missing flag, geography, weight and flags, city/lane categories, and weekday. {selection} No further combinations were selected.
- quote_signal remains important but its definition is unknown. Its presence in validation does not prove it is available before quoting or free of leakage. Ask the data owner.
- Train/future input quantiles, PSI, equipment mix, missingness, unseen categories, flag shares and per-distance errors are generated in metrics.json. Gain importance alone is not proof of feature value. Geography and weight are not called useless.
- Negative weights use an absolute-value assumption and a sign flag. Missing values stay missing with flags. Unseen city/lane categories become missing for LightGBM.
- Median RPM by equipment/distance is the main baseline. Ridge is a simple linear reference with training-fitted fill values, not a strong tuned competitor.
- December chart compares daily validation input averages against fixed December averages. It is an input-sensitivity illustration, not a demand forecast. The scorer's own chart has one line; use reports/december_comparison.png for the two-line analysis.

## Deliverables

- `reports/report.pdf`: generated three-page report.
- `reports/december_comparison.png`: daily vs fixed inputs.
- `outputs/validation_predictions.csv`: exactly load_id,predicted_rate, {m['validation_rows']:,} rows.
- `outputs/december_chart_inputs.csv`: original seven-column format.
- `outputs/december_sensitivity.csv`: both chart lines.
- `outputs/metrics.json`: complete machine-generated evidence.

Input files are excluded and ignored by git. Prediction outputs remain included intentionally as the required submission artifact.
'''
(C.ROOT/'README.md').write_text(readme)
