"""Reproducible evaluation, diagnostics and final inference. Run python -m src.pipeline."""
from __future__ import annotations
import json
import platform
import importlib.metadata
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from . import config as C, data as D, features as F, models as R, metrics as M

ABLATIONS = {
    'Full model': [],
    'Without quote_signal': ['quote_signal', 'quote_x_distance'],
    'Without market_index': ['market_index', 'market_index_missing'],
    'Without geography': ['haversine','distance_ratio','pickup_lat','pickup_lon','delivery_lat','delivery_lon','pickup','delivery','lane'],
    'Without weight': ['weight','weight_missing','weight_at_cap','weight_was_negative'],
    'Without city/lane categories': ['pickup','delivery','lane'],
    'Without weekday': ['dow','is_weekend'],
}

def evaluate(tr, va, ho, drop=None):
    filt = D.RelativeLabelFilter().fit(tr)
    trc = tr[~filt.flag(tr)]
    Xtr, cats = F.build(trc)
    Xva, _ = F.build(va, cats); Xho, _ = F.build(ho, cats)
    keep = [c for c in Xtr if c not in (drop or [])]
    model = R.fit_lgb(Xtr[keep], R.to_target(trc.posted_rate, trc.distance), Xva[keep], R.to_target(va.posted_rate, va.distance))
    p = R.from_target(model.predict(Xho[keep]), ho.distance.values)
    mask = ~filt.flag(ho).values
    result = {'all_rows': M.report(ho.posted_rate,p), 'label_filtered_diagnostic': M.report(ho.posted_rate[mask],p[mask]),
              'best_iteration': int(model.best_iteration_), 'train_rows': len(tr), 'train_flagged': int(filt.flag(tr).sum()),
              'test_rows': len(ho), 'test_flagged': int((~mask).sum())}
    return result, p, model, (Xtr,Xho,trc,cats), filt

def distribution(raw, val):
    result={}
    for c in ['quote_signal','market_index','distance']:
        def summary(d):
            return {'mean':float(d.mean()),'p05':float(d.quantile(.05)),'p50':float(d.median()),'p95':float(d.quantile(.95)),'missing_%':float(100*d.isna().mean())}
        a,b=raw[c],val[c]
        edges=np.unique(np.r_[-np.inf,a.quantile(np.linspace(.1,.9,9)).values,np.inf])
        pa=pd.cut(a,edges).value_counts(sort=False,normalize=True).values
        pb=pd.cut(b,edges).value_counts(sort=False,normalize=True).values
        pa,pb=np.maximum(pa,1e-6),np.maximum(pb,1e-6)
        result[c]={'train':summary(a),'validation':summary(b),'PSI_nonmissing':float(np.sum((pb-pa)*np.log(pb/pa)))}
    result['equipment_mix_%']={k:{'train':float(100*(raw.equipment==k).mean()),'validation':float(100*(val.equipment==k).mean())} for k in sorted(set(raw.equipment)|set(val.equipment))}
    return result

def flag_diagnostics(df, filt):
    frame=df.copy(); frame['flag']=filt.flag(df); frame['ratio']=filt.ratio(df)
    frame['distance_band']=D.bands(df); frame['month']=df.date.dt.strftime('%Y-%m')
    groups={}
    for c in ['distance_band','equipment','month','pickup','delivery']:
        t=frame.groupby(c,observed=True).flag.agg(['size','sum','mean'])
        if c in ['pickup','delivery']: t=t[t['size']>=100].sort_values('mean',ascending=False).head(5)
        groups[c]=[{'group':str(k),'n':int(v['size']),'flagged':int(v['sum']),'flagged_%':float(v['mean']*100)} for k,v in t.iterrows()]
    low=frame.loc[frame.ratio<.5,'ratio']; high=frame.loc[frame.ratio>2.5,'ratio']
    return {'groups':groups,'low_n':len(low),'high_n':len(high),'low_median_ratio':float(low.median()),'high_median_ratio':float(high.median())}

def output_predictions(val,p,template):
    pred=pd.DataFrame({'load_id':val.load_id,'predicted_rate':np.round(p,2)})
    if set(pred.load_id)!=set(template.load_id) or template.load_id.duplicated().any(): raise ValueError('Template ID mismatch')
    out=template[['load_id']].merge(pred,on='load_id',how='left',validate='one_to_one')
    if len(out)!=12000 or not np.isfinite(out.predicted_rate).all() or (out.predicted_rate<=0).any(): raise ValueError('Invalid predictions')
    return out

def main():
    C.OUT.mkdir(exist_ok=True); C.REPORTS.mkdir(exist_ok=True)
    raw=D.load(C.DATA/'train_test.csv',labeled=True); val_raw=D.load(C.DATA/'validation.csv')
    df=D.clean_features(raw); val=D.clean_features(val_raw)
    tr,va,ho=D.time_split(df)
    results={}; predictions={}
    for name,drop in ABLATIONS.items():
        r,p,model,parts,filt=evaluate(tr,va,ho,drop)
        results[name]=r; predictions[name]=p
        if name=='Full model': full_model,full_parts,main_filter=model,parts,filt
        print(name, r['all_rows'],flush=True)
    Xtr,Xho,trc,cats=full_parts
    base=R.RateMedianBaseline().fit(Xtr,trc.posted_rate,trc.distance)
    ridge=R.RidgeModel().fit(Xtr,R.to_target(trc.posted_rate,trc.distance))
    for name,p in [('Median RPM baseline',base.predict(Xho,ho.distance)),('Simple linear reference',R.from_target(ridge.predict_z(Xho),ho.distance.values))]:
        results[name]={'all_rows':M.report(ho.posted_rate,p)}
    folds=[]; no_market_folds=[]
    for train_end,val_end,test_end in [('2025-06-30','2025-07-31','2025-08-31'),('2025-07-31','2025-08-31','2025-09-30'),('2025-08-31','2025-09-30','2025-10-31')]:
        a=df[df.date<=train_end]; b=df[(df.date>train_end)&(df.date<=val_end)]; c=df[(df.date>val_end)&(df.date<=test_end)]
        r,*_=evaluate(a,b,c)
        folds.append({'train_end':train_end,'early_stop_end':val_end,'test_end':test_end,**r})
        nr,*_=evaluate(a,b,c,ABLATIONS['Without market_index'])
        no_market_folds.append({'train_end':train_end,'early_stop_end':val_end,'test_end':test_end,**nr})
        print('Rolling',test_end,r['all_rows'],'no market',nr['all_rows'],flush=True)
    rolling_summary={k:{'mean':float(np.mean([f['all_rows'][k] for f in folds])), 'std':float(np.std([f['all_rows'][k] for f in folds])), 'min':float(min(f['all_rows'][k] for f in folds)), 'max':float(max(f['all_rows'][k] for f in folds))} for k in folds[0]['all_rows']}
    selected='Without market_index' if np.mean([x['all_rows']['MAPE_%'] for x in no_market_folds]) < np.mean([x['all_rows']['MAPE_%'] for x in folds]) else 'Full model'
    selected_drop=ABLATIONS[selected]
    global_filter=D.RelativeLabelFilter().fit(df); allc=df[~global_filter.flag(df)]
    Xall,cats=F.build(allc); keep=[c for c in Xall if c not in selected_drop]
    final=R.fit_lgb(Xall[keep],R.to_target(allc.posted_rate,allc.distance),n_estimators=results[selected]['best_iteration'])
    Xv,_=F.build(val,cats); pv=R.from_target(final.predict(Xv[keep]),val.distance.values)
    out=output_predictions(val,pv,pd.read_csv(C.DATA/'validation_predictions_template.csv'))
    out.to_csv(C.OUT/'validation_predictions.csv',index=False)
    unseen={c:int(Xv[c].isna().sum()) for c in F.CATEGORICAL}
    dec=pd.read_csv(C.DATA/'december_chart_inputs.csv'); dec.date=pd.to_datetime(dec.date)
    coords=pd.concat([raw[['pickup','pickup_lat','pickup_lon']].rename(columns=lambda x:x.replace('pickup','city')),raw[['delivery','delivery_lat','delivery_lon']].rename(columns=lambda x:x.replace('delivery','city'))]).drop_duplicates('city').set_index('city')
    d=dec.drop(columns='predicted_rate').copy()
    for side in ['pickup','delivery']:
        for xy in ['lat','lon']: d[f'{side}_{xy}']=coords.loc[d[side],f'city_{xy}'].values
    daily=val.groupby('date')[['market_index','quote_signal']].mean()
    d=d.join(daily,on='date')
    if d[['market_index','quote_signal']].isna().any().any(): raise ValueError('Missing December daily inputs')
    xd,_=F.build(D.clean_features(d),cats)
    daily_p=R.from_target(final.predict(xd[keep]),d.distance.values)
    fixed=d.copy(); dec_val=val[val.date.dt.month==12]
    for c in ['market_index','quote_signal']: fixed[c]=dec_val[c].mean()
    xf,_=F.build(D.clean_features(fixed),cats)
    fixed_p=R.from_target(final.predict(xf[keep]),fixed.distance.values)
    dec['predicted_rate']=np.round(daily_p,2); dec.date=dec.date.dt.strftime('%Y-%m-%d')
    dec.to_csv(C.OUT/'december_chart_inputs.csv',index=False)
    pd.DataFrame({'date':dec.date,'daily_inputs_rate':np.round(daily_p,2),'fixed_inputs_rate':np.round(fixed_p,2)}).to_csv(C.OUT/'december_sensitivity.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(10.5,3.8)); dates=pd.to_datetime(dec.date)
    ax.plot(dates,daily_p,color='#28766c',lw=2,label='Daily validation input means')
    ax.plot(dates,fixed_p,color='#585254',ls='--',lw=2,label='Fixed December input means (date varies)')
    ax.set(title='December 2025: input sensitivity, not a demand forecast',ylabel='Predicted rate (USD)',xlabel='Lexington to Fort Wayne | 360 mi | Dry Van | 32,000 lb')
    ax.grid(axis='y',alpha=.2); ax.legend(loc='best',fontsize=9); fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(C.REPORTS/'december_comparison.png',dpi=180); plt.close(fig)
    by_band=[]; hp=predictions[selected]
    for band in D.BAND_NAMES:
        mask=D.bands(ho).eq(band).values
        if mask.any(): by_band.append({'band':band,'n':int(mask.sum()),**M.report(ho.posted_rate[mask],hp[mask])})
    imp=pd.Series(full_model.booster_.feature_importance('gain'),index=Xtr.columns); imp=(100*imp/imp.sum()).sort_values(ascending=False)
    info={'rows':len(raw),'validation_rows':len(val),'split':{k:[str(v.date.min().date()),str(v.date.max().date()),len(v)] for k,v in [('train',tr),('early_stop',va),('holdout',ho)]},
          'results':results,'selected_model':selected,'no_market_rolling_folds':no_market_folds,'rolling_folds':folds,'rolling_summary':rolling_summary,'distribution':distribution(raw,val_raw),
          'label_filter':{'thresholds':[.5,2.5],'flagged_all_labeled':int(global_filter.flag(df).sum()),**flag_diagnostics(df,global_filter)},
          'quality':{'negative_weight':int((raw.weight<0).sum()),'weight_at_cap':int((raw.weight.abs()>=C.WEIGHT_CAP).sum()),'missing':{c:{'train':int(raw[c].isna().sum()),'validation':int(val_raw[c].isna().sum())} for c in ['weight','market_index']},'duplicate_rows':int(raw.duplicated().sum())},
          'unseen_validation_categories':unseen,'holdout_distance_bands':by_band,'feature_gain_%':imp.to_dict(),
          'december':{'daily_min':float(min(daily_p)),'daily_max':float(max(daily_p)),'fixed_min':float(min(fixed_p)),'fixed_max':float(max(fixed_p)),'market_index_fixed':float(fixed.market_index.iloc[0]),'quote_signal_fixed':float(fixed.quote_signal.iloc[0])},
          'environment':{'python':platform.python_version(),**{x:importlib.metadata.version(x) for x in ['numpy','pandas','lightgbm','scikit-learn','matplotlib','reportlab']}},
          'final_trees':results[selected]['best_iteration'],'validation_prediction_range':[float(out.predicted_rate.min()),float(out.predicted_rate.max())]}
    (C.OUT/'metrics.json').write_text(json.dumps(info,indent=2,allow_nan=False))
    print('Wrote metrics and predictions.',flush=True)

if __name__=='__main__': main()
