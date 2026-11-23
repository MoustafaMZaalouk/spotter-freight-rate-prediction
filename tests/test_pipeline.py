"""Small engineering checks. Run python -m unittest discover -s tests -v."""
import unittest
import numpy as np
import pandas as pd
from src import data as D, features as F, models as R, metrics as M
from src.pipeline import output_predictions

class PipelineTests(unittest.TestCase):
    def toy(self):
        return pd.DataFrame({'load_id':['a','b','c','d'], 'pickup':['A']*4,'delivery':['B']*4,
          'pickup_lat':[30.]*4,'pickup_lon':[-90.]*4,'delivery_lat':[31.]*4,'delivery_lon':[-89.]*4,
          'distance':[100.]*4,'equipment':['Dry Van']*4,'weight':[-1000.,2000.,3000.,4000.],
          'date':pd.to_datetime(['2025-01-01']*4),'market_index':[1.]*4,'quote_signal':[2.]*4,'posted_rate':[300.,300.,300.,3000.]})
    def test_features_and_unseen(self):
        df=self.toy(); D.validate(df,True); x,c=F.build(D.clean_features(df))
        self.assertFalse(x.isna().any().any()); self.assertTrue(np.isfinite(x.select_dtypes('number')).all().all())
        self.assertEqual(x.weight.iloc[0],1000.)
        df.loc[0,'pickup']='NEW'; z,_=F.build(D.clean_features(df),c)
        self.assertTrue(pd.isna(z.pickup.iloc[0])); self.assertTrue(pd.isna(z.lane.iloc[0]))
    def test_filter_is_training_fitted(self):
        df=self.toy(); filt=D.RelativeLabelFilter().fit(df.iloc[:3]); self.assertEqual(filt.flag(df).tolist(),[False,False,False,True])
        df.loc[3,'posted_rate']=1; self.assertEqual(filt.global_median,3.)
    def test_schema_errors(self):
        df=self.toy(); df.loc[0,'distance']=0
        with self.assertRaises(ValueError): D.validate(df,True)
        with self.assertRaises(ValueError): D.validate(self.toy().drop(columns='date'),True)
    def test_target_round_trip(self):
        y=np.array([200.,3000.]); d=np.array([100.,1500.]); np.testing.assert_allclose(R.from_target(R.to_target(y,d),d),y)
    def test_submission_order_and_ids(self):
        ids=[f'TE-{x:06d}' for x in range(1,12001)]; val=pd.DataFrame({'load_id':ids[::-1]}); tmpl=pd.DataFrame({'load_id':ids})
        out=output_predictions(val,np.full(12000,100.),tmpl)
        self.assertEqual(list(out.columns),['load_id','predicted_rate']); self.assertEqual(out.load_id.tolist(),ids)
        with self.assertRaises(ValueError): output_predictions(val,np.full(12000,np.nan),tmpl)
    def test_missing_values_supported(self):
        df=self.toy(); df.loc[0,'market_index']=np.nan; df.loc[1,'weight']=np.nan
        D.validate(df,True); x,_=F.build(D.clean_features(df))
        self.assertEqual(x.market_index_missing.iloc[0],1); self.assertEqual(x.weight_missing.iloc[1],1)

    def test_split_boundaries(self):
        df=self.toy()
        df['date']=pd.to_datetime(['2025-07-31','2025-08-01','2025-09-15','2025-09-16'])
        tr,va,ho=D.time_split(df)
        self.assertEqual(tr.index.tolist(),[0])
        self.assertEqual(va.index.tolist(),[1,2])
        self.assertEqual(ho.index.tolist(),[3])
        self.assertTrue((tr.date<=pd.Timestamp('2025-07-31')).all())
        self.assertTrue((ho.date>pd.Timestamp('2025-09-15')).all())
        self.assertEqual(len(tr)+len(va)+len(ho),len(df))
    def test_metrics_hand_computed(self):
        # Actual [100,200], prediction [110,180]: abs errors 10,20.
        # MAE 15, MSE 250, APEs 10%,10%, SSE 500, SST 5000.
        r=M.report([100.,200.],[110.,180.])
        expected={'MAE':15.,'RMSE':np.sqrt(250.),'MAPE_%':10.,'MedianAPE_%':10.,'R2':.9}
        for key,value in expected.items(): self.assertAlmostEqual(r[key],value)
    def test_unseen_category_predicts_finite(self):
        df=pd.concat([self.toy()]*30,ignore_index=True)
        x,cats=F.build(D.clean_features(df))
        model=R.fit_lgb(x,R.to_target(df.posted_rate,df.distance),n_estimators=5)
        future=self.toy(); future['pickup']='UNSEEN'; future['delivery']='NEW'
        z,_=F.build(D.clean_features(future),cats)
        self.assertTrue(z.pickup.isna().all()); self.assertTrue(z.lane.isna().all())
        pred=R.from_target(model.predict(z),future.distance)
        self.assertTrue(np.isfinite(pred).all()); self.assertTrue((pred>0).all())

if __name__=='__main__': unittest.main()
