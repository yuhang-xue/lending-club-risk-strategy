import unittest
import numpy as np
import pandas as pd
from src.binning import Binner
from src.common import score,probability
from src.metrics import select_thresholds,policy,psi,metrics
from src.model import PositivePlatt
class Core(unittest.TestCase):
 def test_score_anchor_roundtrip(self):
  self.assertAlmostEqual(score([1/51])[0],600)
  self.assertAlmostEqual(score([1/101])[0],620)
  p=np.linspace(.000001,.999999,10000)
  np.testing.assert_allclose(probability(score(p)),p,atol=1e-12)
  self.assertTrue((np.diff(score(p))<0).all())
 def test_tied_policy_and_empty(self):
  p=np.array([.1]*6+[.2]*4);y=np.array([0]*7+[1]*3)
  cfg=dict(targets=[.5,.7,.9],min_accepted_n=1,min_accepted_share=0)
  rows=select_thresholds(y,p,cfg)
  self.assertEqual(rows[0]['status'],'not_applicable_insufficient_support')
  self.assertEqual(rows[1]['accepted'],6);self.assertEqual(rows[2]['accepted'],6)
  r=policy(y,p,score(p),1000);self.assertTrue(np.isnan(r['actual_bad_rate']))
 def test_woe_train_monotonic_missing_unknown(self):
  rng=np.random.default_rng(1);v=np.repeat(np.arange(20),100);y=rng.binomial(1,.02+.04*v)
  x=pd.DataFrame({'dti':v.astype(float),'purpose':np.where(v<10,'a','b')});x.loc[:70,'dti']=np.nan
  cfg=dict(prebins=20,max_bins=5,min_share=.05,iv_min=.02,correlation=.7)
  b=Binner(cfg).fit(x,y)
  rows=[r for r in b.rows if r['field']=='dti' and r['bin'].startswith('B')]
  self.assertLessEqual(len(rows),5);self.assertTrue(all(r['share']>=.05 for r in rows));self.assertTrue((np.diff([r['bad_rate'] for r in rows])>=-1e-12).all())
  future=pd.DataFrame({'dti':[-999.,999.,np.nan],'purpose':['new','a','b']})
  z=b.transform(future);self.assertEqual(z.purpose.iloc[0],0);self.assertTrue(np.isfinite(z).all().all())
  # Future rows cannot change fitted transformations.
  before=b.transform(x);b.transform(future);pd.testing.assert_frame_equal(before,b.transform(x))
  for f in x:
   rs=[r for r in b.rows if r['field']==f and r['n']>0];g=sum(r['good'] for r in rs);bad=sum(r['bad'] for r in rs);k=len(rs)
   for r in rs:self.assertAlmostEqual(r['woe'],np.log(((r['good']+.5)/(g+.5*k))/((r['bad']+.5)/(bad+.5*k))))
 def test_psi_zero_bins(self):
  v,_=psi(np.array([0,0]),np.array([0,0]),[-np.inf,1,np.inf]);self.assertEqual(v,0)
  v,_=psi(np.array([0,0]),np.array([2,2]),[-np.inf,1,np.inf]);self.assertTrue(np.isfinite(v));self.assertGreater(v,0)
 def test_positive_calibration(self):
  p=np.linspace(.01,.99,2000);y=(np.arange(2000)%7<np.floor(p*7)).astype(int)
  c=PositivePlatt().fit(y,p);self.assertGreater(c.slope,0);self.assertTrue((np.diff(c.predict(p))>0).all())
 def test_ks_ties(self):
  self.assertEqual(metrics([0,1,0,1],[.5]*4)['ks'],0)
if __name__=='__main__':unittest.main()
