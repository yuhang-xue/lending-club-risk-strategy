import time
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit,logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,log_loss
import lightgbm as lgb
from .binning import Binner
from .common import CAT,now
from .metrics import metrics

class PositivePlatt:
 def fit(self,y,p):
  x=logit(np.clip(p,1e-9,1-1e-9));y=np.asarray(y)
  def objective(v):
   z=v[0]+v[1]*x
   loss=np.mean(np.logaddexp(0,z)-y*z)
   e=expit(z)-y
   return loss,np.array([e.mean(),(e*x).mean()])
  fit=minimize(objective,[0.,1.],method='L-BFGS-B',jac=True,bounds=[(None,None),(1e-6,None)],options={'maxiter':1000,'ftol':1e-12})
  assert fit.success,str(fit.message)
  self.intercept,self.slope=fit.x;return self
 def predict(self,p):return expit(self.intercept+self.slope*logit(np.clip(p,1e-9,1-1e-9)))

class Models:
 def __init__(self,cfg):self.cfg=cfg;self.calibrators={};self.trials=[];self.calibration_decisions=[]
 def tree_x(self,x):
  z=x.copy()
  for f,levels in self.categories.items():z[f]=pd.Categorical(z[f].replace('__MISSING__',np.nan),categories=levels)
  return z
 def fit(self,x,y,roles):
  cfg=self.cfg;tr=roles=='train';vm=roles=='V_model';vc=roles=='V_cal';self.fields=x.columns.tolist()
  start=time.perf_counter();self.binner=Binner(cfg['binning']).fit(x.loc[tr],y[tr]);z=self.binner.transform(x.loc[tr|vm]);self.selected=self.binner.selected
  ztr=z.loc[tr[tr|vm],self.selected];zv=z.loc[vm[tr|vm],self.selected]
  candidates=[]
  for c in cfg['lr_C']:
   m=LogisticRegression(C=c,solver='lbfgs',max_iter=2000,tol=1e-8,random_state=cfg['seed'])
   m.fit(ztr,y[tr]);assert m.n_iter_[0]<2000
   p=m.predict_proba(zv)[:,1];auc=roc_auc_score(y[vm],p);loss=log_loss(y[vm],p)
   self.trials.append(dict(model='LR',C=c,auc=auc,logloss=loss,n_iter=int(m.n_iter_[0])));candidates.append((auc,-loss,-c,m))
  self.lr=max(candidates,key=lambda a:a[:3])[-1];self.lr_seconds=time.perf_counter()-start
  self.categories={f:sorted(set(x.loc[tr,f])-{'__MISSING__'}) for f in CAT};tx=self.tree_x(x.loc[tr|vm]);candidates=[];start=time.perf_counter()
  for i,params in enumerate(cfg['lgb_trials']):
   m=lgb.LGBMClassifier(**params,n_estimators=cfg['lgb_max_rounds'],learning_rate=.05,verbosity=-1,n_jobs=4,random_state=cfg['seed'],deterministic=True,force_col_wise=True)
   m.fit(tx.loc[tr[tr|vm]],y[tr],eval_set=[(tx.loc[vm[tr|vm]],y[vm])],eval_metric='auc',callbacks=[lgb.early_stopping(cfg['lgb_early_stopping'],first_metric_only=True,verbose=False)])
   p=m.predict_proba(tx.loc[vm[tr|vm]])[:,1];auc=roc_auc_score(y[vm],p);loss=log_loss(y[vm],p)
   self.trials.append(dict(model='LightGBM',trial=i,auc=auc,logloss=loss,best_iteration=m.best_iteration_,**params));candidates.append((auc,-loss,-i,m))
  self.lgb=max(candidates,key=lambda a:a[:3])[-1];self.lgb_seconds=time.perf_counter()-start
  self.fico_median=float(x.loc[tr,'fico_mean'].median());assert np.isfinite(self.fico_median)
  for name in ['LR','LightGBM']:
   p=self.raw(name,x.loc[vm]);before=metrics(y[vm],p);trigger=before['ece']>cfg['calibration']['trigger_equal_width_ece']
   n=int(vc.sum());b=int(y[vc].sum());support=n>=cfg['calibration']['minimum_n'] and min(b,n-b)>=cfg['calibration']['minimum_class']
   decision=dict(model=name,V_model_ece=before['ece'],trigger=bool(trigger),supported=bool(support),method='positive_slope_Platt_on_logit' if trigger and support else 'none',decision_at=now(),fit_n=n if trigger and support else 0)
   if trigger and support:
    cal=PositivePlatt().fit(y[vc],self.raw(name,x.loc[vc]));self.calibrators[name]=cal;decision.update(intercept=cal.intercept,slope=cal.slope)
   self.calibration_decisions.append(decision)
  return self
 def raw(self,name,x):
  assert x.columns.tolist()==self.fields
  if name=='LR':return self.lr.predict_proba(self.binner.transform(x)[self.selected])[:,1]
  if name=='LightGBM':return self.lgb.predict_proba(self.tree_x(x))[:,1]
  if name=='FICO':return -x.fico_mean.fillna(self.fico_median).to_numpy()
  raise ValueError(name)
 def predict(self,name,x):
  p=self.raw(name,x)
  return self.calibrators[name].predict(p) if name in self.calibrators else p
