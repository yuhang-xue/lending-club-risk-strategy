import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,roc_curve,brier_score_loss
from .common import score

def calibration(y,p,bins=10):
 y=np.asarray(y);p=np.asarray(p);bid=np.minimum((p*bins).astype(int),bins-1);rows=[]
 for i in range(bins):
  mask=bid==i;n=int(mask.sum())
  rows.append(dict(bin=i,lower=i/bins,upper=(i+1)/bins,n=n,mean_pd=float(p[mask].mean()) if n else np.nan,actual=float(y[mask].mean()) if n else np.nan))
 return rows

def metrics(y,p,prob=True):
 y=np.asarray(y);p=np.asarray(p);n=len(y);b=int(y.sum());r=dict(n=n,bad=b,bad_rate=b/n if n else np.nan,auc=np.nan,ks=np.nan,ks_risk_threshold=np.nan,brier=np.nan,ece=np.nan,mean_pd=np.nan)
 if 0<b<n:
  fp,tp,thr=roc_curve(y,p);j=np.argmax(tp-fp);r.update(auc=roc_auc_score(y,p),ks=float(tp[j]-fp[j]),ks_risk_threshold=float(thr[j]))
 if n and prob:
  r.update(brier=brier_score_loss(y,p),ece=sum(a['n']*abs(a['mean_pd']-a['actual']) for a in calibration(y,p) if a['n'])/n,mean_pd=p.mean())
 return r

def wilson(b,n):
 if not n:return np.nan,np.nan
 z=1.959963984540054;p=b/n;a=z*z/n;c=(p+a/2)/(1+a);h=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+a)
 return max(0,c-h),min(1,c+h)

def policy(y,p,s,threshold):
 y=np.asarray(y);p=np.asarray(p);keep=np.asarray(s)>=threshold;n=len(y);k=int(keep.sum());b=int(y.sum());bk=int(y[keep].sum());lo,hi=wilson(bk,k)
 return dict(n=n,accepted=k,accepted_bad=bk,acceptance=k/n if n else np.nan,actual_bad_rate=bk/k if k else np.nan,mean_pd=p[keep].mean() if k else np.nan,good_reject=((y==0)&~keep).sum()/(n-b) if n>b else np.nan,bad_capture=((y==1)&~keep).sum()/b if b else np.nan,bad_escape=bk/b if b else np.nan,wilson_lower=lo,wilson_upper=hi)

def select_thresholds(y,p,cfg):
 s=np.round(score(p),2);v,n=np.unique(s,return_counts=True);counts=np.cumsum(n[::-1]);thresholds=v[::-1];res=[]
 for target in cfg['targets']:
  ok=(counts/len(y)<=target)&(counts>=max(cfg['min_accepted_n'],cfg['min_accepted_share']*len(y)))
  if not ok.any():res.append(dict(target=target,threshold=np.nan,status='not_applicable_insufficient_support'));continue
  i=np.flatnonzero(ok)[-1];th=float(thresholds[i]);res.append(dict(target=target,threshold=th,status='selected',**policy(y,p,s,th)))
 return res

def psi(a,b,edges):
 a=np.histogram(a,edges)[0];b=np.histogram(b,edges)[0];p=(a+.5)/(a.sum()+.5*len(a));q=(b+.5)/(b.sum()+.5*len(b));parts=(q-p)*np.log(q/p)
 return float(parts.sum()),[dict(bin=i,reference_n=int(a[i]),current_n=int(b[i]),reference_share=p[i],current_share=q[i],component=parts[i]) for i in range(len(a))]
