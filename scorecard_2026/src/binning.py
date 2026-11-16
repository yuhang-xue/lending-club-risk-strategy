"""Deterministic supervised adjacent merging; only training observations fit WOE."""
import numpy as np
import pandas as pd
from .common import CAT
DIRECTIONS={'fico_mean':-1,'dti':1,'revol_util':1,'inq_last_6mths':1}
def stats_iv(groups,missing):
 rows=[g for g in groups+[missing] if g['n']>0]
 if not rows:return 0.
 n=np.array([g['n'] for g in rows]);b=np.array([g['b'] for g in rows]);good=n-b;k=len(rows)
 pg=(good+.5)/(good.sum()+.5*k);pb=(b+.5)/(b.sum()+.5*k)
 return float(np.sum((pg-pb)*np.log(pg/pb)))
def merged(groups,i):
 a,b=groups[i:i+2]
 return groups[:i]+[dict(keys=a['keys']+b['keys'],n=a['n']+b['n'],b=a['b']+b['b'])]+groups[i+2:]
def merge(groups,missing,n,direction,cfg):
 groups=[dict(g,keys=list(g['keys'])) for g in groups];log=[]
 while len(groups)>1:
  small={i for i,g in enumerate(groups) if g['n']<cfg['min_share']*n}
  bad=[i for i in range(len(groups)-1) if direction*(groups[i+1]['b']/max(groups[i+1]['n'],1)-groups[i]['b']/max(groups[i]['n'],1)) < -1e-12]
  if small:candidates=[i for i in range(len(groups)-1) if i in small or i+1 in small];reason='small_bin'
  elif direction and bad:candidates=bad;reason='monotonicity'
  elif len(groups)>cfg['max_bins']:candidates=list(range(len(groups)-1));reason='max_bins'
  else:break
  iv=stats_iv(groups,missing)
  i=min(candidates,key=lambda i:(iv-stats_iv(merged(groups,i),missing),i))
  new=merged(groups,i);log.append(dict(reason=reason,left_keys=str(groups[i]['keys']),right_keys=str(groups[i+1]['keys']),iv_loss=iv-stats_iv(new,missing),direction=direction))
  groups=new
 return groups,log
class Binner:
 def __init__(self,cfg):self.cfg=cfg
 def fit(self,x,y):
  y=np.asarray(y,int);assert set(np.unique(y))=={0,1}
  self.spec={};self.rows=[];self.log=[];self.selection=[]
  for f in x:
   s=x[f];categorical=f in CAT
   missing=s.eq('__MISSING__').to_numpy() if categorical else s.isna().to_numpy()
   m=dict(keys=['MISSING'],n=int(missing.sum()),b=int(y[missing].sum()))
   if categorical:
    counts=s[~missing].value_counts();rare=set(counts[counts<self.cfg['min_share']*len(y)].index)
    reduced=s.mask(s.isin(rare),'__OTHER__')
    keys=sorted(set(reduced[~missing]));raw={k:dict(keys=[k],n=int((reduced.eq(k)&~missing).sum()),b=int(y[(reduced.eq(k)&~missing).to_numpy()].sum())) for k in keys}
    groups=sorted(raw.values(),key=lambda a:(a['b']/a['n'],str(a['keys'])))
    edges=[];base_mapping={str(k):('__OTHER__' if k in rare else str(k)) for k in counts.index}
   else:
    v=s.dropna().to_numpy(float)
    edges=np.unique(np.quantile(v,np.linspace(0,1,self.cfg['prebins']+1)[1:-1])).tolist() if len(v) else []
    ids=np.searchsorted(edges,s.to_numpy(float),side='left')
    groups=[dict(keys=[i],n=int(((ids==i)&~missing).sum()),b=int(y[(ids==i)&~missing].sum())) for i in range(len(edges)+1)]
    base_mapping={}
   dirs=[0] if categorical else [DIRECTIONS[f]] if f in DIRECTIONS else [1,-1]
   candidates=[]
   for direction in dirs:
    gs,logs=merge(groups,m,len(y),direction,self.cfg);candidates.append((stats_iv(gs,m),direction,gs,logs))
   best=max(range(len(candidates)),key=lambda i:(candidates[i][0],-i));iv,direction,gs,logs=candidates[best]
   for l in logs:self.log.append(dict(field=f,**l))
   mapping={};woe={};occupied=[g for g in gs+[m] if g['n']>0];k=len(occupied)
   for i,g in enumerate(gs+[m]):
    bid='MISSING' if i==len(gs) else f'B{i}'
    for key in g['keys']:mapping[key]=bid
    pg=(g['n']-g['b']+.5)/((1-y).sum()+.5*k) if g['n'] else 0
    pb=(g['b']+.5)/(y.sum()+.5*k) if g['n'] else 0
    w=float(np.log(pg/pb)) if g['n'] else 0.;woe[bid]=w
    if categorical:definition=str([v for v,target in base_mapping.items() if target in g['keys']]) if bid!='MISSING' else 'missing'
    elif bid=='MISSING':definition='missing'
    else:
     lo=min(g['keys']);hi=max(g['keys']);definition=f"({edges[lo-1] if lo else '-inf'}, {edges[hi] if hi<len(edges) else '+inf'}]"
    self.rows.append(dict(field=f,bin=bid,definition=definition,n=g['n'],good=g['n']-g['b'],bad=g['b'],share=g['n']/len(y),bad_rate=g['b']/g['n'] if g['n'] else np.nan,woe=w,iv=(pg-pb)*w,total_iv=iv,direction=direction,missing_exception=bid=='MISSING' and 0<g['n']<.05*len(y)))
   self.spec[f]=dict(categorical=categorical,edges=edges,base_mapping=base_mapping,mapping=mapping,woe=woe,direction=direction)
   enough=all(g['n']>=self.cfg['min_share']*len(y) for g in gs)
   variable=len(gs)>1 or (m['n']>0 and any(g['n']>0 for g in gs))
   eligible=iv>=self.cfg['iv_min'] and enough and variable
   self.selection.append(dict(field=f,iv=iv,ordinary_bins=len(gs),direction=direction,eligible=eligible,selected=False,reason='candidate' if eligible else 'low_IV_or_constant_or_min_share'))
  z=self.transform(x);keep=[]
  for r in sorted(self.selection,key=lambda r:(-r['iv'],r['field'])):
   if not r['eligible']:continue
   conflict=next((f for f in keep if abs(z[f].corr(z[r['field']]))>=self.cfg['correlation']),None)
   if conflict:r['reason']='correlated_with_'+conflict
   else:keep.append(r['field']);r['selected']=True;r['reason']='retained'
  self.selected=keep;assert keep,'No eligible scorecard variables'
  return self
 def bin_ids(self,x):
  out=pd.DataFrame(index=x.index)
  for f,spec in self.spec.items():
   s=x[f]
   if spec['categorical']:
    ids=s.map(spec['base_mapping']).map(spec['mapping']).fillna('FALLBACK');ids=ids.mask(s.eq('__MISSING__'),'MISSING')
   else:
    ids=pd.Series(np.searchsorted(spec['edges'],s.to_numpy(float),side='left'),index=x.index).map(spec['mapping']).fillna('FALLBACK');ids=ids.mask(s.isna(),'MISSING')
   out[f]=ids
  return out
 def transform(self,x):
  ids=self.bin_ids(x)
  return pd.DataFrame({f:ids[f].map(s['woe']).fillna(0.) for f,s in self.spec.items()},index=x.index)
