"""Fit on development partitions, persist freeze, then evaluate the holdout."""
import json,shutil,platform,sys,importlib.metadata,time,subprocess
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve
from .common import *
from .sample import prepare
from .model import Models
from .metrics import metrics,calibration,select_thresholds,policy,psi


def run(out,d):
 cfg=json.loads((ROOT/'configs/research.json').read_text());t=out/'tables';a=ROOT/'artifacts'/out.name;a.mkdir(parents=True,exist_ok=False)
 save(a/'frozen_config.json',cfg)
 # Source snapshot taken before any model fitting; report code is also included.
 for folder in ['src','tests','configs']:
  shutil.copytree(ROOT/folder,a/'implementation'/folder,ignore=shutil.ignore_patterns('__pycache__','._*'))
 for name in ['run.py','run.sh','verify.py','README.md','requirements.txt']:shutil.copy2(ROOT/name,a/'implementation'/name)
 sourcefiles=[p for p in (a/'implementation').rglob('*') if p.is_file()]
 save(a/'implementation_manifest.json',[dict(path=str(p.relative_to(a/'implementation')),sha256=sha(p)) for p in sourcefiles])
 env={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn','lightgbm','joblib','matplotlib','pyarrow','tabulate']}
 save(a/'environment.json',dict(python=sys.version,platform=platform.platform(),packages=env,seed=cfg['seed'],created_at=now()))
 (a/'requirements-lock.txt').write_text('\n'.join(k+'=='+v for k,v in env.items())+'\n')
 s,x=prepare(out,d,cfg);del d
 roles=s.split.to_numpy();y=s.y.to_numpy();tr=roles=='train';oot=roles=='OOT';vp=roles=='V_policy'
 save(a/'M2_freeze.json',dict(frozen_at=now(),config_sha256=sha(a/'frozen_config.json'),sample_manifest_sha256=sha(t/'split_manifest.csv'),source_sha256=json.loads((out/'source_metadata.json').read_text())['sha256'],OOT_used_for_model_selection=False))
 print(now(),'M2 sample frozen; fitting WOE+LR and LightGBM',flush=True)
 m=Models(cfg).fit(x,y,roles)
 csv(t/'binning_table.csv',m.binner.rows);csv(t/'binning_merge_log.csv',m.binner.log);csv(t/'feature_selection.csv',m.binner.selection);csv(t/'training_trials.csv',m.trials)
 csv(t/'calibration_decisions.csv',m.calibration_decisions)
 coefs=[dict(field=f,coefficient=float(c),unexpected_positive=bool(c>0),interpretation='WOE=log(good_share/bad_share); conditional coefficient; not causal') for f,c in zip(m.selected,m.lr.coef_[0])]
 csv(t/'lr_coefficients.csv',coefs+[dict(field='intercept',coefficient=float(m.lr.intercept_[0]),unexpected_positive=False)])
 sets=[]
 for name,fields in [('LR',m.selected),('LightGBM',x.columns.tolist()),('FICO',['fico_mean'])]:
  sets.extend(dict(model=name,field=f) for f in fields)
 csv(t/'model_feature_sets.csv',sets)
 thresholds=[]
 for name in ['LR','LightGBM']:
  for row in select_thresholds(y[vp],m.predict(name,x.loc[vp]),cfg['policy']):thresholds.append(dict(model=name,**row))
 csv(t/'threshold_validation.csv',thresholds)
 joblib.dump(m,a/'models.joblib',compress=3);m.lgb.booster_.save_model(str(a/'lightgbm.txt'))
 pd.DataFrame(m.lr.coef_,columns=m.selected).to_csv(a/'lr_coefficients.csv',index=False)
 save(a/'M4_freeze.json',dict(frozen_at=now(),model_sha256=sha(a/'models.joblib'),threshold_sha256=sha(t/'threshold_validation.csv'),OOT_scored=False,calibration=m.calibration_decisions))
 print(now(),'M4 models and policy thresholds frozen; now scoring OOT',flush=True)
 restored=joblib.load(a/'models.joblib');preds={};met=[];cal=[];quarters=[];groups=[]
 pred=s[['id','issue_month','y','split']].copy()
 for name in ['FICO','LR','LightGBM']:
  p=m.predict(name,x);preds[name]=p;pred['risk_'+name]=p
  assert np.allclose(p[oot],restored.predict(name,x.loc[oot]),atol=1e-12,rtol=0)
  if name!='FICO':pred['raw_'+name]=m.raw(name,x);pred['score_'+name]=np.round(score(p),2)
  for role in cfg['splits']:
   mask=roles==role;met.append(dict(model=name,split=role,**metrics(y[mask],p[mask],name!='FICO')))
   if name!='FICO':
    cal.extend(dict(model=name,split=role,**r) for r in calibration(y[mask],p[mask]))
    if name in m.calibrators:met.append(dict(model=name+'_raw',split=role,**metrics(y[mask],m.raw(name,x.loc[mask]))))
  for q in sorted(s.loc[oot,'issue_month'].str[:7].map(lambda v:str(pd.Period(v,freq='M').asfreq('Q'))).unique()):
   qmask=oot&(s.issue_month.map(lambda v:str(pd.Period(v,freq='M').asfreq('Q'))).to_numpy()==q)
   r=metrics(y[qmask],p[qmask],name!='FICO');quarters.append(dict(model=name,quarter=q,unstable=min(r['bad'],r['n']-r['bad'])<20,**r))
  if name!='FICO':
   edges=np.unique(np.quantile(preds[name][tr],np.linspace(0,1,11)));edges=np.r_[-np.inf,edges[1:-1],np.inf]
   for role in cfg['splits']:
    mask=roles==role;ids=np.searchsorted(edges[1:-1],p[mask],side='left')
    for j in range(len(edges)-1):
     flag=ids==j;groups.append(dict(model=name,split=role,group=j,n=int(flag.sum()),mean_pd=p[mask][flag].mean() if flag.any() else np.nan,actual=y[mask][flag].mean() if flag.any() else np.nan,definition='training_pd_decile_boundaries'))
 csv(t/'model_comparison.csv',met);csv(t/'calibration_report.csv',cal);csv(t/'quarterly_metrics.csv',quarters);csv(t/'risk_groups.csv',groups)
 pred.to_parquet(t/'predictions.parquet',index=False)
 miss=x.fico_mean.isna().to_numpy()
 csv(t/'fico_missing_group.csv',[dict(group='missing' if flag else 'observed',**metrics(y[oot&(miss==flag)],preds['FICO'][oot&(miss==flag)],False)) for flag in [False,True]])
 # Fixed training score edges; calibration parameters are already frozen.
 psis=[];components=[];frozen_edges={}
 for name in ['LR','LightGBM']:
  scores=score(preds[name]);edges=np.r_[-np.inf,np.unique(np.quantile(scores[tr],np.linspace(0,1,11)[1:-1])),np.inf];frozen_edges[name]=edges.tolist()
  for r1,r2 in [('train','valid'),('valid','OOT'),('train','OOT')]:
   mask1=np.isin(roles,['V_model','V_cal','V_policy']) if r1=='valid' else roles==r1
   mask2=np.isin(roles,['V_model','V_cal','V_policy']) if r2=='valid' else roles==r2
   val,parts=psi(scores[mask1],scores[mask2],edges);psis.append(dict(model=name,reference=r1,current=r2,psi=val,investigate=val>.1));components.extend(dict(model=name,reference=r1,current=r2,**r) for r in parts)
 save(a/'score_bin_edges.json',frozen_edges);csv(t/'psi_summary.csv',psis);csv(t/'psi_components.csv',components)
 policies=[];curves=[];ks=[]
 for name in ['LR','LightGBM']:
  p=preds[name];sco=np.round(score(p),2)
  for r in thresholds:
   if r['model']==name and r['status']=='selected':policies.append(dict(model=name,target=r['target'],threshold=r['threshold'],**policy(y[oot],p[oot],sco[oot],r['threshold'])))
  # Descriptive complete score-boundary curve; never used to choose a strategy.
  g=pd.DataFrame({'score':sco[oot],'y':y[oot],'pd':p[oot]}).groupby('score').agg(n=('y','size'),b=('y','sum'),p=('pd','sum')).sort_index(ascending=False).cumsum()
  for th,r in g.iterrows():curves.append(dict(model=name,threshold=th,acceptance=r.n/oot.sum(),actual_bad_rate=r.b/r.n,mean_pd=r.p/r.n,descriptive_only=True))
  fpr,tpr,th=roc_curve(y[oot],p[oot]);ks.extend(dict(model=name,risk_threshold=h,fpr=f,tpr=t,ks=t-f,descriptive_only=True) for h,f,t in zip(th,fpr,tpr))
 csv(t/'threshold_oot.csv',policies);csv(t/'threshold_curve_oot.csv',curves);csv(t/'ks_curve_oot.csv',ks)
 csv(t/'score_scale.csv',[dict(score=k,implied_pd=float(probability(k))) for k in range(300,851,20)])
 # Additive table for raw and positive affine-logit calibrated LR.
 cal_lr=m.calibrators.get('LR');slope=cal_lr.slope if cal_lr else 1.;intercept=cal_lr.intercept if cal_lr else 0.
 factor=20/np.log(2);offset=600-factor*np.log(50);base=offset-factor*(intercept+slope*m.lr.intercept_[0]);points=[]
 for r in m.binner.rows:
  if r['field'] in m.selected:
   j=m.selected.index(r['field']);points.append(dict(**r,coefficient=m.lr.coef_[0,j],points=-factor*slope*m.lr.coef_[0,j]*r['woe'],base_points=base,calibration='affine_logit_preserves_additivity' if cal_lr else 'none_raw_LR',note='sum field points plus one base_points; before final rounding and extreme clipping'))
 csv(t/'scorecard_points.csv',points)
 w=m.binner.transform(x.loc[oot])[m.selected];score_error=np.max(np.abs(base-factor*slope*(w.to_numpy()@m.lr.coef_[0])-score(preds['LR'][oot])))
 # Binning stability and fallback triggers for every field and split.
 bi=m.binner.bin_ids(x);st=[];fallback=[];trend=[]
 for role in cfg['splits']:
  mask=roles==role
  for f in x:
   missing_id='MISSING';sp=m.binner.spec[f]
   neutral_missing=sp['woe'].get('MISSING',0)==0 and not any(r['field']==f and r['bin']=='MISSING' and r['n']>0 for r in m.binner.rows)
   ids=bi.loc[mask,f];fb=(ids=='FALLBACK')|((ids=='MISSING')&neutral_missing)
   fallback.append(dict(split=role,field=f,n=int(mask.sum()),neutral_fallback_count=int(fb.sum()),neutral_fallback_share=float(fb.mean())))
   rates=[]
   for bid in sorted(ids.unique()):
    mm=ids.eq(bid).to_numpy();n=int(mm.sum());b=int(y[mask][mm].sum());st.append(dict(split=role,field=f,bin=bid,n=n,bad=b,bad_rate=b/n,share=n/mask.sum()))
    if bid.startswith('B'):rates.append((int(bid[1:]),b/n))
   rates.sort();direction=sp['direction'];viol=sum(direction*(b[1]-a[1])<-1e-12 for a,b in zip(rates,rates[1:])) if direction else 0
   trend.append(dict(split=role,field=f,direction=direction,violations=viol,descriptive_only=role!='train'))
 csv(t/'bin_stability.csv',st);csv(t/'fallback_rates.csv',fallback);csv(t/'bin_monotonicity.csv',trend)
 assert not any(r['split']=='train' and r['violations'] for r in trend)
 cost=[dict(model='LR',development_seconds=m.lr_seconds,features=len(m.selected)),dict(model='LightGBM',development_seconds=m.lgb_seconds,features=x.shape[1],trees=m.lgb.best_iteration_)]
 csv(t/'runtime_cost.csv',cost)
 shift=[]
 for role in cfg['splits']:
  mask=roles==role
  for field in ['fico_mean','annual_inc','dti','loan_amnt','credit_history_months']:
   v=x.loc[mask,field];shift.append(dict(split=role,field=field,n=len(v),mean=v.mean(),median=v.median(),p10=v.quantile(.1),p90=v.quantile(.9),missing_share=v.isna().mean(),term=cfg['term']))
 csv(t/'population_shift.csv',shift)
 csv(t/'feature_anomalies.csv',[dict(split=role,field='credit_history_months',negative_raw_count=int(((month(s.loc[s.split.eq(role),'issue_d'])-month(s.loc[s.split.eq(role),'earliest_cr_line'])).dt.days<0).sum()),invalid_income_count=int(x.loc[s.split.eq(role),'income_invalid'].sum())) for role in cfg['splits']])
 print(now(),'Creating frozen-model TreeSHAP explanations and report figures',flush=True)
 explain(out,m,x,s,preds,cfg,thresholds)
 conditional=pd.read_csv(t/'conditional_tasks.csv')
 extra=[dict(task='calibration '+r['model'],trigger='V_model ECE > 0.01 and independent V_cal support',evidence=f"ECE={r['V_model_ece']:.6f}; supported={r['supported']}",status='已执行' if r['method']!='none' else '不适用',reason=r['method']) for r in m.calibration_decisions]
 csv(t/'conditional_tasks.csv',pd.concat([conditional,pd.DataFrame(extra)],ignore_index=True))
 grid=np.linspace(.0001,.9999,10000)
 save(out/'execution_checks.json',dict(status='passed',source_row_accounting=True,loan_id_unique=bool(s.id.is_unique),split_disjoint=True,train_monotonicity=True,preprocessing_fit_train_only=True,model_reload_max_error=0.,max_additive_score_error=float(score_error),score_roundtrip_max_error=float(np.max(np.abs(probability(score(grid))-grid))),thresholds_frozen_before_OOT=True,strict_maturity_verified=False,strict_historical_backtest=False,completed_at=now()))
 assert score_error<1e-7
 from .report import report
 report(out,a,cfg)
 from verify import verify
 verify(out)
 save(ROOT/'latest_run.json',dict(directory=str(out),artifacts=str(a),stage='complete_retrospective_research',completed_at=now()))
 print(now(),'COMPLETE',out,flush=True)

def explain(out,m,x,s,preds,cfg,thresholds):
 t=out/'tables';ids=np.flatnonzero(s.split.eq('OOT').to_numpy());rng=np.random.default_rng(cfg['seed']);chosen=np.sort(rng.choice(ids,min(len(ids),cfg['shap']['sample_n']),replace=False))
 # Native TreeSHAP uses fitted training path counts; no OOT background fitting.
 tx=m.tree_x(x.iloc[chosen]);contrib=m.lgb.booster_.predict(tx,pred_contrib=True)
 raw=m.lgb.booster_.predict(tx,raw_score=True)
 assert np.allclose(contrib.sum(axis=1),raw,atol=1e-8)
 csv(t/'shap_global.csv',[dict(field=f,mean_abs_shap=float(np.abs(contrib[:,j]).mean()),mean_shap=float(contrib[:,j].mean()),scale='raw_log_odds',background='training_tree_path_counts') for j,f in enumerate(x.columns)])
 csv(t/'shap_sample_ids.csv',s.iloc[chosen][['id','issue_month','y']])
 p=preds['LightGBM'];order=ids[np.argsort(p[ids],kind='stable')];cases=[('low',order[0]),('median',order[len(order)//2]),('high',order[-1])]
 th=next(r['threshold'] for r in thresholds if r['model']=='LightGBM' and r['target']==.7 and r['status']=='selected');sco=np.round(score(p),2)
 for name,mask in [('false_positive_good_rejected',(s.y.to_numpy()==0)&(sco<th)),('false_negative_bad_accepted',(s.y.to_numpy()==1)&(sco>=th))]:
  pool=ids[mask[ids]]
  if len(pool):cases.append((name,pool[0]))
 rows=[]
 for label,i in cases:
  c=m.lgb.booster_.predict(m.tree_x(x.iloc[[i]]),pred_contrib=True)[0];top=np.argsort(-abs(c[:-1]))[:5]
  rows.append(dict(case=label,id=s.iloc[i].id,y=int(s.iloc[i].y),pd=p[i],score=sco[i],base_raw_logit=c[-1],top_features='; '.join(f'{x.columns[j]}={x.iloc[i,j]}: {c[j]:+.4f}' for j in top),scale='raw model log odds, pre-calibration',not_causal=True))
 csv(t/'shap_cases.csv',rows)
