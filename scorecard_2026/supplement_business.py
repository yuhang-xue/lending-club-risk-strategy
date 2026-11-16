"""Post-freeze descriptive supplement; does not change primary models or thresholds."""
import json,shutil
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from src.common import ROOT,features,score,probability,sha,save,csv,md,now
from src.metrics import policy,wilson
run=Path(json.loads((ROOT/'latest_run.json').read_text())['directory']);art=ROOT/'artifacts'/run.name
out=run/'supplements/business_review_v1';out.mkdir(parents=True,exist_ok=False)
config=dict(created_at=now(),source_run=run.name,analysis='post-freeze descriptive extension; primary OOT already inspected',targets=[.5,.7,.9],fico_rule='whole FICO groups, maximum validation acceptance <= target; minimum max(500,5% N)',case_rule='70% policy; lowest numeric loan id in each LR/LightGBM error type; full score case first LR false positive',groups={'fico':[680,720],'dti':[15,25]},group_support='n>=200 and bad/good>=20; Wilson95 describes observed rate, not adjusted for dependence/multiple groups',no_refit=True)
save(out/'analysis_config.json',config)
modelhash=sha(art/'models.joblib');thresholdhash=sha(run/'tables/threshold_validation.csv')
m=joblib.load(art/'models.joblib');p=pd.read_parquet(run/'tables/predictions.parquet');p.id=p.id.astype(str)
raw=pd.read_parquet(run/'audit_input.parquet');raw.id=raw.id.astype(str);raw=raw.set_index('id').loc[p.id].reset_index();assert raw.id.tolist()==p.id.tolist();x=features(raw)
y=p.y.to_numpy();vm=p.split.eq('V_policy').to_numpy();oot=p.split.eq('OOT').to_numpy();fico=-p.risk_FICO.to_numpy()
# All FICO cutoffs are selected on V_policy alone, before any supplementary OOT metric.
v,n=np.unique(fico[vm],return_counts=True);counts=np.cumsum(n[::-1]);vals=v[::-1];thresholds=[]
for target in config['targets']:
 ok=(counts/vm.sum()<=target)&(counts>=max(500,.05*vm.sum()));assert ok.any()
 i=np.flatnonzero(ok)[-1];thresholds.append(dict(model='FICO',target=target,threshold=float(vals[i])))
save(out/'fico_threshold_freeze.json',dict(frozen_at=now(),thresholds=thresholds,source='V_policy only',primary_OOT_previously_seen=True))
old=pd.read_csv(run/'tables/threshold_validation.csv');policies=[];selections={}
for role,mask in [('V_policy',vm),('OOT',oot)]:
 for name in ['FICO','LR','LightGBM']:
  for target in config['targets']:
   th=next(r['threshold'] for r in thresholds if r['target']==target) if name=='FICO' else float(old[(old.model==name)&(old.target==target)].threshold.iloc[0])
   sc=fico if name=='FICO' else p['score_'+name].to_numpy();pdv=np.full(len(p),np.nan) if name=='FICO' else p['risk_'+name].to_numpy()
   r=policy(y[mask],pdv[mask],sc[mask],th);r.update(model=name,target=target,split=role,threshold=th,threshold_unit='raw FICO' if name=='FICO' else 'model score')
   if name=='FICO':r['mean_pd']=np.nan
   policies.append(r)
   if role=='OOT':selections[name,target]=sc[oot]>=th
csv(out/'three_model_policy.csv',policies)
# Actual fixed-policy replacement decomposition, not equal-count causal gain.
exchange=[];delta=[]
for name in ['LR','LightGBM']:
 for target in config['targets']:
  a=selections['FICO',target];b=selections[name,target];yy=y[oot];parts={'both_accept':a&b,'FICO_only':a&~b,'model_only':~a&b,'both_reject':~a&~b}
  for group,mask in parts.items():exchange.append(dict(model=name,target=target,group=group,n=int(mask.sum()),bad=int(yy[mask].sum()),good=int((1-yy[mask]).sum()),bad_rate=yy[mask].mean() if mask.any() else np.nan))
  delta.append(dict(model=name,target=target,accepted_delta=int(b.sum()-a.sum()),accepted_bad_delta=int(yy[b].sum()-yy[a].sum()),accepted_good_delta=int((1-yy[b]).sum()-(1-yy[a]).sum()),acceptance_delta_pp=100*(b.mean()-a.mean()),not_equal_volume=True))
csv(out/'policy_exchange.csv',exchange);csv(out/'policy_deltas.csv',delta)
points=pd.read_csv(run/'tables/scorecard_points.csv');excerpt=points[(points.field.isin(['fico_mean','annual_inc']))&points.bin.str.startswith('B')];csv(out/'scorecard_excerpt.csv',excerpt)
ids=np.flatnonzero(oot);ordered=ids[np.argsort(p.iloc[ids].id.astype('int64').to_numpy(),kind='stable')];cases=[];details=[]
for name in ['LR','LightGBM']:
 th=float(old[(old.model==name)&(old.target==.7)].threshold.iloc[0]);accepted=p['score_'+name].to_numpy()>=th
 for label,mask in [('good_rejected',(y==0)&~accepted),('bad_accepted',(y==1)&accepted)]:
  idx=int(next(i for i in ordered if mask[i]));ident=f'{name}_{label}';binids=m.binner.bin_ids(x.iloc[[idx]])
  for field in m.selected:
   bid=binids.iloc[0][field];row=points[(points.field==field)&(points.bin==bid)].iloc[0]
   details.append(dict(case=ident,field=field,value=x.iloc[idx][field],bin=bid,definition=row.definition,woe=row.woe,coefficient=row.coefficient,points=row.points))
  contrib=m.lgb.booster_.predict(m.tree_x(x.iloc[[idx]]),pred_contrib=True)[0];top=np.argsort(-np.abs(contrib[:-1]))[:4]
  cases.append(dict(case=ident,loan_id=p.iloc[idx].id,issue_month=p.iloc[idx].issue_month,label=int(y[idx]),fico=fico[idx],income=x.iloc[idx].annual_inc,dti=x.iloc[idx].dti,loan_income=x.iloc[idx].loan_income,LR_pd=p.iloc[idx].risk_LR,LR_score=p.iloc[idx].score_LR,LightGBM_pd=p.iloc[idx].risk_LightGBM,LightGBM_score=p.iloc[idx].score_LightGBM,FICO_accept=bool(fico[idx]>=thresholds[1]['threshold']),LR_accept=bool(p.iloc[idx].score_LR>=float(old[(old.model=='LR')&(old.target==.7)].threshold.iloc[0])),LightGBM_accept=bool(p.iloc[idx].score_LightGBM>=float(old[(old.model=='LightGBM')&(old.target==.7)].threshold.iloc[0])),raw_shap_top='; '.join(f'{x.columns[j]}: {contrib[j]:+.4f}' for j in top)))
csv(out/'error_cases.csv',cases);csv(out/'case_score_details.csv',details)
full=cases[0];dd=pd.DataFrame(details);d=dd[dd.case==full['case']];base=float(points.base_points.iloc[0]);total=float(base+d.points.sum());pdcalc=float(probability(total));assert abs(round(total,2)-full['LR_score'])<1e-8;assert abs(pdcalc-full['LR_pd'])<1e-12
save(out/'full_score_case.json',dict(**full,base_points=base,variable_points_sum=d.points.sum(),unrounded_score=total,rounding='2 decimals, half-to-even',reconstructed_pd=pdcalc,selection_rule=config['case_rule']))
# Fixed business bands (not selected using group outcomes), missing separate.
groupdefs={'fico_mean':[-np.inf,680,720,np.inf],'dti':[-np.inf,15,25,np.inf]};groups=[]
for field,edges in groupdefs.items():
 bins=pd.cut(x[field],edges,right=False).astype(str).to_numpy()
 for role,mask in [('V_model',p.split.eq('V_model').to_numpy()),('V_policy',vm),('OOT',oot)]:
  for band in sorted(set(bins[mask])):
   z=mask&(bins==band);n=int(z.sum());bad=int(y[z].sum());lo,hi=wilson(bad,n)
   for name in ['LR','LightGBM']:
    mean=float(p.loc[z,'risk_'+name].mean());groups.append(dict(field=field,band=band,split=role,model=name,n=n,bad=bad,observed_rate=bad/n,mean_pd=mean,gap_pp=100*(bad/n-mean),wilson_lower=lo,wilson_upper=hi,supported=n>=200 and min(bad,n-bad)>=20,interval_note='observed-rate only; not simultaneous calibrated-PD test'))
csv(out/'group_calibration.csv',groups)
# Independent policy identities and original results must match.
allpol=pd.DataFrame(policies);orig=pd.read_csv(run/'tables/threshold_oot.csv')
for r in orig.itertuples():
 rr=allpol[(allpol.model==r.model)&(allpol.target==r.target)&(allpol.split=='OOT')].iloc[0]
 for col in ['accepted','accepted_bad','acceptance','actual_bad_rate','good_reject','bad_capture']:assert abs(rr[col]-getattr(r,col))<1e-12
for r in thresholds:
 th=r['threshold'];accepted=fico[vm]>=th;assert accepted.mean()<=r['target'];lower=vals[vals<th]
 if len(lower):assert (fico[vm]>=lower[0]).mean()>r['target']
assert sha(art/'models.joblib')==modelhash and sha(run/'tables/threshold_validation.csv')==thresholdhash
save(out/'verification.json',dict(status='passed',model_hash_unchanged=modelhash,original_threshold_hash_unchanged=thresholdhash,original_policy_reproduced=True,whole_FICO_ties=True,FICO_validation_caps=True,score_reconstruction_error=abs(pdcalc-full['LR_pd']),group_definitions_frozen_before_group_metrics=True,primary_OOT_previously_seen=True,created_at=now()))
shutil.copy2(__file__,out/'supplement_business.py')
print(out);print(allpol[allpol.split=='OOT'].to_string(index=False));print(json.dumps(full,ensure_ascii=False));print(total,base)
