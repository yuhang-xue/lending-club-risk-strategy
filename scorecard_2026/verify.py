"""Read-only independent recomputation from saved predictions and freeze records."""
import argparse,json,hashlib,os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score,brier_score_loss
from src.common import ROOT,sha,save,now

def verify(out):
 t=out/'tables';a=ROOT/'artifacts'/out.name;checks={}
 def check(name,condition):
  checks[name]=bool(condition)
  if not condition:raise AssertionError(name)
 cfg=json.loads((a/'frozen_config.json').read_text());s=pd.read_csv(t/'split_manifest.csv',dtype={'id':str});p=pd.read_parquet(t/'predictions.parquet');p.id=p.id.astype(str)
 check('prediction_membership',s.id.tolist()==p.id.tolist() and s.y.tolist()==p.y.tolist() and s.split.tolist()==p.split.tolist())
 check('loan_ids_unique',s.id.is_unique)
 check('split_boundaries',all(s.loc[s.split.eq(k),'issue_month'].between(*v).all() for k,v in cfg['splits'].items()))
 ledger=pd.read_csv(t/'sample_ledger.csv.gz',dtype={'id':str});meta=json.loads((out/'source_metadata.json').read_text())
 check('ledger_raw_accounting',len(ledger)==meta['rows']);check('ledger_included',ledger.reason.eq('included').sum()==len(s))
 check('label_mapping',set(ledger.loc[ledger.reason.eq('included'),'loan_status'])<=set(['Fully Paid','Charged Off','Default']))
 label=ledger.loc[ledger.reason.eq('included')].set_index('id').loan_status.map({'Fully Paid':0,'Charged Off':1,'Default':1})
 check('individual_labels',np.array_equal(s.y,label.loc[s.id]))
 freeze=json.loads((a/'M4_freeze.json').read_text());m2=json.loads((a/'M2_freeze.json').read_text())
 check('frozen_model_hash',sha(a/'models.joblib')==freeze['model_sha256']);check('frozen_threshold_hash',sha(t/'threshold_validation.csv')==freeze['threshold_sha256'])
 check('M2_manifest',sha(t/'split_manifest.csv')==m2['sample_manifest_sha256']);check('M2_config',sha(a/'frozen_config.json')==m2['config_sha256'])
 check('freeze_order',m2['frozen_at']<freeze['frozen_at'] and freeze['OOT_scored'] is False)
 check('source_hash',sha(Path(meta['source']))==meta['sha256'])
 manifest=json.loads((a/'implementation_manifest.json').read_text())
 check('implementation_snapshot',all(sha(a/'implementation'/r['path'])==r['sha256'] for r in manifest))
 sets=pd.read_csv(t/'model_feature_sets.csv');reg=pd.read_csv(t/'feature_registry.csv');allowed=set(reg.loc[reg.model_eligible,'field'])
 check('whitelist',set(sets.field)<=allowed)
 banned={'grade','sub_grade','int_rate','installment','loan_status','id','member_id','verification_status','last_fico_range_low','last_fico_range_high','total_pymnt','recoveries'}
 check('no_banned_features',not(set(sets.field)&banned))
 bt=pd.read_csv(t/'binning_table.csv');ordinary=bt[bt.bin.str.startswith('B')]
 check('ordinary_minimum_size',(ordinary.share>=.05-1e-12).all());check('maximum_bins',ordinary.groupby('field').size().max()<=5)
 for field,g in bt.groupby('field'):
  g=g[g.n>0];k=len(g);pg=(g.good+.5)/(g.good.sum()+.5*k);pb=(g.bad+.5)/(g.bad.sum()+.5*k)
  check('woe_'+field,np.allclose(np.log(pg/pb),g.woe,atol=1e-12));check('iv_'+field,np.allclose((pg-pb)*np.log(pg/pb),g.iv,atol=1e-12))
 for field,g in ordinary.groupby('field'):
  g=g.assign(order=g.bin.str[1:].astype(int)).sort_values('order');direction=g.direction.iloc[0]
  if direction:check('monotone_'+field,(direction*np.diff(g.bad_rate.dropna())>=-1e-12).all())
 met=pd.read_csv(t/'model_comparison.csv');metric_errors=[]
 for _,r in met.iterrows():
  v=p[p.split==r.split];name=r.model
  risk=v['raw_'+name[:-4]].to_numpy() if name.endswith('_raw') else v['risk_'+name].to_numpy()
  y=v.y.to_numpy();auc=roc_auc_score(y,risk)
  # independently evaluate KS at all tied risk groups using sorted cumulative counts
  g=pd.DataFrame({'risk':risk,'y':y}).groupby('risk').agg(n=('y','size'),bad=('y','sum')).sort_index()
  ks=np.max(np.abs(g.bad.cumsum()/g.bad.sum()-(g.n-g.bad).cumsum()/(g.n-g.bad).sum()))
  metric_errors += [abs(auc-r.auc),abs(ks-r.ks)]
  if name!='FICO':metric_errors.append(abs(np.mean((risk-y)**2)-r.brier))
 check('recomputed_metrics',max(metric_errors)<1e-12)
 for name in ['LR','LightGBM']:
  pdv=np.clip(p['risk_'+name],1e-9,1-1e-9);sc=600-20*np.log2(50)+20*np.log2((1-pdv)/pdv)
  check('scores_'+name,np.allclose(np.round(sc,2),p['score_'+name],atol=1e-10))
 tv=pd.read_csv(t/'threshold_validation.csv');to=pd.read_csv(t/'threshold_oot.csv');err=[]
 for table,role in [(tv,'V_policy'),(to,'OOT')]:
  v=p[p.split==role]
  for _,r in table.iterrows():
   if pd.isna(r.threshold):continue
   keep=v['score_'+r.model]>=r.threshold;y=v.y;pn=keep.sum();b=(y==1).sum();g=(y==0).sum();bk=y[keep].sum()
   check('accepted_count_'+role+r.model+str(r.target),pn==r.accepted)
   values={'acceptance':pn/len(v),'actual_bad_rate':bk/pn,'mean_pd':v.loc[keep,'risk_'+r.model].mean(),'good_reject':((y==0)&~keep).sum()/g,'bad_capture':((y==1)&~keep).sum()/b,'bad_escape':bk/b}
   err.extend(abs(values[k]-r[k]) for k in values)
   if role=='V_policy':
    check('target_cap_'+r.model+str(r.target),pn/len(v)<=r.target)
    lower=np.sort(v['score_'+r.model].unique());lower=lower[lower<r.threshold]
    if len(lower):check('best_attainable_'+r.model+str(r.target),(v['score_'+r.model]>=lower[-1]).mean()>r.target)
 check('recomputed_policies',max(err)<1e-12)
 check('thresholds_transfer',np.allclose(tv.threshold,to.threshold))
 pc=pd.read_csv(t/'psi_components.csv');ps=pd.read_csv(t/'psi_summary.csv');errors=[]
 for key,g in pc.groupby(['model','reference','current']):
  n=len(g);q=(g.current_n+.5)/(g.current_n.sum()+.5*n);r=(g.reference_n+.5)/(g.reference_n.sum()+.5*n)
  val=((q-r)*np.log(q/r)).sum();old=ps[(ps.model==key[0])&(ps.reference==key[1])&(ps.current==key[2])].psi.iloc[0];errors.append(abs(val-old))
 check('recomputed_psi',max(errors)<1e-12)
 report=(out/'完整研究报告.md').read_text();check('limitations_disclosed',all(k in report for k in ['统一快照未证实','不是严格历史回测','member_id','选择偏差','首次未见']))
 result=dict(status='passed',checks=checks,check_count=len(checks),max_metric_error=max(metric_errors),max_policy_error=max(err),max_psi_error=max(errors),verified_at=now(),limitations='strict maturity / historical backtest / borrower isolation not verified')
 save(out/'verification.json',result);print(json.dumps({k:v for k,v in result.items() if k!='checks'},ensure_ascii=False,indent=2));return result
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path);args=parser.parse_args()
 out=args.run or Path(json.loads((ROOT/'latest_run.json').read_text())['directory']);verify(out)
