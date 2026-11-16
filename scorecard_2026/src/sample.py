import json
import numpy as np
import pandas as pd
from .common import *
GOOD=['Fully Paid'];BAD=['Charged Off','Default'];UNRES=['Current','In Grace Period','Late (16-30 days)','Late (31-120 days)']
def prepare(out,d,cfg):
 t=out/'tables';ref=pd.Period(cfg['reference_month_assumption'],freq='M')
 dt=month(d.issue_d);d=d.copy();d['age_assumed']=12*(ref.year-dt.dt.year)+ref.month-dt.dt.month
 valid=d.id.str.fullmatch(r'\d+',na=False)&d.issue_month.notna()
 eligible=valid&(d.term_n==cfg['term'])&(d.age_assumed>=cfg['term']+cfg['buffer_months'])
 resolved=d.loan_status.isin(GOOD+BAD);unresolved=d.loan_status.isin(UNRES)
 reason=pd.Series('included',index=d.index)
 rules=[(~valid,'invalid_id_or_date'),(~d.term_n.eq(cfg['term']),'outside_36_month_term'),(d.age_assumed<cfg['term']+cfg['buffer_months'],'outside_assumed_age_screen'),(d.loan_status.fillna('').str.startswith('Does not meet'),'credit_policy_excluded'),(unresolved,'unresolved'),(~d.loan_status.isin(GOOD+BAD+UNRES),'unknown_status')]
 for mask,name in rules: reason.loc[(reason=='included')&mask]=name
 d['split']='outside_time_windows'
 for name,(a,b) in cfg['splits'].items(): d.loc[d.issue_month.between(a,b),'split']=name
 reason.loc[(reason=='included')&d.split.eq('outside_time_windows')]='outside_time_windows'
 d['reason']=reason;d['y']=d.loan_status.map({**dict.fromkeys(GOOD,0),**dict.fromkeys(BAD,1)})
 csv(t/'sample_flow.csv',d.groupby('reason',dropna=False).size().rename('n').reset_index())
 d[['source_row','id','issue_month','term_n','loan_status','reason','split']].to_csv(t/'sample_ledger.csv.gz',index=False,compression='gzip')
 sens=[];res=[];compare=[]
 for buf in cfg['buffer_sensitivity']:
  base=d[valid&(d.term_n==cfg['term'])&(d.age_assumed>=cfg['term']+buf)&d.loan_status.isin(GOOD+BAD+UNRES)]
  for yr,g in [('all',base)]+list(base.groupby('year')):
   n=len(g);b=int(g.loan_status.isin(BAD).sum());u=int(g.loan_status.isin(UNRES).sum());good=int(g.loan_status.isin(GOOD).sum())
   r=dict(buffer=buf,year=yr,n=n,bad=b,good=good,unresolved=u,unresolved_share=u/n if n else np.nan,rate_resolved=b/(b+good) if b+good else np.nan,bad_lower=b/n if n else np.nan,bad_upper=(b+u)/n if n else np.nan,cutoff=str((ref-cfg['term']-buf)))
   sens.append(r)
   if buf==cfg['buffer_months']:res.append(r)
  f=features(base)
  for state,mask in [('resolved',base.loan_status.isin(GOOD+BAD)),('unresolved',base.loan_status.isin(UNRES))]:
   for field in ['fico_mean','annual_inc','dti','loan_amnt']:
    v=f.loc[mask,field];compare.append(dict(buffer=buf,group=state,field=field,n=len(v),missing_rate=v.isna().mean(),mean=v.mean(),median=v.median(),p10=v.quantile(.1),p90=v.quantile(.9)))
 csv(t/'maturity_sensitivity.csv',sens);csv(t/'maturity_residual_summary.csv',res);csv(t/'resolved_unresolved_comparison.csv',compare)
 sample=d[d.reason.eq('included')].copy().reset_index(drop=True);sample['y']=sample.y.astype(int)
 assert sample.id.is_unique
 summary=sample.groupby('split').agg(n=('y','size'),bad=('y','sum'),bad_rate=('y','mean'),start=('issue_month','min'),end=('issue_month','max')).reset_index();summary['good']=summary.n-summary.bad
 assert set(summary.split)==set(cfg['splits'])
 assert (summary.n>=cfg['minimum_split_n']).all() and (summary[['good','bad']]>=cfg['minimum_split_class']).all().all()
 csv(t/'split_summary.csv',summary);csv(t/'split_manifest.csv',sample[['source_row','id','issue_month','y','split']])
 x=features(sample)
 registry=[]
 allcols=json.loads((out/'source_metadata.json').read_text())['column_names']
 for f in allcols+sorted(set(FEATURES)-set(allcols)):
  admitted=f in FEATURES
  role='predictor' if admitted else ('derivation_only' if f in ['fico_range_low','fico_range_high','earliest_cr_line'] else 'audit_only_not_model')
  registry.append(dict(field=f,model_eligible=admitted,role=role,availability='origination-field definition only; no row-level history' if role!='audit_only_not_model' else 'not used',transformation='see frozen src/common.py features()',reason='reviewed origination whitelist' if admitted else 'verification timing unproven' if f=='verification_status' else 'derivation input only' if role=='derivation_only' else 'not on predictor whitelist'))
 csv(t/'feature_registry.csv',registry)
 av=[dict(field=f,source='raw loan extract',evidence='field definition at origination; no historical update log',simulated_decision='not reconstructed',strict_pass=False,limitation='retrospective approximation') for f in RAW]
 av.append(dict(field='loan_status',source='single raw file',evidence='last_pymnt_d is NOT confirmation time; snapshot unverified',simulated_decision='not reconstructed',strict_pass=False,limitation='resolved-subset retrospective label'))
 csv(t/'temporal_availability_audit.csv',av)
 late=d[eligible&d.loan_status.eq('Late (31-120 days)')];counts=late.groupby('split').size().to_dict()
 late_ok=len(late)>=cfg['late_sensitivity']['min_total_late'] and all(counts.get(k,0)>=cfg['late_sensitivity']['min_late_each_split'] for k in cfg['splits'])
 assert not late_ok,'Late extension triggered: implement before completing pipeline'
 csv(t/'label_sensitivity_summary.csv',[dict(version='main',n=len(sample),bad=int(sample.y.sum()),status='executed'),dict(version='Late_as_bad',n=len(late),bad=np.nan,status='not_applicable_insufficient_Late',detail=json.dumps(counts))])
 csv(t/'conditional_tasks.csv',[
 dict(task='Late label sensitivity',trigger='>=200 Late overall and >=20 each split',evidence=f'Late n={len(late)}, split={counts}',status='不适用',reason='预设支持量不足'),
 dict(task='60-month extension',trigger='optional independent expansion; main 36-month insufficient',evidence=f'36-month n={len(sample)}',status='不适用',reason='主线支持充足，本轮范围限定36期'),
 dict(task='member isolation',trigger='member_id usable',evidence=f'nonmissing={d.member_id.notna().sum()}',status='不适用',reason='重复检查已执行；ID全缺失，只能loan粒度隔离'),
 dict(task='same-variable algorithm contrast',trigger='claim isolated algorithm gain',evidence='only full-pipeline comparison is claimed',status='不适用',reason='不主张纯算法增益'),
 dict(task='risk-target CI constraint',trigger='external bad-rate target supplied',evidence='no external business target',status='不适用',reason='仅通过率档位演示；仍报告描述性Wilson区间'),
 ])
 save(out/'sample_checks.json',dict(raw_rows=len(d),accounted_rows=int(d.groupby('reason').size().sum()),sample_rows=len(sample),unique_ids=sample.id.is_unique,member_id_available=False,strict_historical_backtest=False,confirmed_maturity=False,late_count=len(late),created_at=now()))
 md(out/'sample_definition.md',f'''# 样本定义与冻结

{cfg['research_mode']}。研究参考月为假设2019-03，非已确认快照；36期、假设缓冲6个月，账龄筛查截至2015-09。标签Fully Paid=0，Charged Off/Default=1，其他按互斥流水账排除。样本数 {len(sample):,}。

{summary.to_markdown(index=False)}

V_model=2014-01至04、V_cal=05至08、V_policy=09至12，各至少2000笔且好坏各200笔。已在训练前检验计数门槛。ECE十个等宽概率箱超过0.01触发单一正斜率Platt校准，仅V_cal拟合；不以V_policy或OOT重新选择方法。

无member_id，不声称借款人隔离。主缓冲6个月为研究假设，3/9个月敏感性只比较人群与未决，不以模型表现挑选。残余未决上下界是完整同批次范围诊断，不能修复未知快照及终态选择偏差。

2015批次曾被旧项目研究使用，因此是本次流水线的留出集，非项目层面首次未见验证。默认阈值演示档位已冻结为50/70/90%，并非最优业务策略。
''')
 md(out/'leakage_audit.md','''# 防泄漏边界

仅训练集拟合特征类别映射、分箱、WOE、IV、相关性、基础模型。V_model仅有限调参、早停、校准触发；V_cal仅拟合预设校准器；V_policy仅选分数阈值；全部冻结后才计算OOT模型指标。来源审计和样本计数会查看所有批次，但不计算OOT模型性能。

模型输入为features()显式字段列表。verification_status额外禁用，grade/sub_grade/int_rate/installment、还款/结算/贷后字段、loan_status、id/member_id不进入输入。issue_d仅用于信用历史衍生及切分，信用历史使用放款月近似申请月，不声称精确申请前观测。模型不合并验证集重训。

无法证明逐字段未发生后续更新；来源定义支持贷前字段，但缺少逐笔历史证据。无法确认状态快照及标签可得时点，故不通过严格历史回测验收。
''')
 return sample,x
