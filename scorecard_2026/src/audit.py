"""Full source audit; no predictive model or OOT performance inspection."""
import json,gzip,struct
import numpy as np
import pandas as pd
from .common import *

def audit(out,source):
 t=out/'tables';t.mkdir(parents=True,exist_ok=True)
 log=lambda s:print(now(),s,flush=True)
 digest=sha(source);log('data SHA256 '+digest)
 cols=pd.read_csv(source,nrows=0).columns.tolist()
 use=['id','member_id','issue_d','loan_status','term','last_pymnt_d','last_credit_pull_d']+RAW
 parts=[]
 for i,c in enumerate(pd.read_csv(source,usecols=use,dtype=str,chunksize=200000)):
  parts.append(c);log(f'audit read {(i+1)*200000:,} rows (last chunk may be smaller)')
 d=pd.concat(parts,ignore_index=True);d['source_row']=np.arange(len(d))+2
 d['issue_month']=month(d.issue_d).dt.strftime('%Y-%m');d['year']=month(d.issue_d).dt.year
 d['term_n']=pd.to_numeric(d.term.str.extract(r'(\d+)',expand=False),errors='coerce')
 csv(t/'cohort_status_summary.csv',d.groupby(['year','term_n','loan_status'],dropna=False).size().rename('n').reset_index())
 miss=[]
 for yr,g in [('all',d)]+list(d.groupby('year')):
  for f in RAW: miss.append(dict(year=yr,field=f,n=len(g),missing=int(g[f].isna().sum()),missing_rate=float(g[f].isna().mean())))
 csv(t/'feature_missingness_by_year.csv',miss)
 valid=d.id.str.fullmatch(r'\d+',na=False)&d.issue_month.notna()
 ids=d.loc[valid,'id'];dup=int(ids.duplicated().sum())
 assert dup==0,'Duplicate valid loan IDs: investigate before fitting'
 date=[]
 for f in ['issue_d','last_pymnt_d','last_credit_pull_d']:
  v=month(d[f]);date.append(dict(field=f,min=str(v.min()),max=str(v.max()),invalid=int((d[f].notna()&v.isna()).sum()),is_snapshot_evidence=False))
 csv(t/'date_audit.csv',date)
 evidence=ROOT/'provenance.md'
 save(out/'source_metadata.json',dict(source=str(source),sha256=digest,bytes=source.stat().st_size,rows=len(d),columns=len(cols),column_names=cols,
  dataset='wordsforthewise/lending-club',source_url='https://www.kaggle.com/datasets/wordsforthewise/lending-club',version='historical local audit identifies version3_2018Q4; current hash matches historical freeze',
  expected_historical_sha256='55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d',
  confirmed_snapshot=None,reference_month_assumption='2019-03',license='No standalone license grant verified locally; local research only; raw data not redistributed',
  publication_provenance_sha256=sha(evidence),publication_provenance=str(evidence),
  member_id_nonmissing=int(d.loc[valid,'member_id'].notna().sum()),duplicate_loan_id=dup,invalid_rows=int((~valid).sum()),created_at=now()))
 assert digest=='55c16f75120f897683f02e7aabcf080d0e4a20c4832feb1d592cfa941bd62a2d','Unexpected source version; audit configuration required'
 d.to_parquet(out/'audit_input.parquet',index=False)
 md(out/'data_audit.md',f'''# 数据审计与资产盘点

原始文件 {source.name}：{len(d):,} 行、{len(cols)} 列，SHA-256 `{digest}`。合法贷款ID重复 {dup} 条；异常/尾注行 {(~valid).sum()} 条。member_id 非空 {d.loc[valid,'member_id'].notna().sum()} 条。

来源：Kaggle wordsforthewise/lending-club，本地历史核查记录为 version3_2018Q4；本次全文件哈希与历史冻结值一致。历史核查保存上传日期2019-04-10及记录最晚2019-03，不能证明统一状态截至日。未发现可重建拟合时点的逐笔标签确认记录。旧版2017文件也不能重建本研究拟合时点，故不拼接历史快照冒充严格回测。

**研究降级：单文件已结清/已确认坏样本的放款批次回溯研究。快照日期未证实，不声称完成严格成熟度控制。** 2019-03仅用于假设账龄筛查，6个月缓冲是保守研究设定而非已证实充分成熟期；同时检查3/6/9个月。排除未决样本有选择偏差。

旧 scorecard_v1 是四字段、600/odds20/PDO50、同分拆分的不同实验；本次不复用模型或数字。复用原始数据和已保存来源证据，新增多字段分箱与训练实现。风控1既有研究已看过2014/2015批次，因此本次OOT仅为新流水线开发隔离的较晚批次，不能称项目层面的全新未见外部验证。

授权边界：本地未核实独立许可证条款，产物仅本地研究，不上传或重新分发原始数据。未改变已有项目结果。环境原缺libomp，使用已有RiskLab所保存的本地OpenMP库及许可证副本，运行入口指定本地库目录。

文件：cohort_status_summary.csv、feature_missingness_by_year.csv、date_audit.csv。全部151字段登记；非白名单一律禁用。verification_status为核验流程结果，无法证实初始申请前可得，进一步从主模型排除；issue_d仅用于切分及信用历史的代理时间。
''')
 return d
