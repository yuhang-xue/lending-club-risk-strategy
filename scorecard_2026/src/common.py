from pathlib import Path
import hashlib,json
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
NUM=['loan_amnt','annual_inc','dti','delinq_2yrs','inq_last_6mths','open_acc','pub_rec','pub_rec_bankruptcies','tax_liens','mort_acc','total_acc','revol_util','revol_bal']
CAT=['purpose','home_ownership','addr_state']
RAW=NUM+CAT+['emp_length','fico_range_low','fico_range_high','earliest_cr_line','verification_status']
FEATURES=NUM+CAT+['emp_length','fico_mean','log_income','loan_income','revol_income','credit_history_months','income_invalid']
def now(): return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(2**20),b''): h.update(b)
 return h.hexdigest()
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=lambda o:o.item() if isinstance(o,np.generic) else str(o)),encoding='utf8')
def csv(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 pd.DataFrame(x).to_csv(p,index=False,encoding='utf-8-sig')
def md(p,s): Path(p).write_text(s,encoding='utf8')
def month(s): return pd.to_datetime(s,format='%b-%Y',errors='coerce')
def features(d):
 x=pd.DataFrame(index=d.index)
 for f in NUM: x[f]=pd.to_numeric(d[f].astype('string').str.replace('%','',regex=False),errors='coerce')
 for f in NUM:
  if f!='dti': x.loc[x[f]<0,f]=np.nan
 for f in CAT: x[f]=d[f].astype('string').fillna('__MISSING__').astype(str)
 x['emp_length']=pd.to_numeric(d.emp_length.astype('string').str.extract(r'(\d+)',expand=False),errors='coerce')
 x.loc[d.emp_length.astype('string').str.startswith('<',na=False),'emp_length']=0
 lo=pd.to_numeric(d.fico_range_low,errors='coerce');hi=pd.to_numeric(d.fico_range_high,errors='coerce')
 x['fico_mean']=((lo+hi)/2).where(lo.between(300,850)&hi.between(300,850)&(lo<=hi))
 inc=x.annual_inc.where(x.annual_inc>0)
 x['income_invalid']=(inc.isna()).astype(float)
 x['log_income']=np.log1p(inc);x['loan_income']=x.loan_amnt/inc;x['revol_income']=x.revol_bal/inc
 a=month(d.issue_d);b=month(d.earliest_cr_line)
 hist=(a.dt.year-b.dt.year)*12+a.dt.month-b.dt.month
 x['credit_history_months']=hist.where(hist>=0)
 x=x[FEATURES].replace([np.inf,-np.inf],np.nan)
 return x

def score(p):
 p=np.clip(np.asarray(p),1e-9,1-1e-9)
 return 600-20/np.log(2)*np.log(50)+20/np.log(2)*np.log((1-p)/p)
def probability(s):
 odds=np.exp((np.asarray(s)-(600-20/np.log(2)*np.log(50)))/(20/np.log(2)))
 return 1/(1+odds)
