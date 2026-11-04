"""Reproduce the frozen four-field scorecard from the publisher CSV and audited baseline."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from prepare import load_source, sha
from scorecard_v1.common import FEATURES, load, require
from scorecard_v1.model import Scorecard
from scorecard_v1.binning import quantile_edges,numeric_ids
from scorecard_v1.evaluate import metrics,ranking,bootstrap,psi


def emit(path,rows):
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path,index=False,encoding='utf-8-sig')


def run(data,baseline,output):
    require(not output.exists(),'Output directory must not exist')
    # Verify original S0/S1/S2 replay before trusting its prediction files.
    subprocess.run([sys.executable,str(ROOT/'audit/verify.py'),'--run',str(baseline.resolve())],check=True)
    require(sha(data)==load(baseline/'sample_audit.json')['source_sha256'],'Baseline source differs')
    output.mkdir(parents=True)
    record={'status':'running','method':'public scorecard replay of frozen 20261102 configuration',
            'independent_validation':False,'full_bootstrap_repetitions_per_year':2000,
            'baseline_run_record_sha256':sha(baseline/'run_record.json'),
            'config_sha256':sha(ROOT/'scorecard_v1/config.json')}
    try:
        raw=load_source(data,output);config=load(ROOT/'scorecard_v1/config.json')
        rows=[];intervals=[];models={};count_refs=pd.read_csv(ROOT/'results/scorecard/comparison/scenario_metrics.csv')
        for name,spec in config['windows'].items():
            fit=raw[raw.issue_month.le(spec['train_end'])].reset_index(drop=True)
            ev=raw[raw.approval_year.eq(spec['year'])].reset_index(drop=True)
            require(len(fit)==spec['train_count'] and len(ev)==spec['count'],'Window counts differ')
            p=pd.read_csv(baseline/name/'loan_predictions.csv',dtype={'loan_id':str},float_precision='round_trip')
            require(p.loan_id.is_unique and set(p.loan_id)==set(ev.loan_id),'Baseline identities differ')
            p=p.set_index('loan_id').loc[ev.loan_id].reset_index()
            require(np.array_equal(p.y,ev.y),'Baseline labels differ')
            print(f'{name}: fit four-field scorecard, evaluate {len(ev):,}',flush=True)
            with threadpool_limits(limits=1):
                card=Scorecard(config).fit(fit.loc[:,FEATURES],fit.y.to_numpy(int))
                pred=card.predict(ev.loc[:,FEATURES])
            models[name]=card
            folder=output/name;folder.mkdir()
            joblib.dump(card,folder/'model.joblib',compress=3)
            risk={'S0':p.risk_S0.to_numpy(),'S1':p.risk_S1.to_numpy(),'SC_WOE':pred['risk']}
            emit(folder/'score_table.csv',[{**r,'definition':json.dumps(r['definition'],sort_keys=True)} for r in card.score_table()])
            for strategy,score in risk.items():
                _,rank,_=ranking(ev.loan_id,score)
                for percent in config['retention_percentages']:
                    row={'window':name,'year':spec['year'],'strategy':strategy,'retention_percent':percent,
                         **metrics(ev.y,score,rank<=percent*len(ev)//100)}
                    reference=count_refs[(count_refs.window==name)&(count_refs.strategy==strategy)&(count_refs.retention_percent==percent)].iloc[0]
                    for field in ['N','B','G','K','R','B_keep','B_remove','G_keep','G_remove']:
                        require(row[field]==int(reference[field]),f'Historical count differs: {name}/{strategy}/{percent}/{field}')
                    rows.append(row)
            # Labels never enter ranking; each draw shares loan weights across methods.
            rep=bootstrap(ev.loan_id,ev.y,risk,spec['year'],progress=lambda s:print(s,flush=True))
            for ref in ['S1','S0']:
                selected={r['strategy']:r for r in rows if r['window']==name and r['retention_percent']==80}
                point=selected['SC_WOE']['retained_chargeoff_ratio']-selected[ref]['retained_chargeoff_ratio']
                lo,hi=np.quantile(rep['SC_WOE_minus_'+ref],[.025,.975],method='linear')
                intervals.append({'window':name,'year':spec['year'],'reference':ref,'point_difference':point,
                                  'lower':lo,'upper':hi,'point_difference_pp':100*point,
                                  'lower_pp':100*lo,'upper_pp':100*hi})
            # Per-loan output stays in ignored output, never in the public aggregate package.
            local=ev[['loan_id','issue_month','y']].copy();local['risk_SC_WOE']=pred['risk'];local['score']=pred['score']
            emit(folder/'loan_predictions.csv',local)
        emit(output/'scenario_metrics.csv',rows);emit(output/'bootstrap_intervals.csv',intervals)
        reference=pd.read_csv(ROOT/'results/scorecard/comparison/bootstrap_intervals.csv')
        got=pd.DataFrame(intervals)
        for column in ['point_difference','lower','upper']:
            np.testing.assert_allclose(got[column],reference[column],atol=1e-12,rtol=0)
        # Same W1 model and training-derived score bins for cross-year PSI.
        card=models['W1'];before=card.state_hash();fit=raw[raw.issue_month.le('2013-12')]
        edges=quantile_edges(card.predict(fit.loc[:,FEATURES])['score'])
        catalog={**card.binner_.catalog_,'score':[f'N{i}' for i in range(len(edges)+1)]}
        counts={}
        for year in [2014,2015,2016]:
            ev=raw[raw.approval_year.eq(year)];pred=card.predict(ev.loc[:,FEATURES]);bins=pred['bins'].copy()
            bins['score']=numeric_ids(pred['score'],edges)
            counts[year]={field:[int(bins[field].eq(k).sum()) for k in universe] for field,universe in catalog.items()}
        drift=[]
        for year in [2015,2016]:
            for field in catalog:
                drift.append({'year':year,'field':field,'psi':psi(counts[2014][field],counts[year][field],config['alpha'])[0]})
        require(before==card.state_hash(),'Fixed card changed during evaluation')
        emit(output/'psi_summary.csv',drift)
        ref=pd.read_csv(ROOT/'results/scorecard/fixed_W1_transfer/psi_summary.csv').set_index(['year','field'])
        for row in drift:require(abs(row['psi']-ref.loc[(row['year'],row['field']),'psi'])<=1e-12,'PSI differs')
        record.update(status='passed',scenario_count=len(rows),interval_count=len(intervals),psi_count=len(drift),
                      checks=['audited baseline','source hash','window and identity alignment','all historical integer counts',
                              '6000 paired bootstrap draws; intervals within 1e-12','fixed-W1 PSI within 1e-12',
                              'score decomposition and state invariance'],
                      source_sha256=sha(data),config=config,
                      output_hashes={p.name:sha(p) for p in output.glob('*.csv')})
    except Exception as error:
        record.update(status='failed',error=str(error));raise
    finally:
        (output/'verification.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['config','output_hashes']},ensure_ascii=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--baseline-run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.data,a.baseline_run,a.output)
