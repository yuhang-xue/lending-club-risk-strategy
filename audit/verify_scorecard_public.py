"""Verify published scorecard aggregates without private inputs or model files."""
import csv
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from scorecard_v1.evaluate import psi


def verify(root=ROOT):
    folder=root/'results/scorecard';manifest=json.loads((folder/'publication_manifest.json').read_text())
    for name,expected in manifest['files'].items():
        p=folder/name
        assert p.is_file() and p.stat().st_size==expected['bytes'],name
        assert hashlib.sha256(p.read_bytes()).hexdigest()==expected['sha256'],name
        if p.suffix=='.csv':
            columns=next(csv.reader(p.open(encoding='utf-8-sig')))
            assert not {'loan_id','source_row','id','risk_S1','risk_SC_WOE','eta','tie_sha256'}.intersection(columns),name
    for name,digest in manifest['exported_core_source_sha256'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    metrics=pd.read_csv(folder/'comparison/scenario_metrics.csv')
    assert len(metrics)==36 and not metrics.duplicated(['window','strategy','retention_percent']).any()
    for _,r in metrics.iterrows():
        assert r.B_keep+r.B_remove==r.B and r.G_keep+r.G_remove==r.G and r.K+r.R==r.N
        assert r.K==int(r.retention_percent)*int(r.N)//100
        assert abs(r.retained_chargeoff_ratio-r.B_keep/r.K)<1e-12
    intervals=pd.read_csv(folder/'comparison/bootstrap_intervals.csv')
    assert len(intervals)==6 and not intervals.duplicated(['year','reference']).any()
    for _,r in intervals.iterrows():
        pair=metrics[(metrics.year==r.year)&(metrics.retention_percent==80)].set_index('strategy')
        delta=pair.loc['SC_WOE','retained_chargeoff_ratio']-pair.loc[r.reference,'retained_chargeoff_ratio']
        assert abs(delta-r.point_difference)<1e-12
        assert r.lower<=r.point_difference<=r.upper
        assert r.effective_repetitions==2000 and r.invalid_repetitions==0
    main=intervals[intervals.reference=='S1']
    assert len(main)==3 and (main.lower<0).all() and (main.upper>0).all()
    bins=pd.read_csv(folder/'fixed_W1_transfer/bin_statistics.csv')
    drift=pd.read_csv(folder/'fixed_W1_transfer/psi_summary.csv')
    assert len(drift)==10
    for _,r in drift.iterrows():
        a=bins[(bins.year==2014)&(bins.field==r.field)].set_index('bin').n
        b=bins[(bins.year==r.year)&(bins.field==r.field)].set_index('bin').n
        assert a.index.equals(b.index)
        assert abs(psi(a.to_numpy(),b.to_numpy(),.5)[0]-r.psi)<1e-12
    return {'status':'passed','aggregate_files':len(manifest['files']),
            'scenarios':36,'intervals':6,'fixed_card_psi':10,
            'checks':['file and source hashes','aggregate-only schemas','count conservation',
                      'retention denominators','interval point estimates','three S1 intervals cross zero',
                      'PSI recomputed from same frozen-bin counts'],
            'does_not_retrain_or_establish_independent_validation':True}


if __name__=='__main__':print(json.dumps(verify(),ensure_ascii=False))
