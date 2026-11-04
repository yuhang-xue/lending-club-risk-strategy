"""Frozen rankings, exact paired reselection, profiles and drift statistics."""
import hashlib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .binning import clean
from .common import FEATURES, MISSING, UNKNOWN, STRATEGIES, require


def ranking(ids, risk):
    ids, risk = np.asarray(ids, str), np.asarray(risk, float)
    require(len(ids) == len(risk) and len(set(ids)) == len(ids) and np.isfinite(risk).all(), 'invalid ranking input')
    ties = np.array([hashlib.sha256(('20261022:' + i).encode()).hexdigest() for i in ids])
    order = np.lexsort((ids, ties, risk))
    rank = np.empty(len(ids), dtype=np.int64)
    rank[order] = np.arange(1, len(ids) + 1)
    return order, rank, ties


def ks(y, score):
    y, score = np.asarray(y, int), np.asarray(score, float)
    b, g = int(y.sum()), int((y == 0).sum())
    if not b or not g:
        return None
    order = np.argsort(score, kind='stable')
    sy, ss = y[order], score[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(y)-1]
    return float(np.max(np.abs(np.cumsum(sy)[ends]/b - np.cumsum(1-sy)[ends]/g)))


def metrics(y, risk, keep):
    y, keep = np.asarray(y, int), np.asarray(keep, bool)
    n, b, k = len(y), int(y.sum()), int(keep.sum())
    bk, gr = int(y[keep].sum()), int(((y == 0) & ~keep).sum())
    return {'N': n, 'B': b, 'G': n-b, 'K': k, 'R': n-k,
            'B_keep': bk, 'B_remove': b-bk, 'G_keep': k-bk, 'G_remove': gr,
            'actual_retained_ratio': k/n if n else None,
            'retained_chargeoff_ratio': bk/k if k else None,
            'chargeoff_capture_rate': (b-bk)/b if b else None,
            'paid_exclusion_rate': gr/(n-b) if n-b else None,
            'auc': float(roc_auc_score(y, risk)) if 0 < b < n else None,
            'ks': ks(y, risk), 'not_applicable_reason': '' if 0 < b < n and k else 'empty_or_single_class'}


def weighted_bad_count(weights, order, y, k):
    weights = np.asarray(weights, np.int64)
    sw = weights[order]
    cumulative = np.cumsum(sw)
    require(0 < k <= int(cumulative[-1]), 'bootstrap k invalid')
    selected = np.minimum(sw, np.maximum(k - (cumulative-sw), 0))
    require(int(selected.sum()) == k, 'bootstrap keep count')
    return int(selected @ y[order])


def bootstrap(ids, y, risks, year, repetitions=2000, progress=None):
    y = np.asarray(y, np.int64)
    n, k = len(y), 80*len(y)//100
    orders = {s: ranking(ids, risks[s])[0] for s in STRATEGIES}
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261022, year])))
    rows = []
    for r in range(repetitions):
        sample = rng.integers(0, n, size=n, endpoint=False, dtype=np.int64)
        weights = np.bincount(sample, minlength=n)
        counts = {s: weighted_bad_count(weights, orders[s], y, k) for s in STRATEGIES}
        rows.append({'year': year, 'replicate': r, 'N': n, 'K': k,
                     **{'B_keep_' + s: counts[s] for s in STRATEGIES},
                     'SC_WOE_minus_S1': (counts['SC_WOE']-counts['S1'])/k,
                     'SC_WOE_minus_S0': (counts['SC_WOE']-counts['S0'])/k})
        if progress and (r + 1) % 250 == 0:
            progress(f'bootstrap {year}: {r+1}/{repetitions}')
    return pd.DataFrame(rows)


def exchange(raw, decisions, reference):
    old = decisions['keep_' + reference + '_80'].to_numpy(bool)
    new = decisions['keep_SC_WOE_80'].to_numpy(bool)
    masks = {'common_keep': old & new, 'swap_out': old & ~new,
             'swap_in': ~old & new, 'common_remove': ~old & ~new}
    require(int(masks['swap_out'].sum()) == int(masks['swap_in'].sum()), 'unequal swap size')
    x = clean(raw.loc[:, FEATURES])
    y = raw.y.to_numpy(int)
    summary, profiles, marks = [], [], np.empty(len(raw), dtype=object)
    for group, mask in masks.items():
        marks[mask] = group
        n, b = int(mask.sum()), int(y[mask].sum())
        summary.append({'reference': reference, 'candidate': 'SC_WOE', 'group': group,
                        'n': n, 'bad': b, 'good': n-b, 'share': n/len(raw), 'rate': b/n if n else None})
        vals = x.loc[mask, 'fico_n'].to_numpy(float)
        q = np.quantile(vals, [.25, .5, .75], method='linear').tolist() if n else [None]*3
        profiles.append({'reference': reference, 'group': group, 'field': 'fico_n', 'kind': 'continuous',
                         'category': '', 'n': n, 'count': n, 'share': 1.0 if n else None,
                         'p25': q[0], 'median': q[1], 'p75': q[2]})
        for f in FEATURES[1:]:
            for cat in sorted(set(x[f]) | {MISSING}):
                count = int((x[f].eq(cat).to_numpy() & mask).sum())
                profiles.append({'reference': reference, 'group': group, 'field': f, 'kind': 'categorical',
                                 'category': cat, 'n': n, 'count': count,
                                 'share': count/n if n else None, 'p25': None, 'median': None, 'p75': None})
    delta = int(y[new].sum()-y[old].sum())
    require(delta == int(y[masks['swap_in']].sum()-y[masks['swap_out']].sum()), 'exchange identity')
    return summary, profiles, marks


def wilson(b, n):
    if not n:
        return None, None
    z = 1.959963984540054
    p, a = b/n, z*z/n
    mid = (p + a/2)/(1+a)
    half = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+a)
    return float(max(0, mid-half)), float(min(1, mid+half))


def bin_statistics(bins, y, catalog, year, role):
    y = np.asarray(y, int)
    rows = []
    for f, universe in catalog.items():
        for k in universe:
            mask = np.asarray(bins[f]) == k
            n, b = int(mask.sum()), int(y[mask].sum())
            lo, hi = wilson(b, n)
            rows.append({'year': year, 'role': role, 'field': f, 'bin': k,
                         'n': n, 'bad': b, 'good': n-b, 'share': n/len(y),
                         'chargeoff_rate': b/n if n else None, 'wilson_lower': lo, 'wilson_upper': hi,
                         'unstable': n < 300 or min(b, n-b) < 20,
                         'is_missing': k == MISSING, 'is_unknown': k == UNKNOWN})
    return rows


def psi(reference_counts, target_counts, alpha=0.5):
    a, b = np.asarray(reference_counts, float), np.asarray(target_counts, float)
    require(a.shape == b.shape and len(a) > 0 and (a >= 0).all() and (b >= 0).all(), 'invalid PSI counts')
    p = (a+alpha)/(a.sum()+alpha*len(a))
    q = (b+alpha)/(b.sum()+alpha*len(b))
    parts = (q-p)*np.log(q/p)
    return float(parts.sum()), p, q, parts


def monotonicity(rows):
    rows = sorted([r for r in rows if r['field'] == 'fico_n' and str(r['bin']).startswith('N') and r['n']],
                  key=lambda r: int(r['bin'][1:]))
    return [{'lower_bin': a['bin'], 'higher_bin': b['bin'], 'lower_rate': a['chargeoff_rate'],
             'higher_rate': b['chargeoff_rate'], 'lower_n': a['n'], 'higher_n': b['n'],
             'expected_nonincreasing_violation': b['chargeoff_rate'] > a['chargeoff_rate']}
            for a, b in zip(rows[:-1], rows[1:])]
