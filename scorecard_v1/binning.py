"""Deterministic training-only binning, with explicit unobserved bins."""
import math
import numpy as np
import pandas as pd

from .common import FEATURES, MISSING, OTHER, UNKNOWN, require


def clean(frame, allow_numeric_missing=False):
    require(list(frame.columns) == list(FEATURES), 'four raw feature columns/order required')
    out = frame.copy()
    for f in FEATURES[1:]:
        values = out[f].map(lambda v: '' if pd.isna(v) else str(v).strip())
        require(not values.isin([MISSING, OTHER, UNKNOWN]).any(), 'reserved category collision: ' + f)
        out[f] = values.mask(values.eq(''), MISSING)
    raw = out.fico_n
    text = raw.map(lambda v: '' if pd.isna(v) else str(v).strip())
    values = pd.to_numeric(text, errors='coerce').to_numpy(float)
    missing = text.eq('').to_numpy()
    valid = np.isfinite(values) & (values >= 300) & (values <= 850)
    require(np.all(valid | (missing & allow_numeric_missing)), 'FICO missing, invalid or out of range')
    out['fico_n'] = values
    return out


def numeric_ids(values, edges):
    values = np.asarray(values, dtype=float)
    positions = np.searchsorted(np.asarray(edges, float), values, side='left')
    return np.where(np.isnan(values), MISSING, np.char.add('N', positions.astype(str)))


def quantile_edges(values, n=10):
    values = np.asarray(values, float)
    values = values[np.isfinite(values)]
    if not len(values):
        return []
    if values.min() == values.max():
        return []
    # Endpoints are replaced by infinities. Duplicated internal quantiles collapse.
    edges = np.unique(np.quantile(values, np.linspace(0, 1, n + 1), method='linear')[1:-1])
    return edges.tolist()


class Binner:
    def __init__(self, config):
        self.config = config

    def fit(self, X, y):
        x = clean(X, allow_numeric_missing=True)
        y = np.asarray(y)
        require(len(x) == len(y) and np.isin(y, [0, 1]).all(), 'invalid fitting labels')
        minimum = self.config['minimum_class_count']
        require(int(y.sum()) >= minimum and int((y == 0).sum()) >= minimum, 'training class count below minimum')
        edges = quantile_edges(x.fico_n, self.config['numeric_prebins'])
        log = []
        while True:
            ids = numeric_ids(x.fico_n, edges)
            stats = []
            for i in range(len(edges) + 1):
                mask = ids == f'N{i}'
                n, bad = int(mask.sum()), int(y[mask].sum())
                stats.append({'index': i, 'n': n, 'bad': bad, 'good': n - bad,
                              'rate': bad/n if n else None})
            invalid = next((r for r in stats if r['n'] < self.config['numeric_min_share']*len(x)
                            or min(r['bad'], r['good']) < minimum), None)
            if invalid is None or len(stats) == 1:
                break
            i = invalid['index']
            neighbors = [j for j in (i - 1, i + 1) if 0 <= j < len(stats)]
            def distance(j):
                if invalid['rate'] is None or stats[j]['rate'] is None:
                    return math.inf
                return abs(invalid['rate'] - stats[j]['rate'])
            j = min(neighbors, key=lambda j: (distance(j), j))
            boundary = min(i, j)
            before = edges.copy()
            removed = edges.pop(boundary)
            combined_n = invalid['n'] + stats[j]['n']
            combined_bad = invalid['bad'] + stats[j]['bad']
            log.append({'step': len(log) + 1, 'reason': 'share_or_class_count_below_frozen_minimum',
                        'left_to_right_first_invalid': invalid, 'neighbor': stats[j],
                        'edges_before': before, 'removed_boundary': removed, 'edges_after': edges.copy(),
                        'combined_n': combined_n, 'combined_bad': combined_bad,
                        'combined_good': combined_n - combined_bad})
        self.numeric_edges_ = edges
        self.merge_log_ = log
        self.catalog_ = {'fico_n': [f'N{i}' for i in range(len(edges) + 1)] + [MISSING]}
        self.category_maps_ = {}
        self.category_log_ = []
        for f in FEATURES[1:]:
            mapping = {}
            for category in sorted(set(x[f])):
                mask = x[f].eq(category).to_numpy()
                n, bad = int(mask.sum()), int(y[mask].sum())
                rare = n < self.config['category_min_share']*len(x) or min(bad, n - bad) < minimum
                target = OTHER if rare and category not in ('NI', MISSING) else category
                mapping[category] = target
                self.category_log_.append({'field': f, 'category': category, 'bin': target,
                                           'n': n, 'bad': bad, 'good': n - bad, 'rare': rare,
                                           'protected': category in ('NI', MISSING)})
            self.category_maps_[f] = mapping
            self.catalog_[f] = sorted(set(mapping.values()) | {OTHER, MISSING, UNKNOWN})
        self.training_n_ = len(x)
        return self

    def transform(self, X):
        x = clean(X, allow_numeric_missing=True)
        out = pd.DataFrame(index=x.index)
        out['fico_n'] = numeric_ids(x.fico_n, self.numeric_edges_)
        for f in FEATURES[1:]:
            mapping = self.category_maps_[f]
            out[f] = x[f].map(mapping).fillna(UNKNOWN)
            # Unseen native missing has its own neutral reserved bin.
            out.loc[x[f].eq(MISSING), f] = MISSING
        return out

    def specification(self):
        return {'numeric_edges': self.numeric_edges_, 'interval': 'left_open_right_closed',
                'tails': ['-inf', '+inf'], 'category_maps': self.category_maps_,
                'catalog': self.catalog_, 'merge_log': self.merge_log_,
                'category_log': self.category_log_, 'training_n': self.training_n_}

    def definition(self, field, bin_id):
        if field != 'fico_n':
            categories = [k for k, v in self.category_maps_[field].items() if v == bin_id]
            return {'categories': categories, 'reserved': bin_id in (MISSING, OTHER, UNKNOWN)}
        if bin_id == MISSING:
            return {'missing': True}
        i = int(bin_id[1:])
        return {'lower_exclusive': self.numeric_edges_[i - 1] if i else '-inf',
                'upper_inclusive': self.numeric_edges_[i] if i < len(self.numeric_edges_) else '+inf'}
