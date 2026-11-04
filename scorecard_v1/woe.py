"""WOE/IV is estimated only on occupied training bins."""
import numpy as np
import pandas as pd

from .common import FEATURES, require


class WOE:
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def fit(self, bins, y, catalog):
        y = np.asarray(y, int)
        require(len(y) == len(bins) and np.isin(y, [0, 1]).all(), 'invalid WOE labels')
        B, G = int(y.sum()), int((y == 0).sum())
        require(B > 0 and G > 0, 'both training classes required')
        self.mapping_, self.rows_ = {}, []
        for f in FEATURES:
            occupied = set(bins[f])
            require(occupied <= set(catalog[f]), 'unknown fitting bins')
            m = len(occupied)
            mapping = {}
            for k in catalog[f]:
                mask = bins[f].eq(k).to_numpy()
                n, b = int(mask.sum()), int(y[mask].sum())
                g = n - b
                pg = (g + self.alpha)/(G + self.alpha*m) if n else 0.0
                pb = (b + self.alpha)/(B + self.alpha*m) if n else 0.0
                w = float(np.log(pg/pb)) if n else 0.0
                iv = float((pg-pb)*w) if n else 0.0
                mapping[k] = w
                self.rows_.append({'field': f, 'bin': k, 'n': n, 'good': g, 'bad': b,
                                   'chargeoff_rate': b/n if n else None, 'P_good': pg, 'P_bad': pb,
                                   'woe': w, 'iv_component': iv, 'occupied_bins': m,
                                   'total_good': G, 'total_bad': B,
                                   'evidence': 'training_observed' if n else 'no_training_evidence_neutral'})
            self.mapping_[f] = mapping
        totals = {f: sum(r['iv_component'] for r in self.rows_ if r['field'] == f) for f in FEATURES}
        for r in self.rows_:
            r['field_iv'] = totals[r['field']]
        return self

    def transform(self, bins):
        out = pd.DataFrame(index=bins.index)
        for f in FEATURES:
            out[f] = bins[f].map(self.mapping_[f])
        require(np.isfinite(out.to_numpy()).all(), 'WOE mapping incomplete/nonfinite')
        return out
