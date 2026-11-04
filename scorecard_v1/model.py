"""Four WOE features, fixed L2 logistic regression, exact additive points."""
import warnings
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression

from .binning import Binner, clean
from .woe import WOE
from .common import FEATURES, digest_object, require


def constants(config):
    B = config['pdo']/np.log(2)
    return config['base_score'] - B*np.log(config['base_good_bad_odds']), B


class Scorecard:
    def __init__(self, config):
        self.config = config

    def fit(self, X, y):
        clean(X)  # Formal model never accepts invalid FICO.
        self.binner_ = Binner(self.config).fit(X, y)
        bins = self.binner_.transform(X)
        self.woe_ = WOE(self.config['alpha']).fit(bins, y, self.binner_.catalog_)
        params = self.config['model'].copy()
        params.update({'l1_ratio': 0.0} if tuple(map(int, sklearn.__version__.split('.')[:2])) >= (1, 8)
                      else {'penalty': 'l2'})
        self.lr_ = LogisticRegression(**params)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            self.lr_.fit(self.woe_.transform(bins), y)
        self.warnings_ = [{'category': w.category.__name__, 'message': str(w.message)} for w in caught]
        require(not self.warnings_, 'training warning: ' + str(self.warnings_))
        require(np.array_equal(self.lr_.classes_, [0, 1]), 'class order')
        require(np.isfinite(self.lr_.coef_).all() and np.isfinite(self.lr_.intercept_).all(), 'nonfinite coefficient')
        return self

    def predict(self, X):
        clean(X)
        bins = self.binner_.transform(X)
        w = self.woe_.transform(bins)
        eta = self.lr_.decision_function(w)
        p = self.lr_.predict_proba(w)[:, 1]
        A, B = constants(self.config)
        points = (A-B*self.lr_.intercept_[0])/4 - B*w.to_numpy()*self.lr_.coef_[0]
        total = A - B*eta
        require(np.isfinite(p).all() and np.isfinite(total).all() and ((p >= 0) & (p <= 1)).all(), 'invalid prediction')
        require(np.max(np.abs(points.sum(axis=1)-total), initial=0) <= self.config['tolerances']['points'], 'points do not sum')
        return {'bins': bins, 'woe': w, 'eta': eta, 'risk': p, 'score': total, 'points': points}

    def specification(self):
        return {'binning': self.binner_.specification(), 'woe': self.woe_.mapping_,
                'coefficients': self.lr_.coef_[0].tolist(), 'intercept': float(self.lr_.intercept_[0]),
                'parameters': self.lr_.get_params(), 'n_iter': self.lr_.n_iter_.tolist(),
                'warnings': self.warnings_}

    def state_hash(self):
        return digest_object(self.specification())

    def score_table(self):
        A, B = constants(self.config)
        rows = []
        for r in self.woe_.rows_:
            j = FEATURES.index(r['field'])
            rows.append({**r, 'coefficient': float(self.lr_.coef_[0, j]),
                         'allocated_constant': float((A-B*self.lr_.intercept_[0])/4),
                         'points': float((A-B*self.lr_.intercept_[0])/4-B*self.lr_.coef_[0, j]*r['woe']),
                         'definition': self.binner_.definition(r['field'], r['bin'])})
        return rows
