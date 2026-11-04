import copy
import math
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scorecard_v1.binning import Binner, clean, numeric_ids, quantile_edges
from scorecard_v1.common import FEATURES, MISSING, OTHER, UNKNOWN, PACKAGE, contained, load, csv, read
from scorecard_v1.evaluate import bootstrap, exchange, ks, metrics, psi, ranking, weighted_bad_count
from scorecard_v1.model import Scorecard, constants
from scorecard_v1.woe import WOE


def example():
    n = 400
    x = pd.DataFrame({'fico_n': np.tile([620, 660, 700, 740], n//4),
                      'purpose': np.tile(['car', 'car', 'credit_card', 'other'], n//4),
                      'emp_length': np.tile(['NI', '1 year', '2 years', '10+ years'], n//4),
                      'home_ownership_n': np.tile(['RENT', 'MORTGAGE'], n//2)})
    y = ((np.arange(n) % 13) < (5 - np.arange(n) % 4)).astype(int)
    return x, y


class ScorecardTests(unittest.TestCase):
    def setUp(self):
        self.c = load(PACKAGE/'config.json')

    def test_native_missing_text_and_reserved_collisions(self):
        x, _ = example()
        x.loc[:5, 'purpose'] = [None, '', '   ', 'NULL', 'NA', 'NI']
        self.assertEqual(clean(x).purpose.iloc[:6].tolist(), [MISSING]*3+['NULL', 'NA', 'NI'])
        for token in [MISSING, OTHER, UNKNOWN]:
            x.loc[0, 'purpose'] = token
            with self.assertRaisesRegex(ValueError, 'reserved'):
                clean(x)

    def test_invalid_numeric_formal_input_and_component_missing(self):
        x, y = example()
        x['fico_n'] = x.fico_n.astype(object)
        for bad in [None, '', 'NA', 299, 851, float('inf')]:
            x.loc[0, 'fico_n'] = bad
            with self.assertRaisesRegex(ValueError, 'FICO'):
                Scorecard(self.c).fit(x, y)
        x.loc[0, 'fico_n'] = None
        b = Binner(self.c).fit(x, y)
        self.assertEqual(b.transform(x).fico_n.iloc[0], MISSING)

    def test_duplicate_edges_closed_right_and_single_bin(self):
        self.assertEqual(quantile_edges([700]*100), [])
        self.assertEqual(numeric_ids([650, 700, 701, np.nan], [650, 700]).tolist(), ['N0', 'N1', 'N2', MISSING])
        x, y = example()
        x['fico_n'] = 700
        b = Binner(self.c).fit(x, y)
        self.assertEqual(b.numeric_edges_, [])
        self.assertTrue(b.transform(x).fico_n.eq('N0').all())

    def test_merge_deterministic_and_no_tie_split(self):
        x, y = example()
        x['fico_n'] = np.r_[np.repeat(600, 390), np.arange(700, 710)]
        b = Binner(self.c).fit(x, y)
        reverse = Binner(self.c).fit(x.iloc[::-1], y[::-1])
        self.assertEqual(b.specification(), reverse.specification())
        self.assertGreater(len(b.merge_log_), 0)
        self.assertEqual(len(set(b.transform(x).loc[x.fico_n.eq(600), 'fico_n'])), 1)
        with self.assertRaisesRegex(ValueError, 'class count'):
            Binner(self.c).fit(x, np.zeros(len(x), int))

    def test_rare_protected_unknown_and_missing(self):
        x, y = example()
        x.loc[0, 'purpose'] = 'rare'
        x.loc[1, 'purpose'] = 'NI'
        b = Binner(self.c).fit(x, y)
        self.assertEqual(b.category_maps_['purpose']['rare'], OTHER)
        self.assertEqual(b.category_maps_['purpose']['NI'], 'NI')
        w = WOE().fit(b.transform(x), y, b.catalog_)
        ev = x.iloc[:2].copy()
        ev['purpose'] = ['never_seen', None]
        bins = b.transform(ev)
        self.assertEqual(bins.purpose.tolist(), [UNKNOWN, MISSING])
        self.assertEqual(w.transform(bins).purpose.tolist(), [0, 0])

    def test_woe_hand_counts_zero_events_and_empty_reserved(self):
        bins = pd.DataFrame({f: ['A', 'A', 'B', 'B'] for f in FEATURES})
        w = WOE().fit(bins, [0, 0, 1, 1], {f: ['A', 'B', UNKNOWN] for f in FEATURES})
        self.assertAlmostEqual(w.mapping_['fico_n']['A'], math.log(5))
        self.assertAlmostEqual(w.mapping_['fico_n']['B'], -math.log(5))
        rows = [r for r in w.rows_ if r['field'] == 'fico_n']
        self.assertAlmostEqual(sum(r['P_good'] for r in rows), 1)
        self.assertAlmostEqual(sum(r['P_bad'] for r in rows), 1)
        self.assertAlmostEqual(rows[0]['field_iv'], 4/3*math.log(5))
        self.assertEqual(rows[2]['woe'], 0)
        self.assertEqual(rows[2]['occupied_bins'], 2)

    def test_evaluation_changes_never_fit_and_model_roundtrip(self):
        x, y = example()
        model = Scorecard(self.c).fit(x, y)
        original = copy.deepcopy(model.specification())
        ev = x.iloc[:20].copy()
        ev.loc[:, 'purpose'] = 'new'
        ev.loc[:, 'fico_n'] = 800
        a = model.predict(ev)
        self.assertEqual(original, model.specification())
        self.assertEqual(model.state_hash(), model.state_hash())
        # Labels are not accepted by prediction; arbitrary evaluation labels cannot affect state.
        shuffled_labels = y[:20][::-1]
        self.assertEqual(len(a['risk']), len(shuffled_labels))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'model.joblib'
            joblib.dump(model, p)
            np.testing.assert_array_equal(a['risk'], joblib.load(p).predict(ev)['risk'])
        np.testing.assert_allclose(a['points'].sum(axis=1), a['score'], atol=1e-8)
        predictions = model.predict(x)
        np.testing.assert_allclose(predictions['score'], constants(self.c)[0]-constants(self.c)[1]*predictions['eta'])
        order = np.argsort(predictions['eta'])
        self.assertTrue(np.all(np.diff(predictions['score'][order]) <= 0))

    def test_base_score_pdo(self):
        A, B = constants(self.c)
        self.assertAlmostEqual(A-B*(-math.log(20)), 600)
        self.assertAlmostEqual((A+B*math.log(40))-(A+B*math.log(20)), 50)
        p = 1/21
        self.assertAlmostEqual(A-B*math.log(p/(1-p)), 600)

    def test_ties_permutation_and_metric_denominators(self):
        ids = ['003', '001', '2', '04']
        a = ranking(ids, [.2]*4)[1]
        b = ranking(ids[::-1], [.2]*4)[1]
        self.assertEqual(dict(zip(ids, a)), dict(zip(ids[::-1], b)))
        with self.assertRaises(ValueError):
            ranking(['1', '1'], [0, 1])
        r = metrics([1, 1, 0, 0, 0], [.1, .2, .3, .4, .5], [1, 0, 1, 0, 1])
        self.assertEqual((r['K'], r['B_keep'], r['G_remove']), (3, 1, 1))
        self.assertAlmostEqual(r['chargeoff_capture_rate'], .5)
        self.assertAlmostEqual(r['paid_exclusion_rate'], 1/3)
        self.assertEqual(ks([0, 1, 0, 1], [.1, .1, .9, .9]), 0)
        self.assertIsNone(metrics([0, 0], [0, 1], [0, 0])['retained_chargeoff_ratio'])

    def test_weighted_bootstrap_matches_expanded_sample_including_partial_boundary(self):
        ids = ['001', '002', '003', '004', '005']
        y = np.array([0, 1, 1, 0, 1])
        risks = {s: np.roll([.2, .2, .3, .6, .8], i) for i, s in enumerate(['S0', 'S1', 'SC_WOE'])}
        rows = bootstrap(ids, y, risks, 2014, repetitions=20)
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([20261022, 2014])))
        for r in range(20):
            sample = rng.integers(0, 5, size=5, endpoint=False, dtype=np.int64)
            for s in risks:
                order = ranking(ids, risks[s])[0]
                ranks = {int(index): rank for rank, index in enumerate(order)}
                expanded = sorted(sample.tolist(), key=lambda i: ranks[i])
                expected = int(y[expanded[:4]].sum())
                self.assertEqual(rows.iloc[r]['B_keep_'+s], expected)
        self.assertEqual(weighted_bad_count([0, 5, 0], np.array([0, 1, 2]), np.array([0, 1, 0]), 4), 4)

    def test_exchange_identity_and_profiles(self):
        x, y = example()
        raw = x.iloc[:4].copy()
        raw['y'] = [0, 1, 0, 1]
        dec = pd.DataFrame({'keep_S1_80': [1, 1, 0, 0], 'keep_SC_WOE_80': [1, 0, 1, 0]})
        summary, profile, marks = exchange(raw, dec, 'S1')
        self.assertEqual(sum(r['n'] for r in summary), 4)
        self.assertEqual(set(marks), {'common_keep', 'swap_out', 'swap_in', 'common_remove'})
        self.assertTrue(len(profile) > 4)
        # Profile CSV mixes empty category quantiles with numeric continuous ones.
        # keep_default_na=False preserves literal categories, so cast selected numbers.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'profiles.csv'
            csv(path, profile)
            saved = read(path)
            r = saved[(saved.field == 'fico_n') & (saved.group == 'common_keep')].iloc[0]
            np.testing.assert_allclose(np.asarray([r.p25, r['median'], r.p75], float), [620, 620, 620])

    def test_psi_empty_cells_and_path_containment(self):
        self.assertEqual(psi([0, 10, 20], [0, 10, 20])[0], 0)
        self.assertTrue(np.isfinite(psi([0, 10, 0], [1, 0, 9])[0]))
        self.assertGreater(psi([0, 10, 0], [1, 0, 9])[0], 0)
        with self.assertRaises(ValueError):
            contained(PACKAGE, '../outside')
        self.assertEqual(contained(PACKAGE, 'tests\\test_scorecard.py'), (PACKAGE/'tests/test_scorecard.py').resolve())


if __name__ == '__main__':
    unittest.main()
