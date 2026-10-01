import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from portfolio_risk_lab.core import (load_prices, split_returns, estimate_moments,
    portfolio_metrics, sample_portfolios, optimise_portfolios, buy_and_hold,
    drawdown, realised_metrics)
from portfolio_risk_lab.demo import generate_demo


class NumericalTests(unittest.TestCase):
    def test_single_exposure_matches_asset_moments(self):
        result = portfolio_metrics([1, 0], np.array([.1, .04]), np.diag([.04, .01]), .02)
        self.assertAlmostEqual(result["annual_return_estimate"], .1)
        self.assertAlmostEqual(result["annual_volatility"], .2)
        self.assertAlmostEqual(result["sharpe_estimate"], .4)

    def test_random_allocations_reproducible_and_feasible(self):
        args = (np.array([.08, .05]), np.diag([.04, .01]))
        first, metrics = sample_portfolios(*args, count=100, seed=8)
        second, _ = sample_portfolios(*args, count=100, seed=8)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first.sum(axis=1), 1)
        self.assertTrue((first >= 0).all())
        self.assertTrue(np.isfinite(metrics.to_numpy()).all())

    def test_min_variance_matches_analytical_two_asset_answer(self):
        allocations, frontier = optimise_portfolios(np.array([.10, .05]), np.diag([.04, .01]))
        # Independent assets: w1 = variance2 / (variance1+variance2) = 0.2.
        np.testing.assert_allclose(allocations["Minimum volatility"], [.2, .8], atol=1e-5)
        self.assertEqual(len(frontier), 40)
        self.assertTrue((np.diff(frontier.annual_return_estimate) >= -1e-8).all())
        self.assertTrue((np.diff(frontier.annual_volatility) >= -1e-7).all())

    def test_max_sharpe_beats_dense_two_asset_grid(self):
        mu, cov = np.array([.12, .04]), np.array([[.04, .001], [.001, .01]])
        allocations, _ = optimise_portfolios(mu, cov, .02)
        chosen = portfolio_metrics(allocations["Maximum Sharpe"], mu, cov, .02)["sharpe_estimate"]
        grid = max(portfolio_metrics([w, 1-w], mu, cov, .02)["sharpe_estimate"] for w in np.linspace(0,1,1001))
        self.assertGreaterEqual(chosen+1e-7, grid)

    def test_buy_and_hold_allows_weights_to_drift(self):
        returns = pd.DataFrame({"A": [.1, .1], "B": [0., 0.]})
        result = buy_and_hold(returns, [.5, .5])
        np.testing.assert_allclose(result, [.05, 1.105/1.05-1])
        self.assertNotAlmostEqual(result.iloc[1], .05)

    def test_first_day_loss_included_in_drawdown(self):
        np.testing.assert_allclose(drawdown(pd.Series([-.1, 1/9, -.2])), [-.1, 0, -.2], atol=1e-12)

    def test_tail_metrics_sign_and_order(self):
        result = realised_metrics(pd.Series([.01, -.02, .005, -.1, .02]))
        self.assertGreater(result["daily_var_95"], 0)
        self.assertGreaterEqual(result["daily_expected_shortfall_95"], result["daily_var_95"])
        self.assertAlmostEqual(result["daily_expected_shortfall_95"], .1)

    def test_invalid_weights_rejected(self):
        for weights in ([.5,.6], [-.2,1.2], [np.nan,.5]):
            with self.assertRaises(ValueError):
                portfolio_metrics(weights, np.array([.1,.2]), np.eye(2))

    def test_singular_covariance_rejected(self):
        with self.assertRaises(ValueError):
            estimate_moments(pd.DataFrame({"A":[.1,.2,.3], "B":[.1,.2,.3]}))

    def test_chronological_split_keeps_boundary_return(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"prices.csv"
            generate_demo(path)
            prices = load_prices(path)
            train, holdout = split_returns(prices)
            self.assertLess(train.index[-1], holdout.index[0])
            self.assertEqual(len(train)+len(holdout), len(prices)-1)
            position = prices.index.get_loc(holdout.index[0])
            np.testing.assert_allclose(holdout.iloc[0], prices.iloc[position]/prices.iloc[position-1]-1)

    def test_changing_holdout_cannot_change_estimated_moments(self):
        with tempfile.TemporaryDirectory() as directory:
            prices = generate_demo(Path(directory)/"prices.csv")
            original_train, test = split_returns(prices)
            altered = prices.copy()
            altered.loc[test.index, :] *= 2
            altered_train, _ = split_returns(altered)
            for original, new in zip(estimate_moments(original_train), estimate_moments(altered_train)):
                np.testing.assert_array_equal(original, new)

    def test_missing_nonpositive_and_duplicate_dates_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"prices.csv"
            original = generate_demo(path)
            for value in [np.nan, 0, -1, np.inf]:
                broken = original.copy(); broken.iloc[5,0] = value; broken.to_csv(path)
                with self.assertRaises(ValueError): load_prices(path)
            broken = original.reset_index(); broken.loc[1,"Date"] = broken.loc[0,"Date"]
            broken.to_csv(path,index=False)
            with self.assertRaises(ValueError): load_prices(path)

    def test_unsorted_dates_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"prices.csv"
            frame = generate_demo(path); frame.iloc[::-1].to_csv(path)
            with self.assertRaises(ValueError): load_prices(path)

    def test_cagr_matches_geometric_growth(self):
        r = pd.Series([.01, -.02, .03, .02])
        result = realised_metrics(r, periods=4)
        self.assertAlmostEqual(result["cagr"], (1+r).prod()-1)


if __name__ == "__main__":
    unittest.main()


class RobustnessTests(unittest.TestCase):
    def test_frontier_can_be_skipped_for_repeated_experiments(self):
        allocations, frontier = optimise_portfolios(np.array([.10, .05]), np.diag([.04, .01]), frontier_points=0)
        self.assertEqual(set(allocations), {"Equal weight", "Minimum volatility", "Maximum Sharpe"})
        self.assertTrue(frontier.empty)

    def test_frontier_point_count_is_configurable(self):
        _, frontier = optimise_portfolios(np.array([.10, .05]), np.diag([.04, .01]), frontier_points=7)
        self.assertEqual(len(frontier), 7)

    def test_invalid_frontier_point_count_rejected(self):
        with self.assertRaises(ValueError):
            optimise_portfolios(np.array([.10, .05]), np.diag([.04, .01]), frontier_points=-1)

    def test_demo_frame_is_reproducible(self):
        from portfolio_risk_lab.demo import generate_demo_frame
        a = generate_demo_frame(seed=31, n=120)
        b = generate_demo_frame(seed=31, n=120)
        pd.testing.assert_frame_equal(a, b)

    def test_demo_volatility_scale_changes_path(self):
        from portfolio_risk_lab.demo import generate_demo_frame
        a = generate_demo_frame(seed=31, n=120, volatility_scale=.75)
        b = generate_demo_frame(seed=31, n=120, volatility_scale=1.35)
        self.assertFalse(np.allclose(a.to_numpy(), b.to_numpy()))

    def test_robustness_output_shapes_and_winners(self):
        from portfolio_risk_lab.robustness import run_robustness, summarise_robustness
        results, weights = run_robustness(experiments=12, seed_start=500)
        self.assertEqual(len(results), 36)
        self.assertEqual(len(weights), 12 * 3 * 5)
        winners = results.groupby("experiment").holdout_sharpe_winner.sum()
        self.assertTrue((winners == 1).all())
        summary, stability = summarise_robustness(results, weights)
        self.assertEqual(set(summary.index), {"Equal weight", "Minimum volatility", "Maximum Sharpe"})
        self.assertEqual(len(stability), 15)

    def test_robustness_reproducible(self):
        from portfolio_risk_lab.robustness import run_robustness
        a, _ = run_robustness(experiments=10, seed_start=700)
        b, _ = run_robustness(experiments=10, seed_start=700)
        pd.testing.assert_frame_equal(a, b)

    def test_regime_rotation_is_balanced_to_one(self):
        from portfolio_risk_lab.robustness import run_robustness
        results, _ = run_robustness(experiments=12, seed_start=900)
        counts = results.drop_duplicates("experiment").regime.value_counts()
        self.assertLessEqual(counts.max() - counts.min(), 1)
