# Code walkthrough

This document connects the implementation to the financial reasoning. The code's comments
are in English so the repository can be read by an international audience. For setup and publication steps, start with [PROJECT_GUIDE.md](PROJECT_GUIDE.md).

## Follow one complete run

1. **`run_demo.py` → `cli.main()`**: parse settings. Without `--prices`, generate fictional prices.
2. **`load_prices()`**: enforce the data contract before calculating anything.
3. **`split_returns()`**: compute simple returns, then separate training from evaluation.
4. **`estimate_moments()`**: estimate the vector of annual means and the covariance matrix on training only.
5. **`sample_portfolios()`**: generate the feasible allocation cloud; this is exploratory sampling.
6. **`optimise_portfolios()`**: compute weights with constrained numerical optimisation.
7. **`buy_and_hold()`**: execute each initial allocation on the holdout, with no subsequent rebalancing.
8. **`realised_metrics()`**: evaluate what happened on that holdout path.
9. **`write_report()`**: export tables, metadata, charts and the standalone HTML report.

The order matters. Passing the full return history into step 4 would let the evaluation period
influence the portfolio decision. The test `test_changing_holdout_cannot_change_estimated_moments`
checks that changing the holdout prices leaves the training estimates unchanged.

## Shapes and units

| Variable | Shape in the demo | Meaning / units |
|---|---|---|
| `prices` | 1,261 × 5 | Price level; fictional common denomination |
| `train` | 1,008 × 5 | Simple daily returns, decimals |
| `holdout` | 252 × 5 | Evaluation-period returns, decimals |
| `mean` | 5 | Annualised arithmetic mean returns |
| `covariance` | 5 × 5 | Annualised return covariance |
| sampled `weights` | 20,000 × 5 | One fully invested allocation per row |
| one strategy's weights | 5 | Initial allocation fractions |
| `performance` | 252 × 3 | Executed daily returns, one column per strategy |

Percentages are stored as decimals: 0.05 means 5%. Weight 0.20 means 20% of capital.
Sharpe is dimensionless. Chart percentages are multiplied by 100 only for display.

## Read the key expressions

### Simple returns

```python
returns = prices.pct_change(fill_method=None).iloc[1:]
```

For prices 100 and 105, the second observation is `105 / 100 - 1 = 0.05`.
The initial price has no preceding price, so only its return row is removed.

### Portfolio return and covariance

```python
expected_return = weights @ mean
variance = weights @ covariance @ weights
```

The first expression averages estimated asset returns using capital weights. The second
includes both individual variances and cross-asset covariances. Ignoring off-diagonal
covariances would incorrectly assume every pair of assets is uncorrelated.

### Why a Dirichlet distribution?

```python
weights = rng.dirichlet(np.ones(len(mean)), size=count)
```

It generates non-negative rows that sum to one. Parameters all equal to one make the
distribution uniform over the feasible simplex. The generator seed makes the sample repeatable.

### Why the negative Sharpe objective?

SciPy's minimiser searches for a minimum. Multiplying Sharpe by −1 lets that same routine find
a maximum. Weight bounds and the sum-to-one equality remain active throughout the search.

### Why compounding matters

```python
nav = (1 + returns).cumprod().to_numpy() @ weights
changes = nav / np.r_[1.0, nav[:-1]] - 1
```

`cumprod()` compounds the asset returns. Multiplying each cumulative asset growth by its
initial weight yields total wealth. The inserted 1.0 is starting capital, so day-one returns
are retained. Applying fixed weights to each day's returns instead would imply daily rebalancing.

### Why start the drawdown peak at 1?

If wealth falls from 1 to 0.9 on the first day, drawdown is already −10%. Starting the running
peak at 0.9 would incorrectly report no loss on that day. The implementation and test handle this.

## Understanding the tests

The suite checks independently known answers where possible. For two uncorrelated assets with
variances 0.04 and 0.01, the minimum-variance weight on asset 1 is `0.01 / (0.04 + 0.01) = 0.20`.
That analytical result is compared with the numerical optimiser. A dense allocation grid also
checks the two-asset maximum-Sharpe result. Other tests exercise timing, drift and invalid input.

Run all 22 tests from the repository root after installation:

```bash
python -m unittest discover -s tests -v
```

## Reproduce and inspect the results

```bash
python run_demo.py --output outputs/review
```

Check `allocations.csv`: every strategy's weights must add to one. Compare `holdout_metrics.csv`
with the HTML table. Open `run_metadata.json` to confirm the data hash, seed, sample boundaries
and software versions. See [METHODOLOGY.md](METHODOLOGY.md) for every metric's convention.

## Safe learning experiments

- Change `--seed` while leaving all else unchanged: the random cloud changes, but deterministic
  optimised weights and holdout strategy returns should not. Optimisation does not use the cloud.
- Change `--risk-free-rate`: maximum-Sharpe selection can change, while minimum-volatility selection
  does not use the rate. All reported Sharpe values use the new assumption.
- Change `--train-fraction`: both estimation and evaluation periods change. Do not choose the split
  that gives the best reported result and then call it an untouched out-of-sample test.

These experiments are for understanding sensitivity. Keep the committed baseline identifiable
and describe any change when presenting a new result.

## Version 2: repeated experiments

`robustness.py` generates independent synthetic paths, fits the three strategies on each training sample and evaluates frozen weights on each later holdout. `frontier_points=0` deliberately skips the efficient-frontier trace during repeated experiments, reducing runtime without changing the selected equal-weight, minimum-volatility or maximum-Sharpe portfolios.

`run_robustness.py --experiments 250` writes raw experiment data, all selected weights, aggregate summary statistics, four charts and `robustness_report.html`.

`sensitivity.py` and `run_sensitivity.py` provide a small assumption grid for the train split and risk-free-rate input.
