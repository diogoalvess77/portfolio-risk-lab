# Methodology and implementation choices

## 1. Prices and returns

For each asset i, compute simple period returns:

`r[i,t] = P[i,t] / P[i,t-1] - 1`

No price gaps are filled. No logarithmic returns are used in the portfolio analysis.
The fictional-price generator uses log returns solely to produce positive prices.

Returns are split once, in chronological order, at `floor(N × train_fraction)`.
The first holdout return uses the last training-date price as its denominator. It is not discarded.
Neither holdout returns nor their moments enter the allocation selection routines.

For daily data with K=252, training mean and covariance estimates are:

`mu = K × mean(daily returns)`

`Sigma = K × sample_covariance(daily returns)`

Covariance uses a sample denominator of N−1. Singular/near-singular covariance is rejected with a
descriptive error. There is no hidden regularisation or covariance repair.

## 2. Portfolio estimates

With a fully invested, long-only weight vector w:

`sum(w) = 1` and `0 <= w[i] <= 1`

`expected_arithmetic_return = w.T × mu`

`volatility = sqrt(w.T × Sigma × w)`

`estimated_Sharpe = (expected_arithmetic_return - annual_rf) / volatility`

The training return estimate is arithmetic and annualised. It is **not** a CAGR.
The risk-free input is a fixed annual scalar assumed for the experiment.

## 3. Monte Carlo allocation sampling

Each random weight vector comes from `Dirichlet(1, ..., 1)`, uniform over the simplex.
The default 20,000 draws use a seeded random generator. Every draw obeys the same constraints.
The cloud illustrates feasible allocations; it does not itself compute the exact frontier.

This is not a simulation of future portfolio paths. It does not provide confidence intervals for future returns.

## 4. Numerical optimisation

- Minimum volatility: minimise `w.T × Sigma × w` under the weight constraints.
- Maximum Sharpe: minimise negative estimated Sharpe, using equal weight, minimum volatility and each
  single-asset allocation as starting points. Keep the best successful feasible solution.
- Efficient frontier: choose 40 target returns from the global minimum-variance portfolio's return to
  the largest individual asset mean. Minimise variance at each target under a return equality constraint.

SLSQP convergence, weight constraints and frontier return targets are checked explicitly.
The plotted line is the efficient branch only. Solutions may be concentrated because no allocation cap exists.
If all estimated excess returns are negative, Sharpe optimisation is especially unintuitive and numerical
solutions should be inspected; this code does not claim a universal global optimum.

## 5. Holdout execution: buy and hold

All strategies start with wealth 1 at the final training date. Weights are initial allocations only:

`asset_growth[i,t] = product(1 + holdout_return[i,s], s=1..t)`

`portfolio_wealth[t] = sum(initial_w[i] × asset_growth[i,t])`

`portfolio_return[t] = portfolio_wealth[t]/portfolio_wealth[t-1] - 1`

This allows allocations to drift. It is not daily rebalancing. The equal-weight comparator is
equal-weighted only at inception, using the same execution rule as the optimised portfolios.
The training constant-weight estimates and realised buy-and-hold results therefore represent different
objects. Fees, bid-ask spreads, taxes and market impact are omitted for all strategies.

## 6. Realised evaluation metrics

- **Total return:** `final_wealth - 1`.
- **CAGR:** `final_wealth^(K / number_of_returns) - 1`. Observation-count convention, not calendar-day CAGR.
- **Annualised volatility:** sample standard deviation of portfolio period returns × `sqrt(K)`.
- **Sharpe:** annualised mean period excess return / annualised volatility. Here the annual RF is converted
  to a period rate as `(1 + annual_rf)^(1/K) - 1` before subtraction. This compounding convention means
  realised and training Sharpe calculations are not perfectly identical even for the same return sample.
- **Drawdown:** `wealth / running_peak - 1`, with starting wealth 1 included in the running peak.
- **Maximum drawdown:** the most negative drawdown. Reported as a negative number.
- **Daily 95% VaR:** the NumPy linear-interpolated 95th percentile of period losses `-return`.
- **Daily 95% expected shortfall:** mean of losses greater than or equal to the estimated VaR.
  Ties and finite-sample interpolation mean the tail count need not be exactly 5% of observations.

VaR/ES have a loss sign convention: positive indicates a loss; negative can occur when all tail observations
are gains. VaR is not a guaranteed bound. Tail estimates from 252 observations rely on about 13 extreme days.
Annualising returns or volatility assumes a sampling frequency; the program cannot infer economic frequency
from a CSV. A zero realised volatility yields an unavailable Sharpe, not an infinite value.

## 7. Demonstration data and integrity

Five fictional assets are generated using correlated Gaussian log returns. A factor-loading construction
gives a positive-definite correlation matrix. The demonstration has 1,260 return observations and one initial
price row. Drift and volatility parameters are illustrative, not empirical estimates for real instruments.

Keep the synthetic label on screenshots and descriptions of the included result. No performance claim about
a live portfolio is justified. To run an empirical study, use authorised adjusted prices, document provenance,
predefine the asset universe and split, and evaluate results without tuning against the holdout.

## 8. Validation scope

The 22 tests cover known two-asset solutions, a dense-grid comparison for Sharpe, feasible random weights,
buy-and-hold drift, first-day drawdowns, VaR/ES signs, geometric growth, invalid prices/weights, chronological
boundaries and isolation of training moments from holdout changes. They validate these properties; they
do not prove every possible financial assumption or dataset is appropriate.

## 9. Repeated robustness experiment (v2)

Version 2 repeats the complete fit-then-holdout process across 250 independent synthetic market paths. The experiment cycles through three transparent synthetic volatility regimes and records both training and holdout Sharpe for every strategy. The efficient frontier is skipped inside repeated runs because it is a visualisation of the training opportunity set, not required to fit the three selected allocations.

The robustness analysis reports distributions and win rates rather than using one path to infer general superiority. These results remain conditional on the synthetic generator and should not be interpreted as real-market forecasts.

## 10. Assumption sensitivity (v2)

A separate grid varies the train/holdout split (70/30, 80/20, 90/10) and annual risk-free input (0%, 3%, 5%) on the same reproducible path. This is a compact stress test for whether conclusions depend strongly on a single modelling convention.
