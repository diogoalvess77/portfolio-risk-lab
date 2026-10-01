# Portfolio Risk Lab v2: research report

## Executive overview

Portfolio Risk Lab studies a common portfolio-construction problem: an allocation that looks strong on the sample used to estimate it may not retain that advantage on later observations.

The project implements a reproducible Python pipeline combining Monte Carlo allocation sampling, constrained portfolio optimisation, chronological holdout evaluation, repeated synthetic-market stress testing, sensitivity analysis and automated numerical validation.

The baseline uses five fictional asset series, 20,000 sampled allocations, 40 efficient-frontier points and three allocation rules. Version 2 adds a robustness layer that repeats the full fit-then-holdout process across 250 independent synthetic market paths and three volatility regimes. Twenty-two automated tests cover numerical behaviour, data integrity, chronology and repeated-experiment reproducibility.

**All committed results use synthetic data.** They are educational demonstrations, not historical investment performance, forecasts or investment advice.

## Research question

The central question is:

> Does selecting a portfolio for a high estimated training-period Sharpe ratio lead to stronger risk-adjusted performance on subsequently unseen observations?

A single holdout can illustrate the problem but cannot establish robustness. The v2 design therefore asks a second question:

> How stable is the training-to-holdout relationship when the entire experiment is repeated across many independent synthetic market paths?

## Baseline experimental design

Prices are converted to simple returns and split chronologically. By default, 80% of returns are used for training and 20% for holdout evaluation. Training means and covariances determine the portfolio allocations. Holdout returns are not used to select weights.

The three allocation rules are:

1. **Equal weight** - 20% per fictional asset at inception.
2. **Minimum volatility** - constrained variance minimisation under long-only, fully invested weights.
3. **Maximum Sharpe** - constrained maximisation of estimated excess return per unit of volatility.

The baseline also draws 20,000 long-only allocations from a Dirichlet distribution to visualise the feasible risk-return cloud. A 40-point numerical efficient frontier is solved separately using SLSQP.

## Holdout execution and metrics

Selected weights are frozen before the holdout. The evaluation assumes one initial allocation followed by buy-and-hold weight drift. There is no daily rebalancing, leverage or short selling.

Seven realised metrics are reported:

- Total compounded return
- CAGR
- Annualised sample volatility
- Realised Sharpe ratio
- Maximum drawdown
- Empirical daily 95% Value at Risk
- Empirical daily 95% expected shortfall

No fees, taxes, transaction costs or turnover penalties are included.

## Baseline result

The committed baseline contains 1,261 synthetic price observations, producing 1,008 training returns and 252 holdout returns.

| Holdout measure | Equal weight | Minimum volatility | Maximum Sharpe |
|---|---:|---:|---:|
| Total return | 16.67% | 14.48% | 14.47% |
| Annualised volatility | 10.40% | 4.93% | 10.60% |
| Sharpe | 1.25 | 2.17 | 1.05 |
| Maximum drawdown | -6.48% | -1.89% | -5.27% |
| Daily VaR, 95% | 0.97% | 0.50% | 1.08% |
| Daily expected shortfall, 95% | 1.22% | 0.59% | 1.25% |

The maximum-Sharpe portfolio was chosen from training information only, yet did not produce the strongest realised holdout Sharpe on this path. That is an illustration of estimation error, not a general investment conclusion.

## Repeated robustness experiment

Version 2 repeats the entire training -> optimisation -> holdout sequence across 250 independent synthetic markets.

The market generator cycles through three transparent regimes:

- Low volatility: base volatilities multiplied by 0.75
- Base: original demonstration parameters
- High volatility: base volatilities multiplied by 1.35 with annual drifts shifted down by 1 percentage point

The efficient-frontier trace is intentionally skipped inside repeated runs because it is not needed to fit the three selected allocations. This keeps the experiment computationally efficient while preserving the strategy-selection logic.

For every path and strategy, the project stores training Sharpe, holdout Sharpe, total return, realised volatility, maximum drawdown, VaR, expected shortfall and all asset weights.

## Robustness results

Across the committed 250-path experiment:

| Strategy | Mean holdout Sharpe | Mean training Sharpe | Mean holdout-training gap | Holdout Sharpe win rate |
|---|---:|---:|---:|---:|
| Equal weight | 0.35 | 0.28 | +0.07 | 38.8% |
| Minimum volatility | 0.10 | 0.09 | +0.01 | 33.2% |
| Maximum Sharpe | 0.30 | 0.75 | -0.45 | 28.0% |

Within this synthetic design, maximum Sharpe has the highest average training estimate but a substantially lower realised average holdout Sharpe. The train-to-holdout gap is therefore a direct visual and numerical example of optimisation sensitivity to noisy estimates.

These win rates and averages are descriptive of the simulation design only. They should not be interpreted as forecasts or claims that one strategy is generally superior in real markets.

## Weight stability

Every selected weight is retained across experiments. The project reports the mean and standard deviation of each asset weight, with particular attention to the maximum-Sharpe strategy.

Weight instability is important because an optimiser can react strongly to small sample differences even when headline performance statistics appear similar. The v2 report therefore treats allocation stability as a research output rather than focusing only on returns.

## Sensitivity analysis

A separate sensitivity grid re-runs the same reproducible synthetic path using:

- Train/holdout splits of 70/30, 80/20 and 90/10
- Annual risk-free assumptions of 0%, 3% and 5%

The resulting holdout Sharpe matrix is exported to CSV. This does not replace the repeated robustness experiment; it checks whether the baseline interpretation changes materially under simple modelling conventions.

## Validation and reproducibility

The project contains 22 automated tests. They cover:

- analytical two-asset minimum-variance behaviour;
- maximum Sharpe against a dense numerical grid;
- feasible and reproducible sampled weights;
- buy-and-hold weight drift;
- drawdown and tail-loss metric behaviour;
- invalid prices, dates and weights;
- chronological train/holdout boundaries;
- isolation of training estimates from holdout changes;
- configurable efficient-frontier size;
- deterministic synthetic path generation;
- robustness output structure and winner logic;
- repeated robustness reproducibility;
- balanced regime rotation.

GitHub Actions runs the tests plus baseline, robustness and sensitivity smoke runs on pushes and pull requests.

Run metadata include random seeds, data hashes, sample boundaries, assumptions and software versions.

## Main outputs

The baseline creates:

- `frontier.png`
- `holdout.png`
- `drawdowns.png`
- `allocations.png`
- `correlations.png`
- auditable CSVs
- `report.html`

The robustness layer adds:

- `robustness_holdout_sharpe.png`
- `robustness_train_vs_holdout.png`
- `robustness_win_rates.png`
- `robustness_weight_stability.png`
- raw experiment and weight CSVs
- `robustness_report.html`

The sensitivity layer exports raw results and a Sharpe matrix.

## Limitations

The synthetic generator is deliberately simple and cannot reproduce every property of real financial markets. It omits many features including jumps, fat tails, changing correlations, liquidity constraints and structural breaks.

Repeated simulation improves robustness **within this model**, not external validity to real markets. Expected-return estimates remain noisy, the asset universe is fixed, and no transaction costs, taxes, leverage, short selling, FX effects or turnover constraints are modelled.

Real-data use requires careful price adjustment, provenance, currency consistency and universe selection. VaR and expected shortfall are sample statistics rather than guaranteed loss limits.

## Project disclosure

Implementation and documentation were produced with AI assistance. Source code, assumptions, tests, raw outputs and reports are included so the work can be inspected and reproduced.
