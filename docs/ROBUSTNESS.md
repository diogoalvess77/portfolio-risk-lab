# Robustness layer

## Research purpose

The baseline report answers a narrow question on one synthetic path: how do three portfolios fitted on a training period behave on a later holdout?

A single path is not enough to say whether the observed ranking is stable. Version 2 therefore repeats the full estimation-and-evaluation pipeline across many independent synthetic markets.

## Experimental design

The committed robustness run uses 250 independent paths. Market seeds start at 1000 and increase by one for each experiment. The generator cycles deterministically through three illustrative regimes:

- Low volatility: base annual volatilities × 0.75
- Base: base annual volatilities × 1.00
- High volatility: base annual volatilities × 1.35 and annual drifts shifted down by 1 percentage point

Every path contains five fictional assets. Returns are split chronologically using the same default 80% training / 20% holdout rule as the baseline.

For each path:

1. Estimate annualised sample means and covariance from training returns only.
2. Fit equal-weight, minimum-volatility and maximum-Sharpe allocations.
3. Do **not** compute the 40-point efficient frontier, because it is unnecessary for the repeated experiment and would only add computation.
4. Freeze the selected weights.
5. Apply each allocation to the holdout using buy-and-hold execution.
6. Record training Sharpe, realised holdout Sharpe, return, volatility, maximum drawdown, VaR and expected shortfall.
7. Mark the strategy with the highest realised holdout Sharpe for that path.

## Why the repeated experiment matters

Optimisation can amplify sampling noise. A portfolio may receive a high estimated training Sharpe because the estimated mean vector happened to be favourable in that sample. Repeating the experiment exposes how much that apparent edge survives into independent holdout periods.

The train-vs-holdout scatter plot is therefore one of the most important outputs. Points far below the 45-degree line show cases where estimated Sharpe did not translate into realised Sharpe.

## Weight stability

The robustness layer also stores every asset weight selected in every experiment. The mean and standard deviation of the maximum-Sharpe weights provide a direct view of allocation instability.

This is useful because two optimisation procedures can have similar average performance while one produces far more unstable positions.

## Interpretation limits

A 250-path simulation is more informative than one path, but it is still conditional on the chosen synthetic generator. It does **not** prove that any strategy will outperform in real markets.

The purpose is methodological: demonstrate training/holdout discipline, repeated experimentation, uncertainty, reproducibility and careful interpretation.
