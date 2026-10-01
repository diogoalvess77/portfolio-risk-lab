"""Portfolio calculations, optimisation and holdout evaluation."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize


def load_prices(path: str | Path) -> pd.DataFrame:
    """Load a price CSV with a Date column and at least two assets."""
    # Each remaining column is an asset; dates become the index.
    frame = pd.read_csv(path)
    if "Date" not in frame or len(frame.columns) < 3:
        raise ValueError("CSV needs Date and at least two asset columns.")
    dates = pd.to_datetime(frame.pop("Date"), errors="raise")
    if dates.isna().any() or dates.duplicated().any():
        raise ValueError("Dates must be valid and unique.")
    if not dates.is_monotonic_increasing:
        raise ValueError("Dates must be strictly increasing; sort the CSV first.")
    frame = frame.apply(pd.to_numeric, errors="raise")
    frame.index = pd.DatetimeIndex(dates, name="Date")
    if len(frame) < 100:
        raise ValueError("Provide at least 100 price observations.")
    # Zero prices break percentage returns. Missing data also need an explicit
    # decision, so reject them rather than filling gaps automatically.
    if not np.isfinite(frame.to_numpy()).all() or (frame <= 0).any().any():
        raise ValueError("Prices must be finite, positive and complete. No automatic forward-filling.")
    return frame


def split_returns(prices: pd.DataFrame, train_fraction: float = 0.8):
    """Split returns in date order, keeping the return across the split boundary."""
    if not 0.5 <= train_fraction <= 0.9:
        raise ValueError("train_fraction must be between 0.5 and 0.9.")
    # Compute returns before splitting so we keep the first holdout return.
    # Only the very first price observation has no previous price.
    returns = prices.pct_change(fill_method=None).iloc[1:]
    cutoff = int(len(returns) * train_fraction)
    train, test = returns.iloc[:cutoff].copy(), returns.iloc[cutoff:].copy()
    if len(train) < 60 or len(test) < 20:
        raise ValueError("Need at least 60 training and 20 holdout returns.")
    return train, test


def estimate_moments(returns: pd.DataFrame, periods: int = 252):
    """Estimate annualised mean returns and sample covariance.

Pass training returns here, not the full dataset. Multiplying by the number
of periods per year uses the standard annualisation convention; it doesn't
adjust for serial correlation."""
    if periods < 1:
        raise ValueError("periods must be positive.")
    if not np.isfinite(returns.to_numpy()).all():
        raise ValueError("Returns must be finite.")
    mean = returns.mean().to_numpy() * periods
    covariance = returns.cov().to_numpy() * periods
    # Constant or duplicate assets can make the covariance matrix singular.
    # Require positive variance in every nonzero portfolio direction.
    if np.linalg.eigvalsh(covariance).min() <= 1e-12:
        raise ValueError("Covariance is singular or near-singular; remove constant/duplicate assets or add data.")
    return mean, covariance


def validate_weights(weights, n_assets: int):
    """Check that weights are finite, long-only and sum to one."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n_assets,) or not np.isfinite(weights).all():
        raise ValueError("One finite weight is required per asset.")
    if weights.min() < -1e-8 or not np.isclose(weights.sum(), 1, atol=1e-7):
        raise ValueError("Weights must be long-only and sum to one.")
    return weights


def portfolio_metrics(weights, mean, covariance, risk_free_rate=0.03):
    """Calculate estimated return, volatility and Sharpe for one allocation.

Return and volatility are decimals: 0.10 means 10%. Sharpe is a ratio."""
    weights = validate_weights(weights, len(mean))
    expected_return = float(weights @ mean)
    volatility = float(np.sqrt(max(0, weights @ covariance @ weights)))
    if volatility <= 1e-12:
        raise ValueError("Portfolio volatility is zero.")
    return {"annual_return_estimate": expected_return, "annual_volatility": volatility,
            "sharpe_estimate": (expected_return - risk_free_rate) / volatility}


def sample_portfolios(mean, covariance, count=20_000, seed=42, risk_free_rate=0.03):
    """Draw random long-only allocations and calculate their metrics.

Dirichlet draws sum to one and sample uniformly over the weight simplex.
A local random generator keeps these draws reproducible without affecting
random numbers elsewhere in the project."""
    if count < 100 or count > 1_000_000:
        raise ValueError("Portfolio count must be between 100 and 1,000,000.")
    rng = np.random.default_rng(seed)
    weights = rng.dirichlet(np.ones(len(mean)), size=count)
    expected = weights @ mean
    # Calculate the variance of every sampled allocation in one vectorised operation.
    volatility = np.sqrt(np.einsum("ij,jk,ik->i", weights, covariance, weights))
    metrics = pd.DataFrame({"annual_return_estimate": expected,
                            "annual_volatility": volatility,
                            "sharpe_estimate": (expected-risk_free_rate)/volatility})
    return weights, metrics


def optimise_portfolios(mean, covariance, risk_free_rate=0.03, frontier_points=40):
    """Find the comparison portfolios and trace the efficient frontier.

SLSQP handles the weight bounds and full-investment constraint. Minimising
variance also minimises volatility; minimising negative Sharpe maximises
Sharpe. We try several starting weights for the Sharpe optimisation."""
    n = len(mean)
    equal = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n  # Long-only, with no leverage.
    constraints = [{"type": "eq", "fun": lambda w: w.sum()-1}]
    def solve(objective, initial, extra=()):
        result = minimize(objective, initial, method="SLSQP", bounds=bounds,
                          constraints=constraints+list(extra),
                          options={"ftol": 1e-11, "maxiter": 2000})
        if not result.success:
            raise RuntimeError(f"Optimisation did not converge: {result.message}")
        validate_weights(result.x, n)
        # Clean up tiny rounding errors only after checking the solver result.
        # This must not hide a failed optimisation or invalid allocation.
        weights = np.clip(result.x, 0, 1)
        return weights/weights.sum()
    variance = lambda w: float(w @ covariance @ w)
    min_vol = solve(variance, equal)
    def negative_sharpe(w):
        return -(w @ mean-risk_free_rate)/np.sqrt(variance(w))
    candidates = []
    for initial in [equal, min_vol, *np.eye(n)]:
        try:
            candidates.append(solve(negative_sharpe, initial))
        except RuntimeError:
            continue
    if not candidates:
        raise RuntimeError("All maximum-Sharpe optimisation attempts failed.")
    max_sharpe = min(candidates, key=negative_sharpe)
    if frontier_points < 0 or frontier_points > 500:
        raise ValueError("frontier_points must be between 0 and 500.")
    points = []
    # Robustness runs only need the selected portfolios, so the frontier can be skipped.
    if frontier_points:
        targets = np.linspace(min_vol @ mean, mean.max(), frontier_points)
        for target in targets:
            # Capture this target in the lambda instead of referring to a changing loop value.
            extra = [{"type": "eq", "fun": lambda w, t=target: w @ mean-t}]
            weights = solve(variance, min_vol, extra)
            if abs(weights @ mean-target) > 1e-7:
                raise RuntimeError("Efficient-frontier target constraint was not satisfied.")
            points.append(portfolio_metrics(weights, mean, covariance, risk_free_rate))
    return {"Equal weight": equal, "Minimum volatility": min_vol,
            "Maximum Sharpe": max_sharpe}, pd.DataFrame(points)


def buy_and_hold(returns: pd.DataFrame, weights) -> pd.Series:
    """Apply the initial weights once, then let the positions drift.

Each asset compounds independently. Add up the positions to get portfolio
wealth, then convert that wealth back into daily portfolio returns."""
    weights = validate_weights(weights, returns.shape[1])
    nav = (1+returns).cumprod().to_numpy() @ weights
    changes = nav / np.r_[1.0, nav[:-1]]-1
    return pd.Series(changes, index=returns.index)


def drawdown(returns: pd.Series) -> pd.Series:
    """Calculate drawdowns from the running peak, including the initial investment."""
    nav = (1+returns).cumprod()
    # Include the initial wealth of 1 so a first-day loss counts as a drawdown.
    peak = np.maximum.accumulate(np.r_[1.0, nav.to_numpy()])[1:]
    return nav/peak-1


def realised_metrics(returns: pd.Series, risk_free_rate=0.03, periods=252):
    """Measure performance from realised portfolio returns.

CAGR uses the number of observations. VaR and expected shortfall describe
one-period losses, not annual losses. The tail calculation is documented
in docs/METHODOLOGY.md."""
    values = np.asarray(returns, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all() or (values <= -1).any():
        raise ValueError("Need at least two finite returns greater than -100%.")
    total = float(np.prod(1+values)-1)
    volatility = float(np.std(values, ddof=1)*np.sqrt(periods))
    # Convert the annual risk-free rate to a compounded daily rate.
    # The training estimate uses an annual arithmetic convention instead.
    daily_rf = (1+risk_free_rate)**(1/periods)-1
    sharpe = (values.mean()-daily_rf)*periods/volatility if volatility > 1e-12 else None
    losses = -values  # A return of -2% becomes a loss of +2%.
    var = float(np.quantile(losses, 0.95))
    es = float(losses[losses >= var].mean())
    return {"total_return": total, "cagr": (1+total)**(periods/len(values))-1,
            "annual_volatility": volatility, "sharpe": sharpe,
            "maximum_drawdown": float(drawdown(returns).min()),
            "daily_var_95": var, "daily_expected_shortfall_95": es}
