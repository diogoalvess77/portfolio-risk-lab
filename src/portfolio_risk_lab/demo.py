"""Synthetic price generation used for reproducible demonstrations and stress tests."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ASSET_NAMES = ["Demo_Equity_A", "Demo_Equity_B", "Demo_Bonds", "Demo_Gold", "Demo_REITs"]
BASE_ANNUAL_DRIFT = np.array([.09, .075, .03, .045, .065])
BASE_ANNUAL_VOL = np.array([.19, .22, .055, .14, .20])


def _base_correlation():
    """Return the fixed factor-implied correlation matrix used by the demo."""
    factor = np.array([[.8, .1], [.7, .2], [-.15, .5], [.05, -.1], [.65, .35]])
    correlation = factor @ factor.T
    correlation += np.diag(1 - np.diag(correlation))
    return correlation


def generate_demo_frame(seed=7, n=1260, volatility_scale=1.0, drift_shift=0.0,
                        start="2020-01-02"):
    """Generate a synthetic five-asset price panel in memory.

    The parameters are deliberately simple and transparent. `volatility_scale`
    and `drift_shift` make it possible to create low/base/high-volatility stress
    regimes without pretending that these paths are historical market data.
    """
    if n < 100:
        raise ValueError("n must be at least 100 price-return steps.")
    if not np.isfinite(volatility_scale) or volatility_scale <= 0:
        raise ValueError("volatility_scale must be positive and finite.")
    if not np.isfinite(drift_shift):
        raise ValueError("drift_shift must be finite.")

    rng = np.random.default_rng(seed)
    annual_drift = BASE_ANNUAL_DRIFT + drift_shift
    annual_vol = BASE_ANNUAL_VOL * volatility_scale
    correlation = _base_correlation()
    covariance = np.outer(annual_vol, annual_vol) * correlation / 252
    log_mean = (annual_drift - .5 * annual_vol**2) / 252
    logs = rng.multivariate_normal(log_mean, covariance, n)
    prices = 100 * np.exp(np.vstack([np.zeros(len(ASSET_NAMES)), np.cumsum(logs, axis=0)]))
    dates = pd.bdate_range(start, periods=n + 1)
    frame = pd.DataFrame(prices, columns=ASSET_NAMES, index=dates)
    frame.index.name = "Date"
    return frame


def generate_demo(path: str | Path, seed=7, n=1260, volatility_scale=1.0, drift_shift=0.0):
    """Save one synthetic demonstration path and provenance metadata."""
    frame = generate_demo_frame(seed=seed, n=n, volatility_scale=volatility_scale,
                                drift_shift=drift_shift)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, float_format="%.8f")
    path.with_suffix(".metadata.json").write_text(json.dumps({
        "synthetic": True,
        "source": "Local geometric Brownian demonstration generator",
        "seed": seed,
        "currency": "Fictional common denomination",
        "volatility_scale": volatility_scale,
        "drift_shift": drift_shift,
        "notes": "Generic weekday dates; not actual market prices."
    }, indent=2))
    return frame
