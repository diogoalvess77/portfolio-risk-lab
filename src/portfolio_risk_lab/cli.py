"""Run the analysis from the command line, fitting weights on training data only."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd
import scipy
import matplotlib
from .core import (load_prices, split_returns, estimate_moments, sample_portfolios,
                   optimise_portfolios, buy_and_hold, realised_metrics)
from .demo import generate_demo
from .reporting import write_report


def main():
    """Load data, fit allocations, evaluate the holdout and save the results."""
    parser = argparse.ArgumentParser(description="Portfolio research with training-only optimisation and buy-and-hold holdout.")
    parser.add_argument("--prices", type=Path, help="CSV: Date plus >=2 positive adjusted-price series in one currency.")
    parser.add_argument("--output", type=Path, default=Path("outputs/demo"))
    parser.add_argument("--portfolios", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=42, help="Random portfolio-weight sampling seed.")
    parser.add_argument("--train-fraction", type=float, default=.8)
    parser.add_argument("--risk-free-rate", type=float, default=.03, help="Annual decimal, e.g. 0.03 (assumption, not a live rate).")
    parser.add_argument("--periods-per-year", type=int, default=252)
    args = parser.parse_args()
    if not np.isfinite(args.risk_free_rate) or args.risk_free_rate <= -1:
        parser.error("risk-free-rate must be finite and greater than -1.")
    if args.seed < 0:
        parser.error("seed must be non-negative.")
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        synthetic = args.prices is None
        path = args.prices if args.prices else args.output/"demo_prices.csv"
        if synthetic:
            generate_demo(path, seed=7)
        # Validate the input and split it in chronological order.
        prices = load_prices(path)
        train, holdout = split_returns(prices, args.train_fraction)
        # Fit every allocation using training returns only.
        mean, covariance = estimate_moments(train, args.periods_per_year)
        weights, cloud = sample_portfolios(mean, covariance, args.portfolios, args.seed, args.risk_free_rate)
        allocations, frontier = optimise_portfolios(mean, covariance, args.risk_free_rate)
        # Evaluate those weights on the holdout without fitting them again.
        performance = pd.DataFrame({name: buy_and_hold(holdout, w) for name, w in allocations.items()})
        summary = pd.DataFrame({name: realised_metrics(performance[name], args.risk_free_rate,
                                  args.periods_per_year) for name in performance}).T
        # Save the input hash, assumptions and library versions for reproducibility.
        # The hash identifies the actual data, not just the filename.
        metadata = {"synthetic_demo": synthetic, "data_file": path.name,
                    "data_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "asset_count": len(prices.columns), "assets": list(prices.columns),
                    "price_observations": len(prices), "sampled_portfolios": args.portfolios,
                    "sampling_seed": args.seed, "demo_generator_seed": 7 if synthetic else None,
                    "train_fraction": args.train_fraction, "periods_per_year": args.periods_per_year,
                    "risk_free_rate": args.risk_free_rate,
                    "train_start": str(train.index[0].date()), "train_end": str(train.index[-1].date()),
                    "holdout_start": str(holdout.index[0].date()), "holdout_end": str(holdout.index[-1].date()),
                    "training_returns": len(train), "holdout_returns": len(holdout),
                    "execution": "initial allocation, then buy-and-hold; no costs",
                    "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                 "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__}}
        # Keep the synthetic label if a supplied CSV has matching source metadata.
        if not synthetic and path.with_suffix(".metadata.json").exists():
            source = json.loads(path.with_suffix(".metadata.json").read_text())
            metadata["source_metadata"] = source
            metadata["synthetic_demo"] = bool(source.get("synthetic", False))
        write_report(args.output, train, holdout, allocations, cloud, frontier, mean, covariance,
                     performance, summary, metadata)
        cloud.join(pd.DataFrame(weights, columns=prices.columns)).to_csv(args.output/"sampled_portfolios.csv", index=False)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(2, f"Input or analysis error: {exc}\n")
    print("SYNTHETIC DEMONSTRATION" if metadata["synthetic_demo"] else "USER-SUPPLIED DATA")
    print(f"{len(prices.columns)} assets | {len(train)} training returns | {len(holdout)} holdout returns")
    print(summary.round(4).to_string())
    print(f"\nOpen {args.output / 'report.html'} in your browser.")


if __name__ == "__main__":
    main()
