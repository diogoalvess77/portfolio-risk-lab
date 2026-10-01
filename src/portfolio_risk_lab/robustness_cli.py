"""Console entry point for the repeated robustness experiment."""
import argparse
from pathlib import Path
from .robustness import run_and_write_robustness


def main():
    parser = argparse.ArgumentParser(description="Repeated synthetic-market out-of-sample robustness experiment.")
    parser.add_argument("--experiments", type=int, default=250)
    parser.add_argument("--output", type=Path, default=Path("outputs/robustness"))
    parser.add_argument("--train-fraction", type=float, default=.8)
    parser.add_argument("--risk-free-rate", type=float, default=.03)
    parser.add_argument("--seed-start", type=int, default=1000)
    args = parser.parse_args()
    _, _, summary, _ = run_and_write_robustness(args.output, args.experiments,
                                                 args.train_fraction, args.risk_free_rate,
                                                 seed_start=args.seed_start)
    print("ROBUSTNESS EXPERIMENT COMPLETE")
    print(f"{args.experiments} independent synthetic markets")
    print(summary.round(4).to_string())
    print(f"\nOpen {args.output / 'robustness_report.html'} in your browser.")
