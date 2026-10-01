"""Console entry point for the assumption sensitivity grid."""
import argparse
from pathlib import Path
from .sensitivity import run_sensitivity, write_sensitivity


def main():
    parser = argparse.ArgumentParser(description="Sensitivity grid for train split and risk-free assumptions.")
    parser.add_argument("--output", type=Path, default=Path("outputs/sensitivity"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    frame = run_sensitivity(seed=args.seed)
    matrix = write_sensitivity(args.output, frame)
    print("SENSITIVITY ANALYSIS COMPLETE")
    print(matrix.round(3).to_string())
    print(f"\nSaved to {args.output}")
