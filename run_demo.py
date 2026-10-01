"""Run the demo with `python run_demo.py`.

This wrapper adds the local src folder to Python's import path, so you can run
it without installing the package itself. The dependencies are still required.
Output paths are relative to the directory where you run the command."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from portfolio_risk_lab.cli import main

if __name__ == "__main__":
    main()
