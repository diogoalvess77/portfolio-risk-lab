"""Run with `python run_robustness.py`."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from portfolio_risk_lab.robustness_cli import main
if __name__ == "__main__":
    main()
