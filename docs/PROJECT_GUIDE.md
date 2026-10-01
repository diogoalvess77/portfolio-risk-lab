# Project guide: run, publish and explain

## Start with the committed outputs

Open `examples/demo/report.html` for the baseline and `examples/robustness/robustness_report.html` for the repeated-market experiment. All included data are synthetic.

Read these files in order:

1. `README.md` - project overview and headline results
2. `docs/PROJECT_REPORT.md` - research narrative
3. `docs/METHODOLOGY.md` - modelling conventions
4. `docs/ROBUSTNESS.md` - repeated experiment design
5. `docs/CODE_WALKTHROUGH.md` - code path
6. `docs/MAC_QUICKSTART.md` - exact macOS commands

## Run on macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
python run_demo.py
python run_robustness.py --experiments 250
python run_sensitivity.py
python -m pytest -q
```

Open the HTML reports:

```bash
open outputs/demo/report.html
open outputs/robustness/robustness_report.html
```

Expected test result for this release: `22 passed`.

## Publish on GitHub

Recommended repository name: `portfolio-risk-lab`

Recommended description:

> Portfolio optimisation, out-of-sample risk analysis and repeated robustness testing in Python.

Upload the **project contents**, not a ZIP file inside the repository. Include `src`, `tests`, `docs`, `examples`, `.github`, `README.md`, `pyproject.toml` and the run scripts.

After publication, add these topics:

`python`, `finance`, `quantitative-finance`, `portfolio-optimization`, `risk-management`, `monte-carlo`, `efficient-frontier`, `data-analysis`, `scipy`, `pandas`

Pin the repository on the GitHub profile if it is one of the projects you want recruiters to see first.

## What to understand before an interview

Be able to explain:

- why the holdout must not influence portfolio selection;
- why 20,000 random allocations are a visual search cloud rather than the optimiser itself;
- why SLSQP is used for the constrained optimisation;
- why maximum training Sharpe can deteriorate out of sample;
- why repeated paths are more informative than one simulated holdout;
- why the robustness result still does not prove real-market superiority;
- why maximum-Sharpe weights can be unstable;
- the difference between VaR and expected shortfall;
- the difference between a training estimate and a realised holdout metric.

## Verified project metrics

- Five fictional asset series
- 1,261 prices in the baseline
- 1,008 training returns and 252 holdout returns
- 20,000 sampled allocations
- 40 efficient-frontier points
- Three portfolio strategies
- Seven realised risk/performance metrics
- Five baseline charts
- 250 independent robustness paths
- Three synthetic volatility regimes
- Four robustness charts
- Nine sensitivity configurations
- 22 automated tests

## Practical extensions

Possible next steps include authorised real adjusted-price data, allocation caps, transaction-cost modelling, rolling walk-forward evaluation, covariance shrinkage and a final untouched test sample.

Those are potential extensions, not features already claimed by this release.
