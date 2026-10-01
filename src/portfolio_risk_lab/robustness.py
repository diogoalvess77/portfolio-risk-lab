"""Repeated synthetic-market experiments for out-of-sample robustness analysis."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .demo import generate_demo_frame
from .core import split_returns, estimate_moments, optimise_portfolios, portfolio_metrics, buy_and_hold, realised_metrics

STRATEGIES = ["Equal weight", "Minimum volatility", "Maximum Sharpe"]
REGIMES = {
    "Low volatility": {"volatility_scale": 0.75, "drift_shift": 0.00},
    "Base": {"volatility_scale": 1.00, "drift_shift": 0.00},
    "High volatility": {"volatility_scale": 1.35, "drift_shift": -0.01},
}


def run_robustness(experiments=250, train_fraction=.8, risk_free_rate=.03,
                   periods=252, seed_start=1000):
    """Repeat fit-then-holdout evaluation across independent synthetic paths.

    Regimes rotate deterministically so the experiment includes low, base and
    high-volatility environments. Every path is fitted only on its own training
    sample; its holdout remains untouched until evaluation.
    """
    if experiments < 10 or experiments > 10_000:
        raise ValueError("experiments must be between 10 and 10,000.")
    rows, weight_rows = [], []
    regime_names = list(REGIMES)

    for i in range(experiments):
        regime = regime_names[i % len(regime_names)]
        params = REGIMES[regime]
        market_seed = seed_start + i
        prices = generate_demo_frame(seed=market_seed, volatility_scale=params["volatility_scale"],
                                     drift_shift=params["drift_shift"])
        train, holdout = split_returns(prices, train_fraction)
        mean, covariance = estimate_moments(train, periods)
        allocations, _ = optimise_portfolios(mean, covariance, risk_free_rate, frontier_points=0)

        holdout_results = {}
        for name, weights in allocations.items():
            fitted = portfolio_metrics(weights, mean, covariance, risk_free_rate)
            realised = realised_metrics(buy_and_hold(holdout, weights), risk_free_rate, periods)
            holdout_results[name] = realised
            rows.append({
                "experiment": i + 1,
                "market_seed": market_seed,
                "regime": regime,
                "strategy": name,
                "training_sharpe": fitted["sharpe_estimate"],
                "holdout_sharpe": realised["sharpe"],
                "sharpe_gap": realised["sharpe"] - fitted["sharpe_estimate"],
                "total_return": realised["total_return"],
                "annual_volatility": realised["annual_volatility"],
                "maximum_drawdown": realised["maximum_drawdown"],
                "daily_var_95": realised["daily_var_95"],
                "daily_expected_shortfall_95": realised["daily_expected_shortfall_95"],
            })
            for asset, weight in zip(prices.columns, weights):
                weight_rows.append({"experiment": i + 1, "regime": regime,
                                    "strategy": name, "asset": asset, "weight": weight})

        valid = {k: v["sharpe"] for k, v in holdout_results.items() if v["sharpe"] is not None}
        winner = max(valid, key=valid.get)
        for row in rows[-len(STRATEGIES):]:
            row["holdout_sharpe_winner"] = row["strategy"] == winner

    return pd.DataFrame(rows), pd.DataFrame(weight_rows)


def summarise_robustness(results: pd.DataFrame, weights: pd.DataFrame):
    """Aggregate repeated experiments into recruiter-readable robustness statistics."""
    grouped = results.groupby("strategy", sort=False)
    summary = grouped.agg(
        experiments=("experiment", "count"),
        mean_holdout_sharpe=("holdout_sharpe", "mean"),
        median_holdout_sharpe=("holdout_sharpe", "median"),
        p05_holdout_sharpe=("holdout_sharpe", lambda x: x.quantile(.05)),
        p95_holdout_sharpe=("holdout_sharpe", lambda x: x.quantile(.95)),
        mean_training_sharpe=("training_sharpe", "mean"),
        mean_sharpe_gap=("sharpe_gap", "mean"),
        mean_total_return=("total_return", "mean"),
        mean_annual_volatility=("annual_volatility", "mean"),
        mean_maximum_drawdown=("maximum_drawdown", "mean"),
        holdout_win_rate=("holdout_sharpe_winner", "mean"),
    )
    stability = weights.groupby(["strategy", "asset"], sort=False).weight.agg(["mean", "std"]).reset_index()
    return summary, stability


def _plot_results(output: Path, results: pd.DataFrame, summary: pd.DataFrame,
                  stability: pd.DataFrame):
    """Create four compact robustness charts."""
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white"})

    ordered = STRATEGIES
    data = [results.loc[results.strategy == s, "holdout_sharpe"].dropna().to_numpy() for s in ordered]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(data, tick_labels=ordered, showmeans=True)
    ax.axhline(0, lw=1, alpha=.35)
    ax.set(title="Holdout Sharpe distribution across independent synthetic markets",
           ylabel="Realised holdout Sharpe")
    ax.grid(axis="y", alpha=.15)
    fig.tight_layout(); fig.savefig(output/"robustness_holdout_sharpe.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for strategy in ordered:
        part = results[results.strategy == strategy]
        ax.scatter(part.training_sharpe, part.holdout_sharpe, s=13, alpha=.45, label=strategy)
    lo = min(results.training_sharpe.min(), results.holdout_sharpe.min())
    hi = max(results.training_sharpe.max(), results.holdout_sharpe.max())
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, label="Training = holdout")
    ax.set(title="Training estimate vs realised holdout Sharpe", xlabel="Training Sharpe estimate",
           ylabel="Holdout Sharpe")
    ax.legend(fontsize=8); ax.grid(alpha=.15)
    fig.tight_layout(); fig.savefig(output/"robustness_train_vs_holdout.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    rates = summary.loc[ordered, "holdout_win_rate"] * 100
    ax.bar(ordered, rates)
    ax.set(title="Which strategy has the highest holdout Sharpe?", ylabel="Win rate (%)", ylim=(0, 100))
    for i, value in enumerate(rates):
        ax.text(i, value + 2, f"{value:.1f}%", ha="center")
    ax.grid(axis="y", alpha=.15)
    fig.tight_layout(); fig.savefig(output/"robustness_win_rates.png", dpi=170); plt.close(fig)

    max_sharpe = stability[stability.strategy == "Maximum Sharpe"].copy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(max_sharpe.asset, max_sharpe["mean"] * 100, yerr=max_sharpe["std"] * 100, capsize=4)
    ax.set(title="Maximum-Sharpe weight stability across experiments",
           ylabel="Mean weight ± 1 SD (%)", xlabel="")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=.15)
    fig.tight_layout(); fig.savefig(output/"robustness_weight_stability.png", dpi=170); plt.close(fig)


def write_robustness_report(output, results, weights, summary, stability, settings):
    """Persist tables, charts and a standalone robustness HTML report."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output/"robustness_experiments.csv", index=False)
    weights.to_csv(output/"robustness_weights.csv", index=False)
    summary.to_csv(output/"robustness_summary.csv", index_label="Strategy")
    stability.to_csv(output/"weight_stability.csv", index=False)
    (output/"robustness_settings.json").write_text(json.dumps(settings, indent=2))
    _plot_results(output, results, summary, stability)

    table = summary.copy()
    for col in ["mean_total_return", "mean_annual_volatility", "mean_maximum_drawdown", "holdout_win_rate"]:
        table[col] = table[col].map(lambda x: f"{x:.2%}")
    for col in ["mean_holdout_sharpe", "median_holdout_sharpe", "p05_holdout_sharpe",
                "p95_holdout_sharpe", "mean_training_sharpe", "mean_sharpe_gap"]:
        table[col] = table[col].map(lambda x: f"{x:.2f}")
    table = table.rename(columns={
        "experiments": "Runs", "mean_holdout_sharpe": "Mean holdout Sharpe",
        "median_holdout_sharpe": "Median holdout Sharpe", "p05_holdout_sharpe": "5th pct Sharpe",
        "p95_holdout_sharpe": "95th pct Sharpe", "mean_training_sharpe": "Mean training Sharpe",
        "mean_sharpe_gap": "Mean holdout-training gap", "mean_total_return": "Mean total return",
        "mean_annual_volatility": "Mean ann. volatility", "mean_maximum_drawdown": "Mean max drawdown",
        "holdout_win_rate": "Holdout win rate"
    })
    html_table = table.to_html(border=0, classes="metrics")
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio Risk Lab | Robustness report</title><style>
body{{font:15px/1.6 system-ui,sans-serif;color:#192B3E;background:#EDF2F6;margin:0}}main{{max-width:1080px;margin:auto;background:#fff;padding:36px}}
h1{{font-size:38px;margin:6px 0}}h2{{margin-top:34px}}.notice{{padding:16px;background:#FFF4DB;border-left:4px solid #D68B26}}
.meta{{color:#536479}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:12px}}th,td{{padding:9px;border-bottom:1px solid #DEE5EC;text-align:right}}th:first-child,td:first-child{{text-align:left}}
img{{width:100%;height:auto;margin:8px 0 22px}}@media(max-width:650px){{main{{padding:20px}}h1{{font-size:28px}}}}
</style><main><div class="meta">PORTFOLIO RISK LAB / ROBUSTNESS LAYER</div><h1>Repeated out-of-sample stress test</h1>
<p class="notice">Synthetic research only. These experiments test methodology under repeated fictional market paths and do not forecast real investments.</p>
<p>{settings['experiments']} independent market paths · three rotating volatility regimes · chronological {settings['train_fraction']:.0%}/{1-settings['train_fraction']:.0%} train/holdout split · no holdout data used during fitting.</p>
<h2>Summary</h2><div class="scroll">{html_table}</div>
<h2>Distribution, not one lucky path</h2><img src="robustness_holdout_sharpe.png"><img src="robustness_train_vs_holdout.png">
<h2>Strategy win rates</h2><img src="robustness_win_rates.png">
<h2>Weight stability</h2><img src="robustness_weight_stability.png">
<h2>Interpretation</h2><ul><li>Repeated paths reduce dependence on one particular random seed.</li><li>Training Sharpe is an estimate; the train-vs-holdout plot makes estimation error visible.</li><li>Win rates are descriptive within this synthetic design, not claims of real-world superiority.</li><li>Weight dispersion highlights how optimisation can react strongly to noisy sample estimates.</li></ul>
</main></html>'''
    (output/"robustness_report.html").write_text(report, encoding="utf-8")


def run_and_write_robustness(output, experiments=250, train_fraction=.8,
                             risk_free_rate=.03, periods=252, seed_start=1000):
    """Convenience wrapper used by the CLI and demo script."""
    results, weights = run_robustness(experiments, train_fraction, risk_free_rate, periods, seed_start)
    summary, stability = summarise_robustness(results, weights)
    settings = {"experiments": experiments, "train_fraction": train_fraction,
                "risk_free_rate": risk_free_rate, "periods_per_year": periods,
                "seed_start": seed_start, "regimes": REGIMES}
    write_robustness_report(output, results, weights, summary, stability, settings)
    return results, weights, summary, stability
