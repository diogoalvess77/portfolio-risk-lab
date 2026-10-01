"""Sensitivity grid for train/holdout split and risk-free-rate assumptions."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .demo import generate_demo_frame
from .core import split_returns, estimate_moments, optimise_portfolios, buy_and_hold, realised_metrics


def run_sensitivity(train_fractions=(.70, .80, .90), risk_free_rates=(0.00, .03, .05), seed=7):
    """Evaluate the same synthetic path under a small assumption grid."""
    prices = generate_demo_frame(seed=seed)
    rows = []
    for train_fraction in train_fractions:
        train, holdout = split_returns(prices, train_fraction)
        mean, covariance = estimate_moments(train)
        for risk_free_rate in risk_free_rates:
            allocations, _ = optimise_portfolios(mean, covariance, risk_free_rate, frontier_points=0)
            for strategy, weights in allocations.items():
                metrics = realised_metrics(buy_and_hold(holdout, weights), risk_free_rate)
                rows.append({"train_fraction": train_fraction, "risk_free_rate": risk_free_rate,
                             "strategy": strategy, **metrics})
    return pd.DataFrame(rows)


def write_sensitivity(output, frame):
    """Save raw sensitivity results, a Sharpe matrix, chart and small HTML report."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "sensitivity_results.csv", index=False)
    matrix = frame.pivot_table(index=["train_fraction", "risk_free_rate"], columns="strategy", values="sharpe")
    matrix.to_csv(output / "sensitivity_sharpe_matrix.csv")

    labels = [f"{tf:.0%} train | rf {rf:.0%}" for tf, rf in matrix.index]
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="RdBu_r")
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=20, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Holdout Sharpe sensitivity to train split and risk-free assumption")
    fig.colorbar(im, ax=ax, label="Realised holdout Sharpe")
    fig.tight_layout()
    fig.savefig(output / "sensitivity_sharpe.png", dpi=170)
    plt.close(fig)

    table = matrix.round(3).to_html(classes="metrics", border=0)
    report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio Risk Lab | Sensitivity</title><style>body{{font:15px/1.6 system-ui,sans-serif;color:#192B3E;background:#EDF2F6;margin:0}}main{{max-width:1000px;margin:auto;background:#fff;padding:36px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}th:first-child,td:first-child{{text-align:left}}.notice{{padding:14px;background:#FFF4DB;border-left:4px solid #D68B26}}</style>
<main><h1>Assumption sensitivity</h1><p class="notice">Synthetic demonstration only. The grid varies modelling assumptions on one reproducible fictional path.</p><p>Train/holdout splits: 70/30, 80/20 and 90/10. Annual risk-free inputs: 0%, 3% and 5%.</p>{table}<img src="sensitivity_sharpe.png"><p>The purpose is not to find a preferred setting, but to show how conclusions can move when modelling assumptions change.</p></main></html>'''
    (output / "sensitivity_report.html").write_text(report, encoding="utf-8")
    return matrix
