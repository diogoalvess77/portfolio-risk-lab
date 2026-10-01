"""Save the charts, result tables and a standalone HTML report."""
import base64
import html
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from .core import drawdown, portfolio_metrics

COLOURS = ["#68788D", "#087F8C", "#D68B26"]


def write_report(output, train, holdout, allocations, cloud, frontier, mean, covariance,
                 performance, summary, metadata):
    """Turn the analysis results into charts, CSV files and an HTML report.

All portfolio decisions have already been made before this function runs.
Charts show percentages; the CSV files keep the original decimal values."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "figure.facecolor": "white"})
    def finish(fig, name):
        if metadata["synthetic_demo"]:
            fig.text(.5, -.03, "SYNTHETIC DEMONSTRATION | Fictional asset prices", ha="center",
                     fontsize=8, color="#68788D")
        fig.savefig(output/name, dpi=160, bbox_inches="tight")
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    dots = ax.scatter(cloud.annual_volatility*100, cloud.annual_return_estimate*100,
                      c=cloud.sharpe_estimate, cmap="viridis", s=4, alpha=.35, rasterized=True)
    fig.colorbar(dots, ax=ax, label="Estimated annual Sharpe ratio")
    ax.plot(frontier.annual_volatility*100, frontier.annual_return_estimate*100,
            color="#132A43", lw=2, label="Numerical efficient frontier")
    for (name, weights), colour in zip(allocations.items(), COLOURS):
        stats = portfolio_metrics(weights, mean, covariance, metadata["risk_free_rate"])
        ax.scatter(stats["annual_volatility"]*100, stats["annual_return_estimate"]*100,
                   c=colour, edgecolors="white", s=110, label=name, zorder=5)
    ax.set(xlabel="Annualised volatility (%)", ylabel="Annualised arithmetic return estimate (%)",
           title="Training sample | risk and return")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=.15)
    finish(fig, "frontier.png")

    fig, ax = plt.subplots(figsize=(9, 4.3))
    base_date = train.index[-1]
    for (name, returns), colour in zip(performance.items(), COLOURS):
        wealth = pd.concat([pd.Series([100.], index=[base_date]), 100*(1+returns).cumprod()])
        ax.plot(wealth.index, wealth, label=name, color=colour, lw=2)
    ax.set(title="Holdout | buy-and-hold wealth", ylabel="Initial wealth = 100")
    ax.legend(fontsize=8); ax.grid(alpha=.15)
    finish(fig, "holdout.png")

    fig, ax = plt.subplots(figsize=(9, 3.6))
    for (name, returns), colour in zip(performance.items(), COLOURS):
        ax.plot(returns.index, drawdown(returns)*100, label=name, color=colour)
    ax.set(title="Holdout | drawdowns", ylabel="Drawdown (%)")
    ax.legend(fontsize=8); ax.grid(alpha=.15)
    finish(fig, "drawdowns.png")

    weight_frame = pd.DataFrame(allocations, index=train.columns)
    fig, ax = plt.subplots(figsize=(9, 4.3))
    (weight_frame.T*100).plot.bar(stacked=True, ax=ax, colormap="tab20c", width=.6)
    ax.set(title="Initial allocations | fitted on training data only", ylabel="Weight (%)", xlabel="")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1,1))
    finish(fig, "allocations.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    corr = train.corr()
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j,i,f"{corr.iloc[i,j]:.2f}",ha="center",va="center",
                    color="white" if abs(corr.iloc[i,j])>.65 else "#132A43",fontsize=9)
    ax.set_title("Training sample | daily return correlations")
    fig.colorbar(im,ax=ax,shrink=.7)
    finish(fig, "correlations.png")

    # Keep CSV values unrounded; format percentages only for the HTML table.
    weight_frame.to_csv(output/"allocations.csv", index_label="Asset")
    summary.to_csv(output/"holdout_metrics.csv", index_label="Portfolio")
    performance.to_csv(output/"holdout_daily_returns.csv", index_label="Date")
    frontier.to_csv(output/"efficient_frontier.csv", index=False)
    with (output/"run_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, allow_nan=False)
    formatted = summary.copy().astype(object)
    for col in summary:
        formatted[col] = summary[col].map(lambda v: "N/A" if pd.isna(v) else
                                          f"{v:.2f}" if col == "sharpe" else f"{v:.2%}")
    formatted.columns = ["Total return", "CAGR*", "Ann. volatility", "Sharpe", "Max drawdown",
                         "Daily VaR 95%", "Daily ES 95%"]
    table = formatted.to_html(border=0, classes="metrics", escape=True)
    def img(name):
        # Embed the images so the HTML report can be moved or shared on its own.
        encoded = base64.b64encode((output/name).read_bytes()).decode()
        return f'<img alt="{html.escape(name[:-4])}" src="data:image/png;base64,{encoded}">'
    data_note = ("SYNTHETIC DEMONSTRATION — fictional assets and prices; not historical market performance."
                 if metadata["synthetic_demo"] else
                 "USER-SUPPLIED CSV — source, currency and price-adjustment quality must be checked by the researcher.")
    best = summary["sharpe"].idxmax() if summary["sharpe"].notna().any() else "Not available"
    text = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Portfolio Risk Lab | Research report</title><style>
body{{font:15px/1.6 system-ui,sans-serif;color:#192B3E;background:#EDF2F6;margin:0}}
main{{max-width:1080px;margin:auto;background:white;padding:36px}}h1{{font-size:38px;margin:6px 0}}h2{{margin-top:36px}}
.eyebrow{{letter-spacing:2px;color:#087F8C;font-size:12px;font-weight:700}}.notice{{padding:16px;background:#FFF4DB;border-left:4px solid #D68B26}}
.meta{{color:#536479}}.scroll{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:10px;border-bottom:1px solid #DEE5EC;text-align:right}}th:first-child,td:first-child{{text-align:left}}
img{{width:100%;height:auto;margin:10px 0}}footer{{border-top:1px solid #DEE5EC;margin-top:30px;padding-top:15px;color:#536479}}@media(max-width:650px){{main{{padding:20px}}h1{{font-size:28px}}}}
</style><main><div class="eyebrow">PORTFOLIO RISK LAB / REPRODUCIBLE RESEARCH</div><h1>Allocation, risk &amp; the holdout test</h1>
<p class="notice">{data_note}</p><p class="meta">{metadata['asset_count']} assets · {metadata['sampled_portfolios']:,} sampled allocations · long-only · fully invested · annual risk-free input {metadata['risk_free_rate']:.1%}</p>
<p>Training: {metadata['train_start']} to {metadata['train_end']} ({len(train):,} returns). Holdout: {metadata['holdout_start']} to {metadata['holdout_end']} ({len(holdout):,} returns).</p>
<h2>1. Holdout performance</h2><p>Weights were fixed before this period. All three portfolios use buy-and-hold execution with drifting weights, no leverage and no trading costs. Best realised holdout Sharpe in this run: <b>{html.escape(best)}</b>.</p>
<div class="scroll">{table}</div><p class="meta">*CAGR is annualised using {metadata['periods_per_year']} observations per year, even for a holdout shorter than one year. VaR and ES are empirical daily losses, not guaranteed loss limits. Negative tail-loss values indicate gains.</p>
{img('holdout.png')}{img('drawdowns.png')}
<h2>2. Training estimates and optimisation</h2><p>The cloud samples portfolio weights, not future price paths. Expected returns are annualised historical sample means. The efficient frontier is computed separately with constrained optimisation.</p>{img('frontier.png')}
<h2>3. Allocation and diversification</h2>{img('allocations.png')}{img('correlations.png')}
<h2>4. Interpretation and limitations</h2><ul><li>A higher training Sharpe need not survive out of sample; estimates of mean returns are noisy.</li>
<li>The training model assumes constant portfolio weights for its moment estimates. The holdout uses a single initial allocation and buy-and-hold drift.</li>
<li>One chronological holdout does not establish robustness. Repeatedly choosing models after viewing this holdout contaminates the test.</li>
<li>No fees, taxes, transaction costs, turnover penalties, position caps, currency conversion or estimation-error adjustment are included.</li>
<li>Synthetic data omit market frictions, fat tails and structural breaks. Real-CSV analyses can still contain survivorship and universe-selection bias.</li></ul>
<footer>Educational research project. See README.md, docs/METHODOLOGY.md and run_metadata.json for the equations, assumptions and reproducibility details.</footer></main></html>'''
    (output/"report.html").write_text(text, encoding="utf-8")
