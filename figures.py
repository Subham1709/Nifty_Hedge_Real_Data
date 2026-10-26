"""Charts for the real-price backtest. Run after backtest.py:  python figures.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT, FIG = ROOT / "output", ROOT / "figures"
FIG.mkdir(exist_ok=True)

C = dict(blue="#2a78d6", orange="#eb6834", aqua="#1baf7a", yellow="#eda100",
         magenta="#e87ba4", green="#008300", violet="#4a3aa7", ink="#0b0b0b", ink2="#52514e", grid="#e4e3df")
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 160, "font.size": 9.5, "axes.edgecolor": C["ink2"],
    "axes.labelcolor": C["ink2"], "xtick.color": C["ink2"], "ytick.color": C["ink2"],
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": C["grid"], "grid.linewidth": 0.6, "lines.linewidth": 1.6,
    "axes.titleweight": "bold", "axes.titlesize": 10.5, "legend.frameon": False,
})
PROPOSED = "VIX-timed 3M put + hybrid monetisation (proposed)"
SHOW = {"Unhedged": C["ink"], "Always 1M 95% put": C["orange"], "Always 3M 95% put, rolled monthly": C["magenta"],
        "VIX-timed 3M put, hold": C["aqua"], PROPOSED: C["blue"]}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


vals = pd.read_csv(OUT / "backtest_values_real.csv", index_col=0, parse_dates=True)
ivd = pd.read_csv(OUT / "hedge_put_iv_daily.csv", index_col=0, parse_dates=True)
pc = pd.read_csv(OUT / "premium_model_vs_real.csv", parse_dates=["date"])
perf = {p: pd.read_csv(OUT / f"performance_{p}.csv", index_col=0) for p in ("original", "model", "real")}
rob = pd.read_csv(OUT / "robustness_real.csv")

# 1. growth and drawdown on real prices
fig, ax = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.6, 1]})
for n, c in SHOW.items():
    lw = 2.0 if n in (PROPOSED, "Unhedged") else 1.2
    ax[0].plot(vals.index, vals[n], color=c, lw=lw, label=n)
    ax[1].plot(vals.index, (vals[n] / vals[n].cummax() - 1) * 100, color=c, lw=lw)
ax[0].set_ylabel("Growth of 1 (price index)")
ax[0].legend(loc="upper left", fontsize=8)
ax[0].set_title("Backtest on real NSE option prices, Jan 2020 to Apr 2026")
ax[1].set_ylabel("Fall from peak (%)")
save(fig, "fig1_backtest_real.png")

# 2. implied vol of the 3-month hedge put: market vs model vs VIX
s = ivd[["VIX", "iv_3M", "iv_model_3M"]].rolling(5, min_periods=1).mean()
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(s.index, s.iv_model_3M, color=C["orange"], lw=1.2, label="Model (India VIX + 0.6 pts per 1% out of the money)")
ax.plot(s.index, s.iv_3M, color=C["blue"], lw=1.2, label="Market (implied from NSE prices)")
ax.plot(s.index, s.VIX, color=C["ink2"], lw=0.9, ls="--", label="India VIX")
ax.set_ylim(5, 45)
ax.set_ylabel("Implied volatility (%)")
ax.set_title("3-month 5% out-of-the-money Nifty put: the model priced it too high")
ax.legend(loc="upper right", fontsize=8)
save(fig, "fig2_put_iv_model_vs_market.png")

# 3. premium at each purchase, model vs market
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
for a, (n, title) in zip(ax, [("Always 1M 95% put", "1-month puts"), ("Always 3M 95% put, rolled monthly", "3-month puts")]):
    g = pc[pc.strategy == n]
    a.scatter(g.mid_real, g.mid_model, s=22, color=C["blue"], edgecolor="white", linewidth=0.6, zorder=3)
    hi = max(g.mid_real.max(), g.mid_model.max()) * 1.05
    a.plot([0, hi], [0, hi], color=C["ink2"], lw=0.9, ls="--")
    a.text(hi * 0.97, hi * 0.9, "model = market", ha="right", fontsize=8, color=C["ink2"])
    a.set_xlim(0, hi); a.set_ylim(0, hi)
    a.set_xlabel("Market premium (index points)"); a.set_ylabel("Model premium (index points)")
    a.set_title(f"{title}: model {g.model_over_real.median() - 1:+.0%} (median)")
save(fig, "fig3_premium_model_vs_market.png")

# 4. return a year under three pricing methods
order = list(perf["real"].index)
labels = ["Unhedged", "Always 1M put", "Always 1M put spread", "Always 3M put", "VIX-timed, hold",
          "VIX-timed, hybrid (proposed)", "VIX-timed, sell at 22"]
fig, ax = plt.subplots(figsize=(9, 3.8))
x = np.arange(len(order)); w = 0.27
for i, (p, lab, c) in enumerate([("original", "Original project (model prices)", C["ink2"]),
                                 ("model", "Model prices, real contracts", C["orange"]),
                                 ("real", "Real NSE prices", C["blue"])]):
    ax.bar(x + (i - 1) * w, perf[p].loc[order, "CAGR"], width=w * 0.9, color=c, label=lab)
ax.axhline(perf["real"].loc["Unhedged", "CAGR"], color=C["ink"], lw=0.8, ls=":")
ax.set_xticks(x, labels, rotation=20, ha="right", fontsize=8)
ax.set_ylabel("Return a year (%)")
ax.set_title("Same rules, three ways of pricing the puts")
ax.legend(fontsize=8, loc="upper left")
save(fig, "fig4_three_pricings.png")

# 5. robustness to the bid/ask haircut
h = rob[rob.group == "Bid/ask haircut"].copy()
h["c"] = h.label.str.rstrip("%").astype(float)
names = {"always_put": ("Always 1M put", C["orange"]), "timed_hold": ("Always 3M put", C["magenta"]),
         "timed_hybrid": ("VIX-timed, hybrid (proposed)", C["blue"])}
fig, ax = plt.subplots(figsize=(9, 3.4))
for s_, (lab, c) in names.items():
    g = h[h.strategy == s_]
    ax.plot(g.c, g.CAGR, color=c, marker="o", ms=5, label=lab)
ax.axhline(perf["real"].loc["Unhedged", "CAGR"], color=C["ink"], lw=1, ls="--", label="Unhedged")
ax.set_xlabel("Bid/ask haircut on every trade (% of the premium)")
ax.set_ylabel("Return a year (%)")
ax.set_title("Trading costs: the VIX-timed hedge barely notices them")
ax.legend(fontsize=8, ncol=2)
save(fig, "fig5_cost_sensitivity.png")
print("figures written")
