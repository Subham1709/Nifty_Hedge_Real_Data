"""VIX-timed Nifty put hedge, rebuilt on real NSE option prices.

Run:  python backtest.py      (about a minute; writes output/ and output/results.json)

Data: data/bhavcopy/ (NSE F&O bhavcopies, Jan 2020 to Apr 2026, downloaded with
data_download/nse_fo_download.py), data/nifty50_daily_raw.csv and data/india_vix_daily_raw.csv
(the same spot series as the original project). Backtest period and trading rules are the
original project's, so results can be compared one for one.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import engine as E  # noqa: E402
from chain import Chain, implied_vol, load_bhavcopy  # noqa: E402

DATA, OUT = ROOT / "data", ROOT / "output"
OUT.mkdir(exist_ok=True)
ORIGINAL_PERF = ROOT / "data" / "original_backtest_performance.csv"   # copied from the original project
ORIGINAL_VALUES = ROOT / "data" / "original_backtest_values.csv"
RESULTS = {}

# ---------------------------------------------------------------- data
nifty = pd.read_csv(DATA / "nifty50_daily_raw.csv", parse_dates=["Date"]).set_index("Date")[["Close"]]
vix = pd.read_csv(DATA / "india_vix_daily_raw.csv", encoding="utf-8-sig", parse_dates=["Date"])
vix = vix.rename(columns={"Sum of Close": "VIX"}).set_index("Date")[["VIX"]]
bt = nifty.join(vix, how="inner").dropna().astype(float)
bt = bt[~bt.index.duplicated()].sort_index()
bt["regime"] = pd.cut(bt.VIX, [0, E.LOW_VIX, 20, 1e9], labels=["Low", "Normal", "High"])

bhav = load_bhavcopy(DATA / "bhavcopy")
chain = Chain(bhav)
opt = bhav[bhav.instrument == "OPTIDX"]
RESULTS["data"] = dict(
    start=str(bt.index[0].date()), end=str(bt.index[-1].date()), days=len(bt),
    bhavcopy_days=int(bhav.date.nunique()), option_rows=int(len(opt)),
    put_rows=int((opt.opt == "PE").sum()),
    backtest_days_without_bhavcopy=[str(d.date()) for d in bt.index if d not in chain.dates],
    min_volume=chain.min_volume)
yrs = (bt.index[-1] - bt.index[0]).days / 365.25

# ---------------------------------------------------------------- main backtest
PROPOSED = "VIX-timed 3M put + hybrid monetisation (proposed)"
STRATS = {"Unhedged": ("unhedged", {}), "Always 1M 95% put": ("always_put", {}),
          "Always 1M 95/85 put spread": ("always_spread", {}),
          "Always 3M 95% put, rolled monthly": ("timed_hold", {"low": 1e9}),
          "VIX-timed 3M put, hold": ("timed_hold", {}),
          PROPOSED: ("timed_hybrid", {}),
          "VIX-timed 3M put, sell all at VIX 22": ("timed_all", {})}

runs, ledgers, stats = {}, {}, {}
for pricing in ("real", "model"):
    for name, (s, kw) in STRATS.items():
        o, led, st = E.run(s, bt, chain, pricing=pricing, **kw)
        runs[pricing, name], ledgers[pricing, name], stats[pricing, name] = o, led, st


def perf_table(pricing):
    rows = {}
    for name in STRATS:
        o, led = runs[pricing, name], ledgers[pricing, name]
        m = E.metrics(o.value)
        m.update(E.hedge_cost(led, o.value, yrs))
        m["share_time_hedged"] = (o.notional_hedged > 0).mean() * 100
        m["puts_bought"] = int(((led.action == "open") & (led.qty > 0)).sum()) if len(led) else 0
        rows[name] = m
    return pd.DataFrame(rows).T


perf = {p: perf_table(p) for p in ("real", "model")}
perf["original"] = pd.read_csv(ORIGINAL_PERF, index_col=0)
for p, t in perf.items():
    t.round(3).to_csv(OUT / f"performance_{p}.csv")
vals = pd.DataFrame({n: runs["real", n].value for n in STRATS})
vals.to_csv(OUT / "backtest_values_real.csv")
pd.DataFrame({n: runs["model", n].value for n in STRATS}).to_csv(OUT / "backtest_values_model.csv")
for i, n in enumerate(STRATS):
    if len(ledgers["real", n]):
        ledgers["real", n].to_csv(OUT / f"ledger_real_s{i}.csv", index=False)
print(perf["real"][["CAGR", "vol", "sharpe", "maxDD", "net_hedge_cost_pa", "puts_bought"]].round(2).to_string())

comp = pd.concat({"Original (model, synthetic expiries)": perf["original"][["CAGR", "maxDD", "sharpe", "net_hedge_cost_pa"]],
                  "Model price, real contracts": perf["model"][["CAGR", "maxDD", "sharpe", "net_hedge_cost_pa"]],
                  "Real NSE prices": perf["real"][["CAGR", "maxDD", "sharpe", "net_hedge_cost_pa"]]}, axis=1)
comp.round(2).to_csv(OUT / "comparison_real_vs_model.csv")
RESULTS["strategies"] = list(STRATS)
RESULTS["performance"] = {p: t.round(3).reset_index().rename(columns={"index": "strategy"}).to_dict(orient="records")
                          for p, t in perf.items()}

# where real prices came from (liquidity)
src = pd.DataFrame({n: stats["real", n] for n in STRATS if n != "Unhedged"}).T
src.to_csv(OUT / "price_sources.csv")
allled = pd.concat([ledgers["real", n].assign(strategy=n) for n in STRATS if n != "Unhedged"])
opens = allled[allled.action == "open"]
RESULTS["liquidity"] = dict(
    marks=src.drop(columns="skipped_opens").sum().astype(int).to_dict(),
    trades=int(len(allled[allled.action != "expiry"])),
    trades_by_source=allled[allled.action != "expiry"].source.value_counts().to_dict(),
    opens_median_volume=float(opens.volume.median()), opens_median_oi=float(opens.oi.median()),
    opens_moneyness_mean=float(opens[opens.qty > 0].moneyness.mean()),
    opens_moneyness_range=[float(opens[opens.qty > 0].moneyness.min()), float(opens[opens.qty > 0].moneyness.max())],
    opens_days_to_expiry={"1M": [int(x) for x in opens[(opens.strategy == "Always 1M 95% put")].days_left.agg(["min", "median", "max"])],
                          "3M": [int(x) for x in opens[(opens.strategy == "Always 3M 95% put, rolled monthly")].days_left.agg(["min", "median", "max"])]})

# ---------------------------------------------------------------- model vs market: premiums
cmp_rows = []
for n in ["Always 1M 95% put", "Always 3M 95% put, rolled monthly"]:
    x = ledgers["real", n]
    x = x[(x.action == "open") & (x.qty > 0)].copy()
    x["strategy"] = n
    cmp_rows.append(x)
pc = pd.concat(cmp_rows)
pc["model_over_real"] = pc.mid_model / pc.mid_real
pc[["strategy", "date", "K", "expiry", "days_left", "moneyness", "vix", "mid_real", "mid_model", "model_over_real",
    "volume", "oi"]].to_csv(OUT / "premium_model_vs_real.csv", index=False, float_format="%.4f")
RESULTS["premium_model_vs_real"] = {
    n: dict(n=int(len(g)), median_ratio=float(g.model_over_real.median()),
            share_model_dearer=float((g.model_over_real > 1).mean()),
            low_vix_median_ratio=float(g[g.vix < E.LOW_VIX].model_over_real.median()),
            high_vix_median_ratio=float(g[g.vix >= 20].model_over_real.median()) if (g.vix >= 20).any() else None)
    for n, g in pc.groupby("strategy")}

# ---------------------------------------------------------------- daily implied vol of the hedge put
iv_rows = []
for d, row in bt.iterrows():
    S = row.Close
    rec = dict(date=d, VIX=row.VIX)
    for months, tag in ((1, "1M"), (3, "3M")):
        sel = chain.select(d, S, months, E.PUT_MONEYNESS)
        if sel is None:
            continue
        q = chain.fair(d, sel["exp"], sel["K"], S)
        if q is None:
            continue
        T = (sel["exp"] - d).days / 365
        iv = implied_vol(q[0], chain.forward(d, sel["exp"], S), sel["K"], T)
        otm = max(0.0, (1 - sel["K"] / S) * 100)
        rec[f"iv_{tag}"] = iv * 100 if iv == iv else np.nan
        rec[f"iv_model_{tag}"] = row.VIX + E.SKEW_PER_PCT * otm
        rec[f"cost_{tag}_pct"] = q[0] / S * 100
        rec[f"cost_model_{tag}_pct"] = E.bs_put(S, sel["K"], T, rec[f"iv_model_{tag}"] / 100) / S * 100
    iv_rows.append(rec)
ivd = pd.DataFrame(iv_rows).set_index("date")
ivd.round(4).to_csv(OUT / "hedge_put_iv_daily.csv")
by_reg = ivd.join(bt.regime).groupby("regime", observed=True)[["VIX", "iv_3M", "iv_model_3M", "iv_1M", "iv_model_1M"]].median()
RESULTS["iv_daily"] = dict(
    median=ivd[["VIX", "iv_1M", "iv_model_1M", "iv_3M", "iv_model_3M"]].median().round(2).to_dict(),
    by_regime=by_reg.round(2).to_dict(orient="index"),
    share_3M_real_below_model=float((ivd.iv_3M < ivd.iv_model_3M).mean()),
    median_gap_3M=float((ivd.iv_3M - ivd.iv_model_3M).median()),
    median_gap_1M=float((ivd.iv_1M - ivd.iv_model_1M).median()))

# ---------------------------------------------------------------- episodes
episodes = {"COVID crash (17-Jan to 23-Mar-20)": ("2020-01-17", "2020-03-23"),
            "2022 war/rates (17-Jan to 17-Jun-22)": ("2022-01-17", "2022-06-17"),
            "Election result day (3 to 4-Jun-24)": ("2024-06-03", "2024-06-04"),
            "Yen-carry unwind (1 to 5-Aug-24)": ("2024-08-01", "2024-08-05"),
            "Sep-24 to Mar-25 correction": ("2024-09-26", "2025-03-04"),
            "Apr-25 tariff shock (28-Mar to 7-Apr)": ("2025-03-28", "2025-04-07"),
            "Mar-26 fall (2-Jan to 30-Mar-26)": ("2026-01-02", "2026-03-30"),
            "Full period": (str(vals.index[0].date()), str(vals.index[-1].date()))}
ep = pd.DataFrame({k: (vals.asof(pd.Timestamp(b)) / vals.loc[vals.index >= a].iloc[0] - 1) * 100
                   for k, (a, b) in episodes.items()}).T
ep.round(2).to_csv(OUT / "episode_returns_real.csv")
RESULTS["episodes"] = ep.round(2).to_dict(orient="index")

# trades of the proposed strategy
L = ledgers["real", PROPOSED]
RESULTS["proposed_trades"] = L.assign(date=L.date.astype(str), expiry=L.expiry.astype(str))[
    ["date", "action", "K", "expiry", "days_left", "moneyness", "price", "source", "vix", "mid_model"]].round(3).to_dict(orient="records")
tr = L.groupby("trade_id").agg(cash=("cash", "sum"))
RESULTS["proposed_summary"] = dict(puts=int(len(tr)), winners=int((tr.cash > 0).sum()))

# ---------------------------------------------------------------- robustness
rob = []


def add(group, label, strategy, pricing="real", ch=chain, **kw):
    o, led, _ = E.run(strategy, bt, ch, pricing=pricing, **kw)
    m = E.metrics(o.value)
    mp = E.metrics(o.value.loc["2020-07-01":])
    rob.append(dict(group=group, label=label, strategy=strategy, CAGR=m["CAGR"], maxDD=m["maxDD"],
                    sharpe=m["sharpe"], CAGR_ex_covid=mp["CAGR"], net_cost_pa=E.hedge_cost(led, o.value, yrs)["net_hedge_cost_pa"]))


for c in (0.0, 0.01, 0.02, 0.05, 0.10):
    for s, kw in (("always_put", {}), ("timed_hold", {"low": 1e9}), ("timed_hybrid", {})):
        add("Bid/ask haircut", f"{c:.0%}", s, cost=c, **kw)
for mv in (10, 200):
    ch2 = Chain(bhav, min_volume=mv)
    for s, kw in (("always_put", {}), ("timed_hold", {"low": 1e9}), ("timed_hybrid", {})):
        add("Liquidity cut-off", f"{mv} contracts", s, ch=ch2, **kw)
for s, kw in (("always_put", {}), ("timed_hold", {"low": 1e9}), ("timed_hybrid", {})):
    add("Liquidity cut-off", f"{chain.min_volume} contracts", s, **kw)
for ten in (1, 2, 3, 6):
    add("Tenor (proposed rule)", f"{ten}M", "timed_hybrid", tenor=ten)
for mny in (0.90, 0.95, 1.00):
    add("Strike (proposed rule)", f"{int(mny * 100)}%", "timed_hybrid", moneyness=mny)
for low in (12, 13, 14, 15):
    for s1 in (20, 22, 25):
        add("Thresholds (proposed rule)", f"buy<{low}, sell at {s1}", "timed_hybrid", low=low, spike1=s1)
rob = pd.DataFrame(rob)
rob.round(3).to_csv(OUT / "robustness_real.csv", index=False)
RESULTS["robustness"] = rob.round(2).to_dict(orient="records")
RESULTS["ex_covid"] = {n: {k: round(float(v), 3) for k, v in E.metrics(vals[n].loc["2020-07-01":]).items()} for n in STRATS}

(OUT / "results.json").write_text(json.dumps(RESULTS, indent=1, default=str))
print(comp.round(2).to_string())
print(json.dumps({k: RESULTS[k] for k in ("liquidity", "premium_model_vs_real", "iv_daily")}, indent=1, default=str))
