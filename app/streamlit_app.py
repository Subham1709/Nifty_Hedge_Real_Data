"""Nifty crash insurance on real NSE option prices: a web app.

Results are read from output/ (written by backtest.py). The "Try it yourself" tab re-runs the
backtest live on the real option prices in data/bhavcopy/.

Run locally from the repo root:  streamlit run app/streamlit_app.py
"""
import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
OUT, DATA = ROOT / "output", ROOT / "data"
sys.path.insert(0, str(ROOT / "src"))
import engine as E  # noqa: E402
from chain import Chain, load_bhavcopy  # noqa: E402

AUTHOR = "Subh"
REPO_URL = "https://github.com/Subham1709/Nifty_Hedge_Real_Data"
ORIGINAL_APP = "https://subham-fdrm-vix.streamlit.app"

PROPOSED = "VIX-timed 3M put + hybrid monetisation (proposed)"
SHORT = {"Unhedged": "Unhedged", "Always 1M 95% put": "Always 1M put",
         "Always 1M 95/85 put spread": "Always 1M put spread", "Always 3M 95% put, rolled monthly": "Always 3M put",
         "VIX-timed 3M put, hold": "VIX-timed 3M put, hold", PROPOSED: "VIX-timed 3M put, hybrid sell (proposed)",
         "VIX-timed 3M put, sell all at VIX 22": "VIX-timed 3M put, sell all at 22"}
COLOURS = {"Unhedged": "#222222", "Always 1M put": "#e8772e", "Always 1M put spread": "#e8b02e",
           "Always 3M put": "#c2185b", "VIX-timed 3M put, hold": "#1baf7a",
           "VIX-timed 3M put, hybrid sell (proposed)": "#1f6fd1", "VIX-timed 3M put, sell all at 22": "#4a3aa7",
           "Real NSE prices": "#1f6fd1", "Model prices": "#e8772e", "India VIX": "#777777",
           "Market (NSE prices)": "#1f6fd1", "Model (VIX + skew)": "#e8772e"}
METRICS = {"CAGR": "Return a year (%)", "vol": "Volatility (%)", "sharpe": "Sharpe", "maxDD": "Worst fall (%)",
           "net_hedge_cost_pa": "Insurance cost a year (%)", "share_time_hedged": "Time insured (%)",
           "puts_bought": "Puts bought"}

st.set_page_config(page_title="Nifty crash insurance on real option prices", layout="wide")


@st.cache_data
def load_csv(name, **kw):
    return pd.read_csv(OUT / name, **kw)


@st.cache_resource(show_spinner="Loading 1.2 million NSE option prices...")
def load_chain():
    return Chain(load_bhavcopy(DATA / "bhavcopy"))


@st.cache_data
def load_bt():
    n = pd.read_csv(DATA / "nifty50_daily_raw.csv", parse_dates=["Date"]).set_index("Date")[["Close"]]
    v = pd.read_csv(DATA / "india_vix_daily_raw.csv", encoding="utf-8-sig", parse_dates=["Date"])
    v = v.rename(columns={"Sum of Close": "VIX"}).set_index("Date")[["VIX"]]
    bt = n.join(v, how="inner").dropna().astype(float)
    bt = bt[~bt.index.duplicated()].sort_index()
    bt["regime"] = pd.cut(bt.VIX, [0, 13, 20, 1e9], labels=["Low", "Normal", "High"])
    return bt


R = json.loads((OUT / "results.json").read_text())
perf = {p: load_csv(f"performance_{p}.csv", index_col=0) for p in ("real", "model", "original")}
vals = load_csv("backtest_values_real.csv", index_col=0, parse_dates=True)
ivd = load_csv("hedge_put_iv_daily.csv", index_col=0, parse_dates=True)
pc = load_csv("premium_model_vs_real.csv", parse_dates=["date", "expiry"])
rob = load_csv("robustness_real.csv")
ep = load_csv("episode_returns_real.csv", index_col=0)
real, orig = perf["real"], perf["original"]
p_r, p_o, u = real.loc[PROPOSED], orig.loc[PROPOSED], real.loc["Unhedged"]
pm = R["premium_model_vs_real"]
ratio3 = pm["Always 3M 95% put, rolled monthly"]["median_ratio"]
ratio1 = pm["Always 1M 95% put"]["median_ratio"]


def line_chart(df, title, fmt=".2f", height=320):
    long = df.rename_axis("date").reset_index().melt("date", var_name="Series", value_name="v").dropna()
    names = list(df.columns)
    return alt.Chart(long).mark_line(strokeWidth=1.4).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("v:Q", title=title, scale=alt.Scale(zero=False)),
        color=alt.Color("Series:N", scale=alt.Scale(domain=names, range=[COLOURS.get(n, "#888888") for n in names]),
                        legend=alt.Legend(orient="top", columns=3, title=None, labelLimit=400)),
        tooltip=["date:T", "Series", alt.Tooltip("v:Q", format=fmt)],
    ).properties(height=height)


def perf_view(t):
    t = t.rename(index=SHORT)[list(METRICS)].rename(columns=METRICS).astype(float)
    return t.style.format("{:.2f}").format("{:.0f}", subset=["Puts bought"])


# ---------- header ----------
st.title("Nifty crash insurance, tested on real NSE option prices")
st.markdown(
    f"By **{AUTHOR}** · [Code on GitHub]({REPO_URL}) · Extends a group course project "
    f"(FDRM, IIT Jodhpur) whose [original app]({ORIGINAL_APP}) priced options with a model  \n"
    f"{R['data']['option_rows']:,} daily Nifty option prices from NSE's official F&O bhavcopies, "
    f"{R['data']['start']} to {R['data']['end']}."
)
c = st.columns(4)
c[0].metric("VIX-timed hedge, return a year", f"{p_r.CAGR:.1f}%", f"{p_r.CAGR - u.CAGR:+.1f} pts vs no hedge")
c[1].metric("Worst fall", f"{p_r.maxDD:.0f}%", f"{p_r.maxDD - u.maxDD:+.0f} pts vs no hedge")
c[2].metric("Model premium vs real, 3-month puts", f"{ratio3 - 1:+.0%}", "median over 76 purchases",
            delta_color="off")
c[3].metric("Same rule, model prices", f"{p_o.CAGR:.1f}%", f"real prices {p_r.CAGR - p_o.CAGR:+.1f} pts",
            delta_color="off")

tabs = st.tabs(["Introduction", "Model vs market", "Backtest", "Robustness", "Try it yourself", "Data and method"])

# ---------- 0. introduction ----------
with tabs[0]:
    st.subheader("The question")
    st.markdown(
        "A Nifty 50 investor can buy **put options** as crash insurance: a put pays if the index ends below a set "
        "level. The course project tested a simple rule. On the first trading day of each month, buy a 3-month put "
        "5% below the market **only if India VIX is below 13** (insurance is cheap). Sell half when VIX jumps to 22 "
        "and the rest once the spike fades.\n\n"
        "That project had no historical option prices, so every premium came from the Black-Scholes model with "
        "India VIX as the volatility, plus an assumed extra charge for out-of-the-money puts. A rule that looks "
        "good on model prices might not survive real ones. This version replaces every model price with the "
        "price the option actually traded at on NSE."
    )
    st.subheader("What changed with real prices")
    iv = R["iv_daily"]
    st.markdown(
        f"- **Real puts were cheaper than the model said.** The model's premium was a median {ratio3 - 1:.0%} too "
        f"high for 3-month puts and {ratio1 - 1:.0%} too high for 1-month puts. The market charged an implied "
        f"volatility of about VIX + {iv['median']['iv_3M'] - iv['median']['VIX']:.0f} point for the 3-month put, "
        f"not the VIX + 3 the model assumed.\n"
        f"- **The rule held up.** On real prices the VIX-timed hedge returned {p_r.CAGR:.1f}% a year against "
        f"{u.CAGR:.1f}% with no protection, and its worst fall was {abs(p_r.maxDD):.0f}% instead of "
        f"{abs(u.maxDD):.0f}%. On model prices it was {p_o.CAGR:.1f}%.\n"
        f"- **Every hedge looks better, but the ranking is the same.** Buying insurance every month still cost "
        f"more than it saved; buying only when VIX is low still beat no hedge.\n"
        f"- **Liquidity was not a problem.** {R['liquidity']['trades_by_source'].get('traded', 0)} of "
        f"{R['liquidity']['trades']} trades used the option's own price on a day it traded at least "
        f"{R['data']['min_volume']} contracts (median {R['liquidity']['opens_median_volume']:,.0f} on the day "
        "each put was bought)."
    )
    st.altair_chart(line_chart(vals[[n for n in SHORT if n in vals]].rename(columns=SHORT),
                               "Growth of 1 (price index)").interactive(bind_y=False), width="stretch")
    st.caption("Real NSE prices, with a 2% bid/ask haircut on every trade. Drag to zoom.")
    with st.expander("Glossary"):
        st.markdown(
            "- **Put option:** insurance that pays if Nifty ends below the strike price on expiry.\n"
            "- **5% out of the money:** the strike is 5% below today's Nifty, so the put pays only after a fall of more than 5%.\n"
            "- **India VIX:** the volatility the option market expects over the next 30 days, in % a year.\n"
            "- **Implied volatility:** the volatility that makes the Black-Scholes price equal the market price.\n"
            "- **Bhavcopy:** NSE's official end-of-day file of every derivative's prices, volume and open interest.\n"
            "- **CAGR:** average growth a year, compounded. **Worst fall:** biggest drop from a previous high."
        )

# ---------- 1. model vs market ----------
with tabs[1]:
    st.markdown(
        "Each day, the 3-month put the strategy would buy (listed expiry nearest 3 months, strike nearest 5% "
        "below Nifty) is valued two ways: what NSE prices imply, and what the original model assumed. The model "
        "used India VIX plus 0.6 volatility points for every 1% out of the money (3 points for this put)."
    )
    tenor = st.radio("Put", ["3-month", "1-month"], horizontal=True)
    t = "3M" if tenor == "3-month" else "1M"
    df = ivd[["VIX", f"iv_{t}", f"iv_model_{t}"]].rolling(5, min_periods=1).mean()
    df.columns = ["India VIX", "Market (NSE prices)", "Model (VIX + skew)"]
    st.altair_chart(line_chart(df, "Implied volatility (%)", ".1f").interactive(bind_y=False), width="stretch")
    st.caption("Five-day averages. The COVID spike in March 2020 reaches about 75 to 80.")
    br = pd.DataFrame(R["iv_daily"]["by_regime"]).T[["VIX", f"iv_{t}", f"iv_model_{t}"]]
    br.columns = ["India VIX", "Market", "Model"]
    br.index = [f"{i} VIX days" for i in br.index]
    st.markdown("**Median implied volatility by VIX level**")
    st.dataframe(br.style.format("{:.1f}"), width="content")

    st.subheader("Premium paid at each purchase")
    g = pc[pc.strategy == ("Always 3M 95% put, rolled monthly" if t == "3M" else "Always 1M 95% put")].copy()
    hi = float(max(g.mid_real.max(), g.mid_model.max())) * 1.05
    pts = alt.Chart(g).mark_circle(size=60, color="#1f6fd1", opacity=0.85).encode(
        x=alt.X("mid_real:Q", title="Market premium (index points)", scale=alt.Scale(domain=[0, hi])),
        y=alt.Y("mid_model:Q", title="Model premium (index points)", scale=alt.Scale(domain=[0, hi])),
        tooltip=[alt.Tooltip("date:T", title="Bought"), alt.Tooltip("K:Q", title="Strike"),
                 alt.Tooltip("expiry:T", title="Expiry"), alt.Tooltip("vix:Q", title="VIX", format=".1f"),
                 alt.Tooltip("mid_real:Q", title="Market", format=".1f"),
                 alt.Tooltip("mid_model:Q", title="Model", format=".1f")])
    diag = alt.Chart(pd.DataFrame({"x": [0, hi], "y": [0, hi]})).mark_line(color="#777777", strokeDash=[4, 4]).encode(x="x", y="y")
    st.altair_chart((diag + pts).properties(height=380), width="stretch")
    st.markdown(
        f"Points above the dashed line are purchases where the model charged more than the market. The model was "
        f"higher in {pm['Always 3M 95% put, rolled monthly']['share_model_dearer']:.0%} of 3-month purchases. "
        "The gap is widest when VIX is low, which is exactly when the rule buys: real insurance was even cheaper "
        "than the model suggested at those moments."
    )

# ---------- 2. backtest ----------
with tabs[2]:
    st.markdown(
        "Portfolio: 100% Nifty 50 (price index). Puts are paid for by selling Nifty, and every rupee the puts "
        "return goes back into Nifty the same day. Puts are real listed contracts held to their real expiry."
    )
    st.markdown("**Results on real NSE prices**")
    st.dataframe(perf_view(real), width="stretch")
    st.caption("Insurance cost: premiums paid minus money recovered, as % of the portfolio a year. "
               "Negative means the puts earned more than they cost.")
    st.markdown("**Same rules, three ways of pricing the puts (return a year, %)**")
    comp = pd.DataFrame({"Original project (model, synthetic expiries)": orig.CAGR,
                         "Model prices, real contracts": perf["model"].CAGR,
                         "Real NSE prices": real.CAGR}).rename(index=SHORT)
    st.dataframe(comp.style.format("{:.2f}"), width="stretch")
    st.markdown(
        "The middle column keeps the real contracts and dates but prices them with the model, so the step from "
        "the middle to the right column is the effect of real prices alone."
    )
    st.markdown("**Returns in each market fall (%)**")
    st.dataframe(ep.rename(columns=SHORT).style.format("{:.1f}"), width="stretch")
    with st.expander("Every trade of the proposed strategy"):
        L = pd.DataFrame(R["proposed_trades"])
        st.dataframe(L.rename(columns={"date": "Date", "action": "Action", "K": "Strike", "expiry": "Expiry",
                                       "days_left": "Days to expiry", "moneyness": "Strike / Nifty",
                                       "price": "Price paid or received", "source": "Price source",
                                       "vix": "VIX", "mid_model": "Model price"}), width="stretch", hide_index=True)

# ---------- 3. robustness ----------
with tabs[3]:
    st.markdown("Each table changes one choice and re-runs the backtest on real prices.")
    for grp, note in [
        ("Bid/ask haircut", "Bhavcopies have no bid/ask quotes, so every buy pays the price plus a haircut and "
                            "every sale gets the price minus it. The main results use 2%."),
        ("Liquidity cut-off", "A strike's own price is used only if at least this many contracts traded that "
                              "day; otherwise it is read off the day's volatility smile from liquid strikes."),
        ("Tenor (proposed rule)", "A 1-month or 2-month put bought in January 2020 expired before the COVID crash."),
        ("Strike (proposed rule)", "Strike as % of Nifty when the put is bought."),
        ("Thresholds (proposed rule)", "Buy only if VIX is below the first number; sell half at the second.")]:
        st.markdown(f"**{grp}.** {note}")
        g = rob[rob.group == grp].copy()
        g["strategy"] = g.strategy.map({"always_put": "Always 1M put", "timed_hold": "Always 3M put",
                                        "timed_hybrid": "VIX-timed (proposed)"})
        g = g[["label", "strategy", "CAGR", "maxDD", "sharpe", "CAGR_ex_covid", "net_cost_pa"]].rename(columns={
            "label": "Setting", "strategy": "Strategy", "CAGR": "Return a year (%)", "maxDD": "Worst fall (%)",
            "sharpe": "Sharpe", "CAGR_ex_covid": "Return a year from Jul 2020 (%)",
            "net_cost_pa": "Insurance cost a year (%)"})
        st.dataframe(g.style.format(precision=2), width="stretch", hide_index=True)
    xc = R["ex_covid"]
    st.markdown(
        f"**Without the COVID crash.** From July 2020 the proposed rule returned {xc[PROPOSED]['CAGR']:.1f}% a year "
        f"against {xc['Unhedged']['CAGR']:.1f}% unhedged. Much of the full-period lead comes from one put bought "
        "in January 2020, which is the main limitation of this result: six years contain only a handful of "
        "crashes."
    )

# ---------- 4. try it ----------
with tabs[4]:
    st.markdown("Change the rule and re-run it on real NSE prices (and on model prices for comparison).")
    a, b, c_ = st.columns(3)
    low = a.slider("Buy only if VIX is below", 10.0, 20.0, 13.0, 0.5)
    spike = a.slider("Sell half when VIX reaches", 18.0, 35.0, 22.0, 1.0)
    tenor_m = b.select_slider("Put length (months)", [1, 2, 3, 6], 3)
    mny = b.select_slider("Strike (% of Nifty)", [90, 95, 100], 95) / 100
    rule = c_.radio("Selling rule", ["Hybrid (half at the spike, rest when it fades)", "Hold to expiry",
                                     "Sell all at the spike"])
    cost = c_.select_slider("Bid/ask haircut", [0.0, 0.01, 0.02, 0.05, 0.10], 0.02, format_func=lambda x: f"{x:.0%}")
    strat = {"Hybrid (half at the spike, rest when it fades)": "timed_hybrid", "Hold to expiry": "timed_hold",
             "Sell all at the spike": "timed_all"}[rule]
    chain, bt = load_chain(), load_bt()
    kw = dict(low=low, spike1=spike, tenor=tenor_m, moneyness=mny)
    o_r, led, _ = E.run(strat, bt, chain, pricing="real", cost=cost, **kw)
    o_m, _, _ = E.run(strat, bt, chain, pricing="model", **kw)
    rows = {}
    for name, o, l_ in (("Real NSE prices", o_r, led), ("Model prices", o_m, None)):
        m = E.metrics(o.value)
        rows[name] = {"Return a year (%)": m["CAGR"], "Worst fall (%)": m["maxDD"], "Sharpe": m["sharpe"]}
    rows["Unhedged"] = {"Return a year (%)": u.CAGR, "Worst fall (%)": u.maxDD, "Sharpe": u.sharpe}
    st.dataframe(pd.DataFrame(rows).T.style.format("{:.2f}"), width="content")
    df = pd.DataFrame({"Real NSE prices": o_r.value, "Model prices": o_m.value, "Unhedged": vals["Unhedged"]})
    st.altair_chart(line_chart(df, "Growth of 1").interactive(bind_y=False), width="stretch")
    with st.expander(f"Trades ({int((led.action == 'open').sum()) if len(led) else 0} puts bought)"):
        if len(led):
            st.dataframe(led[["date", "action", "K", "expiry", "days_left", "price", "source", "vix", "mid_model"]]
                         .rename(columns={"date": "Date", "action": "Action", "K": "Strike", "expiry": "Expiry",
                                          "days_left": "Days to expiry", "price": "Price", "source": "Price source",
                                          "vix": "VIX", "mid_model": "Model price"}), width="stretch", hide_index=True)

# ---------- 5. data and method ----------
with tabs[5]:
    lq = R["liquidity"]
    st.markdown(
        f"""**Data.** NSE F&O bhavcopies for every trading day from Jan 2020 to Apr 2026 ({R['data']['bhavcopy_days']:,}
files), downloaded with the script in the repo and filtered to Nifty options (expiry within 200 days) and futures.
One file (30 Mar 2021) could not be downloaded; positions held that day keep their previous price. Nifty and India
VIX closes are the same series as the course project, so the two backtests line up day for day.

**Which put is bought.** The listed monthly or quarterly expiry closest to the chosen length (3-month puts had
{lq['opens_days_to_expiry']['3M'][0]} to {lq['opens_days_to_expiry']['3M'][2]} days left when bought), and the strike
nearest 5% below Nifty among strikes that traded at least {R['data']['min_volume']} contracts that day (actual
strikes were {lq['opens_moneyness_range'][0]:.1%} to {lq['opens_moneyness_range'][1]:.1%} of Nifty).

**Which price is used.**
1. The option's own closing price, if at least {R['data']['min_volume']} contracts traded that day. NSE's close is the
   average price of the last 30 minutes of trading.
2. Otherwise, a price read off that day's volatility smile, built from the strikes of the same expiry that did trade
   at least {R['data']['min_volume']} contracts. A single contract can trade 50% away from its neighbours, so thin prints
   are not trusted. This also values deep in-the-money puts, which the download leaves out.
3. Bid/ask: buys pay the price plus 2%, sales receive the price minus 2%. Puts held to expiry are settled at
   their value on the Nifty close of the expiry day.

**Same as the course project.** Backtest period, portfolio, roll dates, VIX thresholds, selling rules, metrics.

**Limitations.**
- Bhavcopies give end-of-day prices, not bid/ask quotes, so the trading cost is an assumption (tested from 0% to 10%).
- Positions are a fraction of the portfolio, not whole lots. For portfolios above a few lakh rupees, rounding to
  lots changes little.
- Dividends are left out for every strategy, and the risk-free rate is a flat 6.5%.
- Six years hold one large crash (2020) and a few smaller falls, so the hedge's edge rests on few events.
"""
    )
    st.markdown(f"Code, data and the download notebook: [{REPO_URL}]({REPO_URL})")

st.divider()
st.caption(f"{AUTHOR} · Built on NSE F&O bhavcopy data. For learning and research, not investment advice.")
