"""Put-hedge backtest on real NSE option prices.

Same portfolio and trading rules as the original FDRM VIX project (fdrm_vix_analysis.py, run()):
100% Nifty, options funded by selling Nifty units, all option cash reinvested in Nifty the same day,
roll dates = first trading day of each month, VIX < 13 buy rule, half sold at VIX 22, rest when VIX
is 20% below its peak.

What changes:
* Contracts are real listed Nifty puts (monthly/quarterly expiry closest to the tenor, strike
  nearest the target among liquid strikes), held to their real expiry and settled at the Nifty close.
* pricing="real": premiums come from NSE closing prices (see chain.Chain.fair), with a bid/ask
  haircut of `cost` (buy at price x (1 + cost), sell at price x (1 - cost)).
* pricing="model": the original Black-Scholes price (India VIX + 0.6 vol pts per 1% OTM, +/- 0.5
  vol pts spread) of the same real contracts, to isolate the effect of pricing.
"""
import math

import numpy as np
import pandas as pd

from chain import Q, R, bs_put

SKEW_PER_PCT = 0.6
HALF_SPREAD = 0.5
COST = 0.02            # real prices: bid/ask haircut, 2% of the premium each way
LOW_VIX = 13.0
SPIKE_1, SPIKE_2 = 22.0, 30.0
TRAIL = 0.20
TENOR = 3
PUT_MONEYNESS = 0.95
SPREAD_SHORT = 0.85
TRADING_DAYS = 252


def run(strategy, bt, chain, pricing="real", cost=COST, low=LOW_VIX, spike1=SPIKE_1, spike2=SPIKE_2,
        tenor=TENOR, moneyness=PUT_MONEYNESS, skew=SKEW_PER_PCT, half_spread=HALF_SPREAD, tranche=1.0,
        stagger=False, trail=TRAIL):
    """bt: DataFrame with Close, VIX, regime. Returns (daily DataFrame, trade ledger, stats dict)."""
    dates = bt.index
    first = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).transform("min")
    roll_dates = set(first.unique())
    close = bt.Close
    units = 1.0 / close.iloc[0]
    pos, trades, rec = [], [], []
    stats = dict(marks=0, skipped_opens=0, traded=0, smile=0, thin=0, stale=0)
    tid = 0

    def model_mid(p, d, S, v, side=0):
        T = (p["exp"] - d).days / 365
        otm = max(0.0, (1 - p["K"] / S) * 100)
        sig = max((v + skew * otm + side * half_spread) / 100, 0.01)
        return float(bs_put(S, p["K"], T, sig))

    def real_mid(p, d, S):
        q = chain.fair(d, p["exp"], p["K"], S)
        if q is None:   # no data that day: last price, but never below discounted intrinsic value
            T = (p["exp"] - d).days / 365
            floor = math.exp(-R * T) * max(p["K"] - S * math.exp((R - Q) * T), 0.0)
            return max(p.get("last_px", 0.0), floor), "stale"
        p["last_px"], p["exp"] = q[0], q[2]
        return q[0], q[1]

    def price(p, d, S, v, side=0):
        """side +1 = we buy, -1 = we sell, 0 = mid (for marking)."""
        if p["exp"] <= d:   # expired: settle at intrinsic value on the expiry-day close
            return max(p["K"] - close.asof(p["exp"]), 0.0), "expiry"
        if pricing == "model":
            return model_mid(p, d, S, v, side), "model"
        px, src = real_mid(p, d, S)
        return px * (1 + side * cost), src

    def log(d, p, action, q, px, src, cash, v, regime, S):
        m = (real_mid(p, d, S)[0] if p["exp"] > d else np.nan)
        trades.append(dict(date=d, trade_id=p["tid"], action=action, K=p["K"], expiry=p["exp"],
                           days_left=(p["exp"] - d).days, moneyness=p["K"] / S, qty=q * p["sign"], price=px,
                           source=src, cash=cash, vix=v, regime=regime, open_regime=p["open_regime"],
                           mid_real=m, mid_model=model_mid(p, d, S, v) if p["exp"] > d else np.nan,
                           volume=p.get("volume", np.nan) if action == "open" else np.nan,
                           oi=p.get("oi", np.nan) if action == "open" else np.nan))

    def close_pos(p, d, S, v, frac, reason, regime):
        nonlocal units
        q = p["qty"] * frac
        px, src = price(p, d, S, v, side=-1 if p["sign"] > 0 else +1)
        cash = p["sign"] * q * px
        units += cash / S
        p["qty"] -= q
        log(d, p, reason, q, px, src, cash, v, regime, S)

    def open_pos(d, S, v, months, mny, sign, regime, frac=1.0):
        nonlocal units, tid
        sel = chain.select(d, S, months, mny)
        if sel is None:
            stats["skipped_opens"] += 1
            return
        V = units * S + sum(price(p, d, S, v)[0] * p["qty"] * p["sign"] for p in pos)
        p = dict(K=sel["K"], exp=sel["exp"], qty=frac * V / S, sign=sign, tid=tid, open_regime=regime,
                 sold1=False, volume=sel["volume"], oi=sel["oi"])
        px, src = price(p, d, S, v, side=+1 if sign > 0 else -1)
        cash = -sign * p["qty"] * px
        units += cash / S
        pos.append(p)
        log(d, p, "open", p["qty"], px, src, cash, v, regime, S)
        tid += 1

    for d, row in bt.iterrows():
        S, v, regime = row.Close, row.VIX, str(row.regime)
        for p in pos:
            if p["qty"] > 0 and p["exp"] <= d:
                close_pos(p, d, S, v, 1.0, "expiry", regime)
        pos = [p for p in pos if p["qty"] > 1e-12]
        is_roll = d in roll_dates
        if strategy == "always_put" and is_roll:
            open_pos(d, S, v, 1, moneyness, +1, regime)
        elif strategy == "always_spread" and is_roll:
            open_pos(d, S, v, 1, moneyness, +1, regime)
            open_pos(d, S, v, 1, SPREAD_SHORT, -1, regime)
        elif strategy.startswith("timed"):
            if strategy in ("timed_ladder", "timed_all", "timed_trail", "timed_hybrid"):
                for p in pos:
                    if strategy in ("timed_trail", "timed_hybrid"):
                        if v >= spike1:
                            p["peak"] = max(p.get("peak", 0.0), v)
                        if strategy == "timed_hybrid" and not p["sold1"] and v >= spike1:
                            close_pos(p, d, S, v, 0.5, "monetise_1", regime); p["sold1"] = True
                        if p.get("peak") and v <= (1 - trail) * p["peak"] and p["qty"] > 0:
                            close_pos(p, d, S, v, 1.0, "monetise_trail", regime)
                    elif strategy == "timed_ladder":
                        if not p["sold1"] and v >= spike1:
                            close_pos(p, d, S, v, 0.5, "monetise_1", regime); p["sold1"] = True
                        if p["sold1"] and v >= spike2 and p["qty"] > 0:
                            close_pos(p, d, S, v, 1.0, "monetise_2", regime)
                    elif v >= spike1:
                        close_pos(p, d, S, v, 1.0, "monetise", regime)
                pos = [p for p in pos if p["qty"] > 1e-12]
            if is_roll and v < low:
                if not stagger:
                    for p in pos:
                        close_pos(p, d, S, v, 1.0, "roll_sell", regime)
                    pos = []
                open_pos(d, S, v, tenor, moneyness, +1, regime, frac=tranche)
        hedge = 0.0
        for p in pos:
            px, src = price(p, d, S, v)
            hedge += px * p["qty"] * p["sign"]
            stats["marks"] += 1
            if src in stats:
                stats[src] += 1
        rec.append(dict(date=d, equity=units * S, hedge=hedge, value=units * S + hedge,
                        notional_hedged=sum(p["qty"] * p["sign"] for p in pos if p["sign"] > 0) * S))
    return pd.DataFrame(rec).set_index("date"), pd.DataFrame(trades), stats


def metrics(val, rf=R):
    r = val.pct_change().dropna()
    yrs = (val.index[-1] - val.index[0]).days / 365.25
    cagr = (val.iloc[-1] / val.iloc[0]) ** (1 / yrs) - 1
    vol = r.std() * math.sqrt(TRADING_DAYS)
    dd = val / val.cummax() - 1
    downside = r[r < 0].std() * math.sqrt(TRADING_DAYS)
    return dict(final=val.iloc[-1], CAGR=cagr * 100, vol=vol * 100, sharpe=(cagr - rf) / vol,
                sortino=(cagr - rf) / downside, maxDD=dd.min() * 100, calmar=cagr / abs(dd.min()),
                worst_day=r.min() * 100, worst_21d=(val / val.shift(21) - 1).min() * 100)


def hedge_cost(led, val, yrs):
    """Premium paid, recovered and net cost, each as % of portfolio a year (as in the original)."""
    if len(led) == 0:
        return dict(premium_paid_pa=0.0, recovered_pa=0.0, net_hedge_cost_pa=0.0)
    pct = led.cash.values / val.reindex(led.date).values * 100
    paid, got = -pct[pct < 0].sum(), pct[pct > 0].sum()
    return dict(premium_paid_pa=paid / yrs, recovered_pa=got / yrs, net_hedge_cost_pa=(paid - got) / yrs)
