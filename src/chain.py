"""Real Nifty put prices from NSE F&O bhavcopies.

Price of a put on a day, in order of preference:
1. "traded": its own closing price, if at least MIN_VOLUME contracts traded that day.
   (NSE's close is the average price of the last 30 minutes of trading.)
2. "smile": otherwise, read off that day's implied-volatility smile for the same expiry, built
   from the strikes that did trade at least MIN_VOLUME contracts. Thinly traded strikes print
   unreliable prices (a single contract can trade 50% away from its neighbours), and deep
   in-the-money puts are not in the downloaded data, so this is how they are valued.
3. "thin": its own close (or NSE settlement price if untraded) when no smile can be built.
4. "stale": no data at all that day (one missing file, 30 Mar 2021): the engine uses the last
   price, floored at intrinsic value.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq

R, Q = 0.065, 0.013   # same risk-free rate and dividend yield as the original project
MIN_VOLUME = 50       # contracts traded that day for a strike's own price to count as reliable


def _ncdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def load_bhavcopy(folder):
    files = sorted(Path(folder).glob("nifty_fo_*.parquet"))
    df = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    df = df[df.symbol == "NIFTY"]
    return df


class Chain:
    def __init__(self, bhav, min_volume=MIN_VOLUME):
        self.min_volume = min_volume
        puts = bhav[(bhav.instrument == "OPTIDX") & (bhav.opt == "PE")].copy()
        puts["traded"] = puts.volume_raw.fillna(0) > 0
        puts["px"] = np.where(puts.traded, puts.close, puts.settle)
        puts = puts[puts.px > 0]
        puts = puts[["date", "expiry", "strike", "px", "traded", "volume_raw", "oi"]]
        self.puts = puts
        self.dates = set(puts.date.unique())
        self._row = dict(zip(zip(puts.date, puts.expiry, puts.strike),
                             zip(puts.px, puts.volume_raw.fillna(0))))
        self._exps = puts.groupby("date").expiry.unique().map(lambda a: sorted(a)).to_dict()
        # liquid strikes per (date, expiry), for the smile
        liq = puts[puts.volume_raw >= min_volume]
        self._liquid = {k: (g.strike.values, g.px.values) for k, g in liq.groupby(["date", "expiry"])}
        # the last listed expiry in each calendar month (on that day) is the monthly/quarterly contract
        by_date = {}
        for d, g in puts.groupby("date"):
            e = pd.Series(g.expiry.unique())
            by_date[d] = g[g.expiry.isin(set(e.groupby([e.dt.year, e.dt.month]).max()))]
        self._by_date = by_date
        fut = bhav[bhav.instrument == "FUTIDX"]
        self._fut = dict(zip(zip(fut.date, fut.expiry), fut.close))
        self._smiles = {}

    # ---------------------------------------------------------------- lookups
    def _listed_expiry(self, d, exp):
        """NSE moved some expiries (holidays; Thursday to Tuesday from Sep 2025). Our contracts are
        always monthly ones, so a relabelled contract is the last listed expiry in the same month."""
        exps = self._exps.get(d)
        if exps is None:
            return None
        if exp in exps:
            return exp
        same = [e for e in exps if e.year == exp.year and e.month == exp.month]
        return max(same) if same else None

    def smile(self, d, exp, S):
        """(strikes, implied vols) from the liquid strikes of one expiry on one day, or None."""
        key = (d, exp)
        if key not in self._smiles:
            out = None
            if key in self._liquid:
                Ks, pxs = self._liquid[key]
                T = (exp - d).days / 365
                F = self.forward(d, exp, S)
                iv = np.array([implied_vol(p, F, k, T) for k, p in zip(Ks, pxs)])
                ok = ~np.isnan(iv)
                if ok.sum() >= 2:
                    order = np.argsort(Ks[ok])
                    out = (Ks[ok][order], iv[ok][order])
            self._smiles[key] = out
        return self._smiles[key]

    def fair(self, d, exp, K, S):
        """(price, source, expiry label used) for one put, or None if there is no data that day."""
        exp = self._listed_expiry(d, exp)
        if exp is None:
            return None
        row = self._row.get((d, exp, K))
        if row and row[1] >= self.min_volume:
            return row[0], "traded", exp
        sm = self.smile(d, exp, S)
        if sm is not None:
            T = (exp - d).days / 365
            vol = float(np.interp(K, sm[0], sm[1]))      # flat beyond the last liquid strike
            return black76_put(self.forward(d, exp, S), K, T, vol), "smile", exp
        if row:
            return row[0], "thin", exp
        return None

    def forward(self, d, exp, S):
        f = self._fut.get((d, exp))
        if f and f > 0:
            return f
        T = max((exp - d).days, 0) / 365
        return S * math.exp((R - Q) * T)

    def select(self, d, S, months, moneyness, min_days=7, band=0.025):
        """Pick the real contract for a 'months'-month put at 'moneyness' x spot, bought on day d.

        Expiry: the listed monthly/quarterly expiry closest to `months` x 30.4 days.
        Strike: the strike nearest moneyness x spot among those within +/- 2.5% of the target that
        traded at least MIN_VOLUME contracts; if there is none, the nearest listed strike.
        Returns dict(K, exp, volume, oi) or None.
        """
        day = self._by_date.get(d)
        if day is None:
            return None
        day = day[(day.expiry - d).dt.days >= min_days]
        if day.empty:
            return None
        e = pd.Series(day.expiry.unique())
        exp = e.iloc[((e - d).dt.days - months * 30.4).abs().argmin()]
        c = day[day.expiry == exp]
        liquid = c[(c.volume_raw >= self.min_volume) & ((c.strike / S - moneyness).abs() <= band)]
        pool = liquid if len(liquid) else c
        row = pool.iloc[(pool.strike - moneyness * S).abs().argmin()]
        return dict(K=float(row.strike), exp=exp, volume=float(row.volume_raw), oi=float(row.oi))


# ---------------------------------------------------------------- Black-Scholes
def bs_put(S, K, T, sigma, r=R, q=Q):
    T, sigma = max(T, 1e-8), max(sigma, 1e-4)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * math.exp(-q * T) * _ncdf(-d1)


def black76_put(F, K, T, sigma, r=R):
    T, sigma = max(T, 1e-8), max(sigma, 1e-4)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return math.exp(-r * T) * (K * _ncdf(-d2) - F * _ncdf(-d1))


def implied_vol(px, F, K, T, r=R):
    """Implied volatility of a put from its price, using the futures price as the forward."""
    lo_px = math.exp(-r * T) * max(K - F, 0.0)
    if T <= 0 or px <= lo_px + 1e-6:
        return np.nan
    f = lambda s: black76_put(F, K, T, s, r) - px
    try:
        return brentq(f, 1e-3, 3.0)
    except ValueError:
        return np.nan
