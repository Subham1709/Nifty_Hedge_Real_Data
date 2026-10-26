"""
Download NSE F&O daily bhavcopies and keep only the NIFTY index options and futures.

Two file formats are handled:
  * old format (up to 5 Jul 2024):  fo02JAN2020bhav.csv.zip
  * UDiFF format (from 8 Jul 2024): BhavCopy_NSE_FO_0_0_0_20240708_F_0000.csv.zip

Output (in OUT_DIR):
  days/2020-01-02.parquet ...        one small file per trading day (lets a rerun resume)
  nifty_fo_2020.parquet ... 2026     one file per year: these are the files to upload
  download_log.csv                   what happened on every weekday (ok / holiday / failed)

Run locally:  pip install pandas pyarrow requests   then   python nse_fo_download.py
"""
import datetime as dt
import io
import random
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------- settings
START = "2020-01-01"
END = "2026-04-30"
SYMBOL = "NIFTY"            # exact match, so FINNIFTY / NIFTYNXT50 are dropped
OUT_DIR = Path("nse_fo_nifty")
WORKERS = 3                 # parallel downloads; lower to 1 if NSE starts refusing
MAX_TRIES = 5               # retries per URL, with growing waits
MAX_DAYS_TO_EXPIRY = 200    # covers monthly and quarterly expiries out to ~6 months
PUT_BAND = (0.65, 1.10)     # keep puts with strike between 65% and 110% of the near future
CALL_BAND = (0.90, 1.15)    # keep calls near the money (for implied vol / parity checks)
DROP_DEAD = True            # drop strikes with zero open interest and zero volume

UDIFF_START = dt.date(2024, 7, 8)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/all-reports-derivatives",
}

# ---------------------------------------------------------------- download
_local = threading.local()


def _warm(s):
    # NSE sets cookies on its home page; the archive host usually works without them
    try:
        s.get("https://www.nseindia.com", timeout=15)
    except requests.RequestException:
        pass


def _session():
    if not hasattr(_local, "s"):
        _local.s = requests.Session()
        _local.s.headers.update(HEADERS)
        _warm(_local.s)
    return _local.s


def urls_for(d):
    Y, MON, DD = d.strftime("%Y"), d.strftime("%b").upper(), d.strftime("%d")
    old = f"content/historical/DERIVATIVES/{Y}/{MON}/fo{DD}{MON}{Y}bhav.csv.zip"
    new = f"content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    if d >= UDIFF_START:
        return [f"https://nsearchives.nseindia.com/{new}"]
    return [f"https://nsearchives.nseindia.com/{old}",
            f"https://archives.nseindia.com/{old}",
            f"https://nsearchives.nseindia.com/{new}"]


def fetch(d):
    """Return (zip bytes, url) or (None, reason). Every URL giving 404 means no file (holiday)."""
    s = _session()
    reasons = []
    for url in urls_for(d):
        for attempt in range(MAX_TRIES):
            try:
                r = s.get(url, timeout=45)
            except requests.RequestException as e:
                reason = type(e).__name__
            else:
                if r.status_code == 200 and r.content[:2] == b"PK":
                    return r.content, url
                if r.status_code == 404:
                    reason = "404"
                    break
                reason = f"HTTP {r.status_code}"
                if r.status_code in (401, 403):
                    _warm(s)
            time.sleep(min(60, 2 ** attempt) + random.random())
        reasons.append(reason)
    return None, ("holiday" if all(x == "404" for x in reasons) else "; ".join(reasons))


# ---------------------------------------------------------------- parsing
def _dates(s):
    s = s.astype(str).str.strip()
    best = None
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%d-%b-%y"):
        x = pd.to_datetime(s, format=fmt, errors="coerce")
        if best is None or x.notna().sum() > best.notna().sum():
            best = x
    return best


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _str(s):
    return s.astype(str).str.strip()


def parse(content):
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        name = [n for n in z.namelist() if n.lower().endswith(".csv")][0]
        raw = pd.read_csv(z.open(name), low_memory=False)
    raw.columns = [c.strip() for c in raw.columns]
    nan = np.nan
    if "INSTRUMENT" in raw.columns:  # old format
        df = pd.DataFrame({
            "date": _dates(raw["TIMESTAMP"]), "fmt": "old",
            "instrument": _str(raw["INSTRUMENT"]), "symbol": _str(raw["SYMBOL"]),
            "expiry": _dates(raw["EXPIRY_DT"]), "strike": _num(raw["STRIKE_PR"]),
            "opt": _str(raw["OPTION_TYP"]),
            "open": _num(raw["OPEN"]), "high": _num(raw["HIGH"]), "low": _num(raw["LOW"]),
            "close": _num(raw["CLOSE"]), "settle": _num(raw["SETTLE_PR"]), "last": nan, "prev_close": nan,
            "volume_raw": _num(raw["CONTRACTS"]), "turnover_raw": _num(raw["VAL_INLAKH"]), "n_trades": nan,
            "oi": _num(raw["OPEN_INT"]), "chg_oi": _num(raw["CHG_IN_OI"]),
            "underlying": nan, "lot": nan,
        })
    elif "FinInstrmTp" in raw.columns:  # UDiFF format
        g = lambda c: raw[c] if c in raw.columns else pd.Series(nan, index=raw.index)
        df = pd.DataFrame({
            "date": _dates(raw["TradDt"]), "fmt": "udiff",
            "instrument": _str(raw["FinInstrmTp"]).map({"IDO": "OPTIDX", "IDF": "FUTIDX",
                                                        "STO": "OPTSTK", "STF": "FUTSTK"}),
            "symbol": _str(raw["TckrSymb"]), "expiry": _dates(raw["XpryDt"]),
            "strike": _num(g("StrkPric")), "opt": _str(g("OptnTp")).replace({"nan": "XX", "": "XX"}).fillna("XX"),
            "open": _num(g("OpnPric")), "high": _num(g("HghPric")), "low": _num(g("LwPric")),
            "close": _num(g("ClsPric")), "settle": _num(g("SttlmPric")),
            "last": _num(g("LastPric")), "prev_close": _num(g("PrvsClsgPric")),
            "volume_raw": _num(g("TtlTradgVol")), "turnover_raw": _num(g("TtlTrfVal")),
            "n_trades": _num(g("TtlNbOfTxsExctd")),
            "oi": _num(g("OpnIntrst")), "chg_oi": _num(g("ChngInOpnIntrst")),
            "underlying": _num(g("UndrlygPric")), "lot": _num(g("NewBrdLotQty")),
        })
    else:
        raise ValueError(f"unknown columns: {list(raw.columns)[:12]}")

    df = df[(df.symbol == SYMBOL) & df.instrument.isin(["OPTIDX", "FUTIDX"])].copy()
    fut = df[df.instrument == "FUTIDX"].sort_values("expiry")
    if fut.empty:
        raise ValueError("no NIFTY futures rows")
    near = fut.close.iloc[0] if fut.close.iloc[0] > 0 else fut.settle.iloc[0]
    df["near_fut"] = near

    opt = df.instrument == "OPTIDX"
    dte = (df.expiry - df.date).dt.days
    m = df.strike / near
    keep = ~opt | (
        (dte <= MAX_DAYS_TO_EXPIRY)
        & (((df.opt == "PE") & m.between(*PUT_BAND)) | ((df.opt == "CE") & m.between(*CALL_BAND)))
    )
    if DROP_DEAD:
        keep &= ~opt | (df.oi.fillna(0) > 0) | (df.volume_raw.fillna(0) > 0)
    return df[keep].reset_index(drop=True)


# ---------------------------------------------------------------- driver
def process_day(d, days_dir, raw_dir):
    out = days_dir / f"{d:%Y-%m-%d}.parquet"
    content, info = fetch(d)
    if content is None:
        return dict(date=d, status="holiday" if info == "holiday" else "failed", info=info, rows=0)
    try:
        df = parse(content)
    except Exception as e:  # keep the raw zip so it can be re-parsed later
        (raw_dir / f"{d:%Y-%m-%d}.zip").write_bytes(content)
        return dict(date=d, status="parse_error", info=f"{type(e).__name__}: {e}"[:300], rows=0)
    df.to_parquet(out, index=False)
    return dict(date=d, status="ok", info=info, rows=len(df))


def run(start=START, end=END, out_dir=OUT_DIR, workers=WORKERS):
    out_dir = Path(out_dir)
    days_dir, raw_dir = out_dir / "days", out_dir / "raw_unparsed"
    days_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "download_log.csv"
    log = pd.read_csv(log_path, parse_dates=["date"]) if log_path.exists() else pd.DataFrame(
        columns=["date", "status", "info", "rows"])
    done = {p.stem for p in days_dir.glob("*.parquet")}
    done |= {f"{x:%Y-%m-%d}" for x in log.loc[log.status == "holiday", "date"]}
    todo = [d.date() for d in pd.bdate_range(start, end) if f"{d:%Y-%m-%d}" not in done]
    print(f"{len(done)} days already done, {len(todo)} to fetch")

    rows, t0 = [], time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(process_day, d, days_dir, raw_dir): d for d in todo}
        for i, f in enumerate(as_completed(futs), 1):
            rows.append(f.result())
            if i % 25 == 0 or i == len(todo):
                st = pd.Series([r["status"] for r in rows]).value_counts().to_dict()
                print(f"{i}/{len(todo)}  {time.time() - t0:.0f}s  {st}", flush=True)
            if i % 100 == 0:  # save the log as we go
                _write_log(log, rows, log_path)
    log = _write_log(log, rows, log_path)
    failed = log[log.status.isin(["failed", "parse_error"])]
    if len(failed):
        print(f"\n{len(failed)} days failed. Run again to retry them. First few:")
        print(failed.head(10).to_string(index=False))
    return log


def _write_log(log, rows, log_path):
    new = pd.DataFrame(rows)
    if len(new):
        new["date"] = pd.to_datetime(new["date"])
        log = pd.concat([log[~log.date.isin(new.date)], new]).sort_values("date")
    log.to_csv(log_path, index=False)
    return log


def combine(out_dir=OUT_DIR):
    out_dir = Path(out_dir)
    files = sorted((out_dir / "days").glob("*.parquet"))
    by_year = {}
    for p in files:
        by_year.setdefault(p.stem[:4], []).append(p)
    written = []
    for y, ps in sorted(by_year.items()):
        df = pd.concat([pd.read_parquet(p) for p in ps], ignore_index=True)
        path = out_dir / f"nifty_fo_{y}.parquet"
        df.to_parquet(path, index=False, compression="zstd")
        written.append(path)
        print(f"{path.name}: {df.date.nunique()} days, {len(df):,} rows, {path.stat().st_size / 1e6:.1f} MB")
    return written


if __name__ == "__main__":
    run()
    combine()
