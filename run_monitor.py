#!/usr/bin/env python3
"""
etf_ma_screen.py

Uses Adj Close prices with per-ticker caching.
- Cache schema: Date, AdjClose
- Incremental updates from last cached date
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import find_dotenv, load_dotenv
from SES import AmazonSES

load_dotenv(find_dotenv())

# -----------------------
# Defaults
# -----------------------
UNIVERSE_START = "2012-01-01"
MIN_AVG_VOL = 1_000_000

KEEP_PATTERNS = [
    "Index",
    "Bond",
    "Commodity",
    "Equity Sector",
    "Factor",
    "Country/Region",
    "Other",
    "Growth",
    "REIT",
    "Thematic/Innovation",
    "Leveraged/Inverse",
    "Dividend/Income",
    "Global/World",
    "Option Strategy",
    "Short",
    "Loan",
]

MA_WINS = [50, 100, 150, 200, 250, 300]


# -----------------------
# Strategy metrics
# -----------------------
def html_th(label: str) -> str:
    return (
        '<th style="border:1px solid #ddd; padding:6px 10px; '
        'background:#f5f5f5; text-align:left; white-space:nowrap;">'
        f"{label}</th>"
    )


def html_td(value: object, align: str = "right") -> str:
    return (
        '<td style="border:1px solid #ddd; padding:6px 10px; '
        f'text-align:{align}; white-space:nowrap;">{value}</td>'
    )


def build_trend_monitor_tables(
    df_filtered: pd.DataFrame, cols: list[str], top_n: int = 10
):
    """
    Grouped Trend Monitor tables by Category.

    - One table per Category
    - Sorted by Sharpe within each Category
    - Supports Fund Name, Start Date, Category, Avg-Vol
    - Rounds numeric metrics to 3 decimals
    - Formats Avg-Vol with commas
    - Email-safe HTML (no pandas class leakage)
    """

    if df_filtered is None or df_filtered.empty:
        return (
            "<p><b>Trend Monitor</b>: No rows matched filters.</p>",
            "Trend Monitor: No rows matched filters.\n",
        )

    # Ensure TimeInMarket is included if present
    cols_use = list(cols)
    if "TimeInMarket" in df_filtered.columns and "TimeInMarket" not in cols_use:
        cols_use.append("TimeInMarket")

    # Keep only columns that exist
    cols_use = [c for c in cols_use if c in df_filtered.columns]
    if not cols_use:
        return (
            "<p><b>Trend Monitor</b>: No valid columns found.</p>",
            "Trend Monitor: No valid columns found.\n",
        )

    df_disp = df_filtered[cols_use].copy()

    # ---------- Formatting ----------
    metric_cols = [
        c for c in ["Sharpe", "CAGR", "MaxDD", "TimeInMarket"] if c in df_disp.columns
    ]
    for c in metric_cols:
        df_disp[c] = pd.to_numeric(df_disp[c], errors="coerce").round(3)

    if "Avg-Vol" in df_disp.columns:
        df_disp["Avg-Vol"] = pd.to_numeric(df_disp["Avg-Vol"], errors="coerce")

    if "Start Date" in df_disp.columns:
        df_disp["Start Date"] = pd.to_datetime(
            df_disp["Start Date"], errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    # Ensure Category exists for grouping
    if "Category" not in df_disp.columns:
        df_disp["Category"] = "Uncategorized"

    # ---------- Plain text ----------
    plain_blocks = []
    for category, g in df_disp.groupby("Category", dropna=False):
        g = g.sort_values("Sharpe", ascending=False).head(top_n).copy()

        if "Avg-Vol" in g.columns:
            g["Avg-Vol"] = g["Avg-Vol"].map(
                lambda x: f"{int(x):,}" if pd.notna(x) else ""
            )

        plain_blocks.append(f"\nCategory: {category}\n" + g.to_string(index=False))

    plain_out = (
        "\nTrend Monitor (Grouped by Category)\n" + "\n".join(plain_blocks) + "\n"
    )

    # ---------- HTML ----------
    html_blocks = []

    for category, g in df_disp.groupby("Category", dropna=False):
        g = g.sort_values("Sharpe", ascending=False).head(top_n).copy()

        if "Avg-Vol" in g.columns:
            g["Avg-Vol"] = g["Avg-Vol"].map(
                lambda x: f"{int(x):,}" if pd.notna(x) else ""
            )

        if "Ticker" in g.columns:
            g["Ticker"] = (
                g["Ticker"]
                .astype(str)
                .map(
                    lambda t: (
                        f'<a clicktracking="off" '
                        f'href="https://finviz.com/quote.ashx?t={t}">{t}</a>'
                    )
                )
            )

        headers = g.columns.tolist()
        thead = "<tr>" + "".join(html_th(h) for h in headers) + "</tr>"

        left_cols = {"Ticker", "Fund Name", "Category"}
        body_rows = []

        for _, row in g.iterrows():
            tds = []
            for h in headers:
                align = "left" if h in left_cols else "right"
                tds.append(html_td(row[h], align=align))
            body_rows.append("<tr>" + "".join(tds) + "</tr>")

        table_html = (
            f"<br><b>Category: {category}</b><br>"
            '<div style="width:100%; overflow-x:auto;">'
            '<table style="width:100%; min-width:1100px; border-collapse:collapse; '
            'font-family:Arial, sans-serif; font-size:12px;">'
            f"<thead>{thead}</thead>"
            f"<tbody>{''.join(body_rows)}</tbody>"
            "</table></div>"
        )

        html_blocks.append(table_html)

    html_out = (
        "<br><b>Trend Monitor (Grouped by Category)</b><br>"
        + "".join(html_blocks)
        + "<br>"
    )

    return html_out, plain_out


def ma_strategy_metrics(price: pd.Series, ma_wins: Iterable[int]) -> pd.DataFrame:
    price = price.dropna()
    if price.empty:
        return pd.DataFrame()

    ret = price.pct_change()
    rows = []

    for w in ma_wins:
        ma = price.rolling(w).mean()
        signal = (price > ma).shift(1)

        strat_ret = (ret * signal).dropna()
        if strat_ret.empty:
            continue

        cum = (1 + strat_ret).cumprod()
        yrs = (cum.index[-1] - cum.index[0]).days / 365.25
        if yrs <= 0:
            continue

        vol = strat_ret.std() * np.sqrt(252)
        sharpe = (strat_ret.mean() * 252) / vol if vol > 0 else np.nan

        rows.append(
            {
                "MaWin": w,
                "CAGR": cum.iloc[-1] ** (1 / yrs) - 1,
                "Sharpe": sharpe,
                "Vol": vol,
                "MaxDD": (cum / cum.cummax() - 1).min(),
                "TimeInMarket": float(signal.mean()),
            }
        )

    return pd.DataFrame(rows)


# -----------------------
# Cache helpers (Adj Close)
# -----------------------
def _safe_ticker(t: str) -> str:
    return t.replace("/", "-").replace(".", "-").upper()


def cache_path(cache_dir: str, ticker: str) -> str:
    return os.path.join(cache_dir, f"{_safe_ticker(ticker)}.csv")


def read_cache(path: str) -> pd.Series:
    df = pd.read_csv(path, parse_dates=["Date"])
    s = pd.Series(df["AdjClose"].values, index=df["Date"])
    s.name = "AdjClose"
    return s[~s.index.duplicated(keep="last")].sort_index()


def write_cache(path: str, s: pd.Series) -> None:
    out = pd.DataFrame({"Date": s.index, "AdjClose": s.values})
    out.to_csv(path, index=False)


def ensure_adjclose_cache(
    ticker: str, cache_dir: str, start: str, end: str
) -> pd.Series:
    os.makedirs(cache_dir, exist_ok=True)
    path = cache_path(cache_dir, ticker)
    end_ts = pd.Timestamp(end)

    if os.path.exists(path):
        s_old = read_cache(path)
        last_dt = s_old.index.max()

        if last_dt >= end_ts:
            return s_old

        dl_start = last_dt + pd.Timedelta(days=1)
    else:
        s_old = pd.Series(dtype=float)
        dl_start = pd.Timestamp(start)

    if dl_start > end_ts:
        return s_old

    dl = yf.download(
        ticker,
        start=dl_start.strftime("%Y-%m-%d"),
        end=end_ts.strftime("%Y-%m-%d"),
        auto_adjust=False,
        progress=False,
    )
    dl.columns = dl.columns.droplevel(1)
    if dl is None or dl.empty or "Adj Close" not in dl:
        return s_old

    s_new = dl["Adj Close"].dropna()
    parts = [s for s in (s_old, s_new) if not s.empty]
    if not parts:
        return pd.Series(dtype=float)

    s_all = pd.concat(parts).sort_index()

    s_all = s_all[~s_all.index.duplicated(keep="last")]

    write_cache(path, s_all)
    return s_all


def load_px_from_cache(tickers: list[str], cache_dir: str) -> pd.DataFrame:
    series = []
    for t in tickers:
        path = cache_path(cache_dir, t)
        if not os.path.exists(path):
            continue
        s = read_cache(path)
        if not s.empty:
            series.append(s.rename(t))

    if not series:
        return pd.DataFrame()

    px = pd.concat(series, axis=1).sort_index()
    return px.ffill()


# -----------------------
# Universe filter
# -----------------------
def filter_universe(df: pd.DataFrame) -> pd.DataFrame:
    mask = (
        df["Category"]
        .astype(str)
        .str.contains("|".join(KEEP_PATTERNS), case=False, na=False)
    )

    df = df[df["Avg-Vol"] >= MIN_AVG_VOL].copy()
    df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")

    univ_start = pd.Timestamp(UNIVERSE_START)
    df = df[df["Start Date"].notna() & (df["Start Date"] <= univ_start) & mask]

    df["Ticker"] = df["Ticker"].str.upper().str.strip()
    return df.drop_duplicates("Ticker")


# -----------------------
# Main
# -----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--universe_csv", default="etf-list.csv")
    ap.add_argument("--cache_dir", default="./cache_prices")
    ap.add_argument("--end", default=None)
    ap.add_argument("--out_filtered", default=None)
    ap.add_argument(
        "--email",
        action="store_true",
        help="Send email with trend monitor results",
    )
    args = ap.parse_args()

    end = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    df_univ = pd.read_csv(args.universe_csv)
    df_univ_filt = filter_universe(df_univ)

    tickers = df_univ_filt["Ticker"].tolist()
    print(f"Tickers after filters: {len(tickers)}")

    # Update cache
    for i, t in enumerate(tickers, 1):
        s = ensure_adjclose_cache(t, args.cache_dir, UNIVERSE_START, end)
        if not s.empty:
            print(f"[{i}/{len(tickers)}] {t}: cached → {s.index.max().date()}")

    # Build px from cache
    px = load_px_from_cache(tickers, args.cache_dir)
    if px.empty:
        print("No price data loaded.")
        return

    dfs = []
    for t in px.columns:
        stats = ma_strategy_metrics(px[t], MA_WINS)
        if not stats.empty:
            dfs.append(stats.assign(Ticker=t))

    df_results = pd.concat(dfs, ignore_index=True)
    df_results = df_results.merge(df_univ, on="Ticker", how="left")

    df_filtered = df_results[
        (df_results["Sharpe"] >= float(os.getenv("MIN_SHARPE", 0.6)))
        & (df_results["Sharpe"] <= float(os.getenv("MAX_SHARPE", 2.0)))
        & (df_results["MaxDD"] >= float(os.getenv("MAX_DD", -0.25)))
    ].copy()

    # Normalize ticker for safe merging
    df_univ["Ticker"] = df_univ["Ticker"].astype(str).str.upper().str.strip()
    df_filtered["Ticker"] = df_filtered["Ticker"].astype(str).str.upper().str.strip()

    # Merge universe metadata onto filtered results
    meta_cols = ["Ticker", "Fund Name", "Start Date", "Category", "Avg-Vol"]

    df_filtered = df_filtered.merge(
        df_univ,
        on="Ticker",
        how="left",
        suffixes=("", "_univ"),
    )

    # Drop the unwanted duplicates
    df_filtered = df_filtered.drop(
        columns=[c for c in df_filtered.columns if c.endswith("_univ")]
    )

    cols = [
        "Ticker",
        "Fund Name",
        "Category",
        "Avg-Vol",
        "Start Date",
        "MaWin",
        "Sharpe",
        "CAGR",
        "MaxDD",
        "TimeInMarket",
    ]
    out = (
        df_filtered[cols]
        .assign(
            Sharpe=lambda x: x["Sharpe"].round(3),
            CAGR=lambda x: x["CAGR"].round(3),
            MaxDD=lambda x: x["MaxDD"].round(3),
            TimeInMarket=lambda x: x["TimeInMarket"].round(3),
        )
        .sort_values("Sharpe", ascending=False)
        .head(50)
    )

    print(out.to_string(index=False))

    if args.out_filtered:
        df_filtered[cols].to_csv(args.out_filtered, index=False)

    if args.email:

        message_body_html, message_body_plain = build_trend_monitor_tables(
            df_filtered, cols=cols, top_n=10
        )

        TO_ADDRESSES = os.getenv("TO_ADDRESSES", "").split(",")
        FROM_ADDRESS = os.getenv("FROM_ADDRESS", "")
        ses = AmazonSES(
            region=os.environ.get("AWS_SES_REGION_NAME"),
            access_key=os.environ.get("AWS_SES_ACCESS_KEY_ID"),
            secret_key=os.environ.get("AWS_SES_SECRET_ACCESS_KEY"),
            from_address=os.environ.get("FROM_ADDRESS"),
        )

        today_str = datetime.now(timezone.utc).strftime("%B %d, %Y")
        subject = f"ETF Trend Monitor – Week of {today_str}"

        for to_address in TO_ADDRESSES:
            ses.send_html_email(
                to_address=to_address, subject=subject, content=message_body_html
            )


if __name__ == "__main__":
    main()
