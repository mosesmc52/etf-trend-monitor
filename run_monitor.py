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


def getenv_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default

    value = value.strip()
    if not value:
        return default

    return float(value)


def getenv_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default

    value = value.strip().lower()
    if not value:
        return default

    return value in {"1", "true", "yes", "y", "on"}

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
    "Dividend/Income",
    "Global/World",
    "Option Strategy",
    "Loan",
]

MA_WINS = [50, 100, 150, 200, 250, 300]
ROBUST_OUTPUT_COLS = [
    "Ticker",
    "Fund Name",
    "Category",
    "Avg-Vol",
    "Start Date",
    "RobustScore",
    "SharpeMedian",
    "SharpeMin",
    "SharpeRange",
    "CAGRMedian",
    "MaxDDWorst",
    "ProfitableWindows",
    "AvgTimeInMarket",
    "BH_Sharpe",
    "BH_CAGR",
    "BH_MaxDD",
    "SharpeEdge",
    "DrawdownReduction",
    "AboveMA",
    "DistanceFromMA",
    "MA_1M_Slope",
    "MA_3M_Slope",
    "Momentum_3M",
    "Momentum_6M",
]


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
    df_filtered: pd.DataFrame, cols: list[str], top_n: int = 10, sort_by: str = "Sharpe"
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
        c
        for c in [
            "Sharpe",
            "CAGR",
            "MaxDD",
            "TimeInMarket",
            "RobustScore",
            "SharpeMedian",
            "SharpeMin",
            "SharpeRange",
            "CAGRMedian",
            "MaxDDWorst",
            "ProfitableWindows",
            "AvgTimeInMarket",
            "BH_Sharpe",
            "BH_CAGR",
            "BH_MaxDD",
            "SharpeEdge",
            "DrawdownReduction",
            "DistanceFromMA",
            "MA_1M_Slope",
            "MA_3M_Slope",
            "Momentum_3M",
            "Momentum_6M",
        ]
        if c in df_disp.columns
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
        if sort_by in g.columns:
            g = g.sort_values(sort_by, ascending=False)
        g = g.head(top_n).copy()

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
        if sort_by in g.columns:
            g = g.sort_values(sort_by, ascending=False)
        g = g.head(top_n).copy()

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


def _annualized_return_stats(ret: pd.Series) -> dict:
    ret = pd.to_numeric(ret, errors="coerce").dropna()
    if ret.empty:
        return {"CAGR": np.nan, "Sharpe": np.nan, "Vol": np.nan, "MaxDD": np.nan}

    cum = (1 + ret).cumprod()
    yrs = (cum.index[-1] - cum.index[0]).days / 365.25
    if yrs <= 0:
        return {"CAGR": np.nan, "Sharpe": np.nan, "Vol": np.nan, "MaxDD": np.nan}

    vol = ret.std() * np.sqrt(252)
    sharpe = (ret.mean() * 252) / vol if pd.notna(vol) and vol > 0 else np.nan
    return {
        "CAGR": cum.iloc[-1] ** (1 / yrs) - 1,
        "Sharpe": sharpe,
        "Vol": vol,
        "MaxDD": (cum / cum.cummax() - 1).min(),
    }


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

        perf = _annualized_return_stats(strat_ret)
        if pd.isna(perf["CAGR"]):
            continue

        rows.append(
            {
                "MaWin": w,
                "CAGR": perf["CAGR"],
                "Sharpe": perf["Sharpe"],
                "Vol": perf["Vol"],
                "MaxDD": perf["MaxDD"],
                "TimeInMarket": float(signal.mean()),
            }
        )

    return pd.DataFrame(rows)


def buy_hold_metrics(price: pd.Series) -> dict:
    price = pd.to_numeric(price, errors="coerce").dropna()
    if len(price) < 2:
        return {
            "BH_CAGR": np.nan,
            "BH_Sharpe": np.nan,
            "BH_MaxDD": np.nan,
            "BH_Vol": np.nan,
        }

    perf = _annualized_return_stats(price.pct_change().dropna())
    return {
        "BH_CAGR": perf["CAGR"],
        "BH_Sharpe": perf["Sharpe"],
        "BH_MaxDD": perf["MaxDD"],
        "BH_Vol": perf["Vol"],
    }


def current_trend_features(price: pd.Series, ma_window: int = 200) -> dict:
    price = pd.to_numeric(price, errors="coerce").dropna()
    out = {
        "AboveMA": np.nan,
        "DistanceFromMA": np.nan,
        "MA_1M_Slope": np.nan,
        "MA_3M_Slope": np.nan,
        "Momentum_3M": np.nan,
        "Momentum_6M": np.nan,
    }
    if price.empty or ma_window <= 0:
        return out

    ma = price.rolling(ma_window).mean().dropna()
    if ma.empty:
        return out

    latest_price = price.iloc[-1]
    latest_ma = ma.iloc[-1]
    if pd.notna(latest_price) and pd.notna(latest_ma) and latest_ma != 0:
        out["AboveMA"] = bool(latest_price > latest_ma)
        out["DistanceFromMA"] = latest_price / latest_ma - 1

    if len(ma) > 21:
        prev_1m = ma.iloc[-22]
        if pd.notna(prev_1m) and prev_1m != 0:
            out["MA_1M_Slope"] = latest_ma / prev_1m - 1

    if len(ma) > 63:
        prev_3m = ma.iloc[-64]
        if pd.notna(prev_3m) and prev_3m != 0:
            out["MA_3M_Slope"] = latest_ma / prev_3m - 1

    if len(price) > 63:
        px_3m = price.iloc[-64]
        if pd.notna(px_3m) and px_3m != 0:
            out["Momentum_3M"] = latest_price / px_3m - 1

    if len(price) > 126:
        px_6m = price.iloc[-127]
        if pd.notna(px_6m) and px_6m != 0:
            out["Momentum_6M"] = latest_price / px_6m - 1

    return out


def robust_trend_score(stats: pd.DataFrame) -> pd.Series:
    if stats is None or stats.empty:
        return pd.Series(
            {
                "SharpeMedian": np.nan,
                "SharpeMin": np.nan,
                "SharpeMax": np.nan,
                "SharpeRange": np.nan,
                "CAGRMedian": np.nan,
                "MaxDDWorst": np.nan,
                "ProfitableWindows": np.nan,
                "AvgTimeInMarket": np.nan,
                "RobustScore": np.nan,
            }
        )

    sharpe = pd.to_numeric(stats["Sharpe"], errors="coerce")
    cagr = pd.to_numeric(stats["CAGR"], errors="coerce")
    maxdd = pd.to_numeric(stats["MaxDD"], errors="coerce")
    time_in = pd.to_numeric(stats["TimeInMarket"], errors="coerce")

    sharpe_median = sharpe.median()
    sharpe_min = sharpe.min()
    sharpe_max = sharpe.max()
    sharpe_range = sharpe_max - sharpe_min if pd.notna(sharpe_max) and pd.notna(sharpe_min) else np.nan
    cagr_median = cagr.median()
    maxdd_worst = maxdd.min()
    profitable_windows = (cagr > 0).mean() if cagr.notna().any() else np.nan
    avg_time_in = time_in.mean()
    robust_score = np.nan
    inputs = [sharpe_median, sharpe_min, cagr_median, profitable_windows, maxdd_worst]
    if all(pd.notna(x) for x in inputs):
        robust_score = (
            0.35 * sharpe_median
            + 0.20 * sharpe_min
            + 0.20 * cagr_median
            + 0.15 * profitable_windows
            + 0.10 * maxdd_worst
        )

    return pd.Series(
        {
            "SharpeMedian": sharpe_median,
            "SharpeMin": sharpe_min,
            "SharpeMax": sharpe_max,
            "SharpeRange": sharpe_range,
            "CAGRMedian": cagr_median,
            "MaxDDWorst": maxdd_worst,
            "ProfitableWindows": profitable_windows,
            "AvgTimeInMarket": avg_time_in,
            "RobustScore": robust_score,
        }
    )


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
    include_leveraged = getenv_bool("INCLUDE_LEVERAGED", False)
    df = df.copy()
    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df["Category"] = df["Category"].astype(str)

    mask = (
        df["Category"]
        .astype(str)
        .str.contains("|".join(KEEP_PATTERNS), case=False, na=False)
    )
    if not include_leveraged:
        mask &= ~df["Category"].str.contains(
            "Leveraged|Inverse|Short", case=False, na=False
        )

    df = df[df["Avg-Vol"] >= MIN_AVG_VOL].copy()
    df["Start Date"] = pd.to_datetime(df["Start Date"], errors="coerce")

    univ_start = pd.Timestamp(UNIVERSE_START)
    df = df[df["Start Date"].notna() & (df["Start Date"] <= univ_start) & mask]
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
    ap.add_argument("--mode", choices=["detailed", "robust"], default="robust")
    ap.add_argument(
        "--email",
        action="store_true",
        help="Send email with trend monitor results",
    )
    args = ap.parse_args()

    end = args.end or datetime.now(timezone.utc).strftime("%Y-%m-%d")

    df_univ = pd.read_csv(args.universe_csv)
    total_loaded = len(df_univ)
    df_univ["Ticker"] = df_univ["Ticker"].astype(str).str.upper().str.strip()
    df_univ_filt = filter_universe(df_univ)
    filtered_universe_count = len(df_univ_filt)

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
    valid_price_count = int(px.notna().any().sum())

    dfs = []
    for t in px.columns:
        stats = ma_strategy_metrics(px[t], MA_WINS)
        if not stats.empty:
            dfs.append(stats.assign(Ticker=t))

    if not dfs:
        print("No valid strategy metrics computed.")
        return

    df_results = pd.concat(dfs, ignore_index=True)
    meta_cols = ["Ticker", "Fund Name", "Start Date", "Category", "Avg-Vol"]
    df_results = df_results.merge(df_univ_filt[meta_cols], on="Ticker", how="left")

    cols_detailed = [
        "Ticker",
        "Fund Name",
        "Category",
        "Avg-Vol",
        "Start Date",
        "RobustScore",
        "MaWin",
        "Sharpe",
        "CAGR",
        "MaxDD",
        "TimeInMarket",
    ]
    min_sharpe = getenv_float("MIN_SHARPE", 0.6)
    max_sharpe = getenv_float("MAX_SHARPE", 2.0)
    max_dd = getenv_float("MAX_DD", -0.25)
    min_profitable_windows = getenv_float("MIN_PROFITABLE_WINDOWS", 0.70)

    df_filtered = df_results[
        (df_results["Sharpe"] >= min_sharpe)
        & (df_results["Sharpe"] <= max_sharpe)
        & (df_results["MaxDD"] >= max_dd)
    ].copy()

    robust_rows = []
    for t in px.columns:
        price = px[t].dropna()
        if price.empty:
            continue

        robust = robust_trend_score(df_results[df_results["Ticker"] == t])
        robust_rows.append(
            {
                "Ticker": t,
                **robust.to_dict(),
                **buy_hold_metrics(price),
                **current_trend_features(price, ma_window=200),
            }
        )

    df_ranked = pd.DataFrame(
        robust_rows,
        columns=[
            "Ticker",
            "SharpeMedian",
            "SharpeMin",
            "SharpeMax",
            "SharpeRange",
            "CAGRMedian",
            "MaxDDWorst",
            "ProfitableWindows",
            "AvgTimeInMarket",
            "RobustScore",
            "BH_CAGR",
            "BH_Sharpe",
            "BH_MaxDD",
            "BH_Vol",
            "AboveMA",
            "DistanceFromMA",
            "MA_1M_Slope",
            "MA_3M_Slope",
            "Momentum_3M",
            "Momentum_6M",
        ],
    )
    if not df_ranked.empty:
        df_ranked = df_ranked.merge(df_univ_filt[meta_cols], on="Ticker", how="left")
        df_ranked["SharpeEdge"] = df_ranked["SharpeMedian"] - df_ranked["BH_Sharpe"]
        df_ranked["DrawdownImprovement"] = (
            df_ranked["BH_MaxDD"] - df_ranked["MaxDDWorst"]
        )
        df_ranked["DrawdownReduction"] = (
            df_ranked["BH_MaxDD"].abs() - df_ranked["MaxDDWorst"].abs()
        )
    else:
        for col in meta_cols[1:] + ["SharpeEdge", "DrawdownImprovement", "DrawdownReduction"]:
            df_ranked[col] = pd.Series(dtype=float if col != "Fund Name" and col != "Category" and col != "Start Date" else object)

    df_filtered = df_filtered.merge(
        df_ranked[["Ticker", "RobustScore"]],
        on="Ticker",
        how="left",
    )

    df_robust_filtered = df_ranked[
        (df_ranked["SharpeMedian"] >= min_sharpe)
        & (df_ranked["MaxDDWorst"] >= max_dd)
        & (df_ranked["ProfitableWindows"] >= min_profitable_windows)
        & (df_ranked["Avg-Vol"] >= MIN_AVG_VOL)
        & (df_ranked["AboveMA"] == True)
    ].copy()

    df_robust_filtered = df_robust_filtered.sort_values(
        ["RobustScore", "SharpeEdge", "DrawdownReduction"],
        ascending=[False, False, False],
    )

    if args.mode == "detailed":
        out_df = (
            df_filtered[cols_detailed]
            .assign(
                RobustScore=lambda x: pd.to_numeric(x["RobustScore"], errors="coerce").round(3),
                Sharpe=lambda x: x["Sharpe"].round(3),
                CAGR=lambda x: x["CAGR"].round(3),
                MaxDD=lambda x: x["MaxDD"].round(3),
                TimeInMarket=lambda x: x["TimeInMarket"].round(3),
            )
            .sort_values(["RobustScore", "Sharpe"], ascending=[False, False])
            .head(50)
        )
        output_df = df_filtered[cols_detailed]
        email_df = df_filtered
        email_cols = cols_detailed
        email_sort = "RobustScore"
        passing_count = len(df_filtered)
    else:
        robust_cols = [c for c in ROBUST_OUTPUT_COLS if c in df_robust_filtered.columns]
        out_df = df_robust_filtered[robust_cols].copy().head(20)
        round_cols = [c for c in robust_cols if c not in {"Ticker", "Fund Name", "Category", "Start Date", "AboveMA"}]
        for c in round_cols:
            out_df[c] = pd.to_numeric(out_df[c], errors="coerce").round(3)
        output_df = df_robust_filtered[robust_cols]
        email_df = df_robust_filtered
        email_cols = robust_cols
        email_sort = "RobustScore"
        passing_count = len(df_robust_filtered)

    print(f"Total ETFs loaded: {total_loaded}")
    print(f"ETFs after universe filter: {filtered_universe_count}")
    print(f"ETFs with valid price data: {valid_price_count}")
    if args.mode == "robust":
        print(f"ETFs passing robust filters: {passing_count}")
    else:
        print(f"ETFs passing detailed filters: {passing_count}")
    print("Top 20 ranked ETFs:")
    print(out_df.to_string(index=False))

    if args.out_filtered:
        output_df.to_csv(args.out_filtered, index=False)

    if args.email:
        message_body_html, message_body_plain = build_trend_monitor_tables(
            email_df, cols=email_cols, top_n=10, sort_by=email_sort
        )
        attachment_df = output_df.copy()
        if email_sort in attachment_df.columns:
            attachment_df = attachment_df.sort_values(email_sort, ascending=False)
        attachment_csv = attachment_df.to_csv(index=False).encode("utf-8")

        TO_ADDRESSES = [x.strip() for x in os.getenv("TO_ADDRESSES", "").split(",") if x.strip()]
        FROM_ADDRESS = os.getenv("FROM_ADDRESS", "").strip()
        if not TO_ADDRESSES:
            raise ValueError("Missing required environment variable: TO_ADDRESSES")
        ses = AmazonSES(
            region=os.environ.get("AWS_SES_REGION_NAME"),
            access_key=os.environ.get("AWS_SES_ACCESS_KEY_ID"),
            secret_key=os.environ.get("AWS_SES_SECRET_ACCESS_KEY"),
            from_address=FROM_ADDRESS,
        )

        today_str = datetime.now(timezone.utc).strftime("%B %d, %Y")
        date_stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        subject = f"ETF Trend Monitor – Week of {today_str}"
        attachment_name = f"etf-trend-monitor-{args.mode}-{date_stamp}.csv"

        for to_address in TO_ADDRESSES:
            ses.send_html_email_with_attachment(
                to_address=to_address,
                subject=subject,
                html_content=message_body_html,
                text_content=message_body_plain,
                attachment_name=attachment_name,
                attachment_bytes=attachment_csv,
            )


if __name__ == "__main__":
    main()
