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
from regime_detector import RegimeDetector

load_dotenv(find_dotenv())

REGIME_ORDER = [
    "Stable Risk-On",
    "Fragile",
    "Vol Shock",
    "Crisis",
]


def getenv_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default

    value = value.strip()
    if not value:
        return default

    return float(value)

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


def _annualized_metrics(strat_ret: pd.Series, signal: pd.Series) -> dict[str, float]:
    strat_ret = strat_ret.dropna()
    signal = signal.reindex(strat_ret.index).dropna()

    if strat_ret.empty:
        return {
            "CAGR": np.nan,
            "Sharpe": np.nan,
            "Vol": np.nan,
            "MaxDD": np.nan,
            "TimeInMarket": np.nan,
            "Obs": 0,
        }

    cum = (1 + strat_ret).cumprod()
    yrs = (cum.index[-1] - cum.index[0]).days / 365.25
    vol = strat_ret.std() * np.sqrt(252)
    sharpe = (strat_ret.mean() * 252) / vol if vol > 0 else np.nan
    cagr = cum.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else np.nan

    return {
        "CAGR": cagr,
        "Sharpe": sharpe,
        "Vol": vol,
        "MaxDD": (cum / cum.cummax() - 1).min(),
        "TimeInMarket": float(signal.mean()) if not signal.empty else np.nan,
        "Obs": int(len(strat_ret)),
    }


def regime_strategy_metrics(
    price: pd.Series, ma_wins: Iterable[int], regimes: pd.Series
) -> pd.DataFrame:
    price = price.dropna()
    regimes = regimes.dropna()
    if price.empty or regimes.empty:
        return pd.DataFrame()

    ret = price.pct_change()
    rows = []

    for w in ma_wins:
        ma = price.rolling(w).mean()
        signal = (price > ma).shift(1)
        strat_ret = (ret * signal).dropna()
        aligned = pd.DataFrame(
            {
                "StratRet": strat_ret,
                "Signal": signal.reindex(strat_ret.index),
                "RegimeLabel": regimes.reindex(strat_ret.index),
            }
        ).dropna(subset=["RegimeLabel"])

        if aligned.empty:
            continue

        for regime_label, group in aligned.groupby("RegimeLabel"):
            metrics = _annualized_metrics(group["StratRet"], group["Signal"])
            if metrics["Obs"] == 0:
                continue

            rows.append(
                {
                    "MaWin": w,
                    "RegimeLabel": regime_label,
                    "RegimeDays": metrics["Obs"],
                    "Sharpe": metrics["Sharpe"],
                    "CAGR": metrics["CAGR"],
                    "Vol": metrics["Vol"],
                    "MaxDD": metrics["MaxDD"],
                    "TimeInMarket": metrics["TimeInMarket"],
                }
            )

    return pd.DataFrame(rows)


def summarize_regime_category_etfs(
    df_regime_filtered: pd.DataFrame,
    top_categories: int = 5,
    top_etfs_per_category: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if df_regime_filtered.empty:
        return pd.DataFrame(), pd.DataFrame()

    df_ranked = df_regime_filtered.copy()
    df_ranked["Category"] = df_ranked["Category"].fillna("Uncategorized")
    df_ranked["Fund Name"] = df_ranked["Fund Name"].fillna("")

    category_summary = (
        df_ranked.sort_values(
            ["RegimeLabel", "Sharpe", "CAGR"], ascending=[True, False, False]
        )
        .groupby(["RegimeLabel", "Category"], dropna=False)
        .agg(
            ETFCount=("Ticker", "nunique"),
            AvgSharpe=("Sharpe", "mean"),
            AvgCAGR=("CAGR", "mean"),
            AvgMaxDD=("MaxDD", "mean"),
            AvgTimeInMarket=("TimeInMarket", "mean"),
        )
        .reset_index()
    )

    category_summary["CategoryRank"] = category_summary.groupby("RegimeLabel")[
        "AvgSharpe"
    ].rank(method="first", ascending=False)
    category_summary = category_summary[
        category_summary["CategoryRank"] <= top_categories
    ].copy()

    for col in ["AvgSharpe", "AvgCAGR", "AvgMaxDD", "AvgTimeInMarket"]:
        category_summary[col] = pd.to_numeric(
            category_summary[col], errors="coerce"
        ).round(3)

    category_summary["ETFCount"] = pd.to_numeric(
        category_summary["ETFCount"], errors="coerce"
    ).astype("Int64")
    category_summary["CategoryRank"] = pd.to_numeric(
        category_summary["CategoryRank"], errors="coerce"
    ).astype("Int64")

    regime_category_keys = category_summary[["RegimeLabel", "Category"]].drop_duplicates()
    etf_rankings = df_ranked.merge(
        regime_category_keys,
        on=["RegimeLabel", "Category"],
        how="inner",
    )
    etf_rankings = etf_rankings.sort_values(
        ["RegimeLabel", "Category", "Sharpe", "CAGR"],
        ascending=[True, True, False, False],
    )
    etf_rankings["ETFCategoryRank"] = etf_rankings.groupby(
        ["RegimeLabel", "Category"]
    ).cumcount() + 1
    etf_rankings = etf_rankings[
        etf_rankings["ETFCategoryRank"] <= top_etfs_per_category
    ].copy()
    etf_rankings = etf_rankings.merge(
        category_summary[["RegimeLabel", "Category", "CategoryRank"]],
        on=["RegimeLabel", "Category"],
        how="left",
    )

    for col in ["Sharpe", "CAGR", "MaxDD", "TimeInMarket"]:
        etf_rankings[col] = pd.to_numeric(etf_rankings[col], errors="coerce").round(3)

    etf_rankings["MaWin"] = pd.to_numeric(etf_rankings["MaWin"], errors="coerce").astype(
        "Int64"
    )
    etf_rankings["RegimeDays"] = pd.to_numeric(
        etf_rankings["RegimeDays"], errors="coerce"
    ).astype("Int64")
    etf_rankings["CategoryRank"] = pd.to_numeric(
        etf_rankings["CategoryRank"], errors="coerce"
    ).astype("Int64")
    etf_rankings["ETFCategoryRank"] = pd.to_numeric(
        etf_rankings["ETFCategoryRank"], errors="coerce"
    ).astype("Int64")

    category_summary = category_summary.sort_values(
        ["RegimeLabel", "CategoryRank", "AvgSharpe"], ascending=[True, True, False]
    )
    etf_rankings = etf_rankings.sort_values(
        ["RegimeLabel", "CategoryRank", "ETFCategoryRank", "Sharpe"],
        ascending=[True, True, True, False],
    )

    return category_summary, etf_rankings


def build_regime_email_tables(
    regime_summary: dict,
    regime_category_summary: pd.DataFrame,
    regime_etf_rankings: pd.DataFrame,
) -> tuple[str, str]:
    html_parts = [
        "<br><b>Regime Monitor</b><br>",
        (
            "<p style='font-family:Arial, sans-serif; font-size:13px;'>"
            f"<b>As of:</b> {regime_summary['as_of']}<br>"
            f"<b>Dominant regime ({regime_summary['dominance_window']}d):</b> "
            f"{regime_summary['dominant_label']}<br>"
            f"<b>Latest regime:</b> {regime_summary['last_label']}"
            "</p>"
        ),
    ]
    plain_parts = [
        "Regime Monitor",
        (
            f"As of: {regime_summary['as_of']}\n"
            f"Dominant regime ({regime_summary['dominance_window']}d): "
            f"{regime_summary['dominant_label']}\n"
            f"Latest regime: {regime_summary['last_label']}\n"
        ),
    ]

    summary_rows = []
    for label in REGIME_ORDER:
        days = int(regime_summary["counts"].get(label, 0))
        summary_rows.append({"Regime": label, "DaysInWindow": days})

    summary_df = pd.DataFrame(summary_rows)
    summary_thead = "<tr>" + "".join(html_th(h) for h in summary_df.columns) + "</tr>"
    summary_body = "".join(
        "<tr>"
        + html_td(row["Regime"], align="left")
        + html_td(row["DaysInWindow"])
        + "</tr>"
        for _, row in summary_df.iterrows()
    )
    html_parts.append(
        '<div style="width:100%; overflow-x:auto;">'
        '<table style="width:100%; max-width:500px; border-collapse:collapse; '
        'font-family:Arial, sans-serif; font-size:12px;">'
        f"<thead>{summary_thead}</thead><tbody>{summary_body}</tbody></table></div><br>"
    )
    plain_parts.append(summary_df.to_string(index=False) + "\n")

    if regime_category_summary.empty or regime_etf_rankings.empty:
        html_parts.append("<p>No regime-specific category leaders matched the filters.</p>")
        plain_parts.append("No regime-specific category leaders matched the filters.\n")
        return "".join(html_parts), "\n".join(plain_parts)

    for label in REGIME_ORDER:
        category_section = regime_category_summary[
            regime_category_summary["RegimeLabel"] == label
        ].copy()
        if category_section.empty:
            html_parts.append(f"<p><b>{label}</b>: No category leaders matched filters.</p>")
            plain_parts.append(f"{label}: No category leaders matched filters.\n")
            continue

        html_parts.append(
            (
                "<br>"
                f"<div style='font-family:Arial, sans-serif; font-size:16px; "
                f"font-weight:700; margin:10px 0 4px 0;'>{label}</div>"
            )
        )
        plain_parts.append(f"\n{label}\n")

        for _, category_row in category_section.iterrows():
            category_name = category_row["Category"]
            etf_section = regime_etf_rankings[
                (regime_etf_rankings["RegimeLabel"] == label)
                & (regime_etf_rankings["Category"] == category_name)
            ].copy()

            html_parts.append(
                (
                    "<p style='font-family:Arial, sans-serif; font-size:13px; margin:12px 0 6px 0;'>"
                    f"<b>#{int(category_row['CategoryRank'])} {category_name}</b> "
                    f"(ETFs: {int(category_row['ETFCount'])}, "
                    f"Avg Sharpe: {category_row['AvgSharpe']}, "
                    f"Avg CAGR: {category_row['AvgCAGR']}, "
                    f"Avg MaxDD: {category_row['AvgMaxDD']}, "
                    f"Avg Time In Market: {category_row['AvgTimeInMarket']})"
                    "</p>"
                )
            )
            plain_parts.append(
                f"#{int(category_row['CategoryRank'])} {category_name} "
                f"(ETFs: {int(category_row['ETFCount'])}, Avg Sharpe: {category_row['AvgSharpe']}, "
                f"Avg CAGR: {category_row['AvgCAGR']}, Avg MaxDD: {category_row['AvgMaxDD']}, "
                f"Avg Time In Market: {category_row['AvgTimeInMarket']})\n"
            )

            if etf_section.empty:
                html_parts.append("<p>No ETFs matched filters in this category.</p>")
                plain_parts.append("No ETFs matched filters in this category.\n")
                continue

            etf_display_cols = [
                "ETFCategoryRank",
                "Ticker",
                "Fund Name",
                "MaWin",
                "Sharpe",
                "CAGR",
                "MaxDD",
                "TimeInMarket",
                "RegimeDays",
            ]
            etf_section["Ticker"] = etf_section["Ticker"].astype(str).map(
                lambda t: (
                    f'<a clicktracking="off" href="https://finviz.com/quote.ashx?t={t}">{t}</a>'
                )
            )
            thead = "<tr>" + "".join(html_th(h) for h in etf_display_cols) + "</tr>"
            body_rows = []
            for _, etf_row in etf_section.iterrows():
                body_rows.append(
                    "<tr>"
                    + "".join(
                        html_td(
                            etf_row[col],
                            align="left" if col in {"Ticker", "Fund Name"} else "right",
                        )
                        for col in etf_display_cols
                    )
                    + "</tr>"
                )
            html_parts.append(
                '<div style="width:100%; overflow-x:auto;">'
                '<table style="width:100%; min-width:1100px; border-collapse:collapse; '
                'font-family:Arial, sans-serif; font-size:12px;">'
                f"<thead>{thead}</thead><tbody>{''.join(body_rows)}</tbody></table></div>"
            )

            plain_section = regime_etf_rankings[
                (regime_etf_rankings["RegimeLabel"] == label)
                & (regime_etf_rankings["Category"] == category_name)
            ][etf_display_cols].copy()
            plain_parts.append(plain_section.to_string(index=False) + "\n")

    return "".join(html_parts) + "<br>", "\n".join(plain_parts)


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

    detector = RegimeDetector(
        vix_high_pct=getenv_float("REGIME_VIX_HIGH_PCT", 0.70),
        spread_wide_pct=getenv_float("REGIME_SPREAD_WIDE_PCT", 0.70),
        lookback=int(getenv_float("REGIME_LOOKBACK", 252)),
        ema_span=int(getenv_float("REGIME_EMA_SPAN", 10)),
        dominance_window=int(getenv_float("REGIME_DOMINANCE_WINDOW", 20)),
        credit_mode=os.getenv("REGIME_CREDIT_MODE", "ratio").strip() or "ratio",
    )
    regime_frame = detector.build_regimes(start_date=UNIVERSE_START, end_date=end)
    regime_labels = regime_frame["RegimeLabel"].dropna()
    regime_summary = detector.dominant_regime(as_of=end, return_diagnostics=True)

    dfs = []
    regime_dfs = []
    for t in px.columns:
        stats = ma_strategy_metrics(px[t], MA_WINS)
        if not stats.empty:
            dfs.append(stats.assign(Ticker=t))
        regime_stats = regime_strategy_metrics(px[t], MA_WINS, regime_labels)
        if not regime_stats.empty:
            regime_dfs.append(regime_stats.assign(Ticker=t))

    if not dfs:
        print("No moving-average strategy results were generated.")
        return

    df_results = pd.concat(dfs, ignore_index=True)
    df_results = df_results.merge(df_univ, on="Ticker", how="left")

    if regime_dfs:
        df_regime_results = pd.concat(regime_dfs, ignore_index=True)
        df_regime_results = df_regime_results.merge(df_univ, on="Ticker", how="left")
    else:
        df_regime_results = pd.DataFrame(
            columns=[
                "Ticker",
                "MaWin",
                "RegimeLabel",
                "RegimeDays",
                "Sharpe",
                "CAGR",
                "Vol",
                "MaxDD",
                "TimeInMarket",
            ]
        )

    df_filtered = df_results[
        (df_results["Sharpe"] >= getenv_float("MIN_SHARPE", 0.6))
        & (df_results["Sharpe"] <= getenv_float("MAX_SHARPE", 2.0))
        & (df_results["MaxDD"] >= getenv_float("MAX_DD", -0.25))
    ].copy()

    df_regime_filtered = df_regime_results[
        (df_regime_results["Sharpe"] >= getenv_float("MIN_SHARPE", 0.6))
        & (df_regime_results["Sharpe"] <= getenv_float("MAX_SHARPE", 2.0))
        & (df_regime_results["MaxDD"] >= getenv_float("MAX_DD", -0.25))
    ].copy()

    # Normalize ticker for safe merging
    df_univ["Ticker"] = df_univ["Ticker"].astype(str).str.upper().str.strip()
    df_filtered["Ticker"] = df_filtered["Ticker"].astype(str).str.upper().str.strip()
    df_regime_filtered["Ticker"] = (
        df_regime_filtered["Ticker"].astype(str).str.upper().str.strip()
    )

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
    df_regime_filtered = df_regime_filtered.merge(
        df_univ,
        on="Ticker",
        how="left",
        suffixes=("", "_univ"),
    )
    df_regime_filtered = df_regime_filtered.drop(
        columns=[c for c in df_regime_filtered.columns if c.endswith("_univ")]
    )
    regime_category_summary, regime_etf_rankings = summarize_regime_category_etfs(
        df_regime_filtered,
        top_categories=int(getenv_float("REGIME_TOP_CATEGORIES", 5)),
        top_etfs_per_category=int(getenv_float("REGIME_TOP_ETFS_PER_CATEGORY", 5)),
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
        regime_html, regime_plain = build_regime_email_tables(
            regime_summary,
            regime_category_summary,
            regime_etf_rankings,
        )
        message_body_html = regime_html + message_body_html
        message_body_plain = regime_plain + "\n" + message_body_plain

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
        subject = f"ETF Trend Monitor – Week of {today_str}"

        for to_address in TO_ADDRESSES:
            ses.send_html_email(
                to_address=to_address, subject=subject, content=message_body_html
            )


if __name__ == "__main__":
    main()
