#!/usr/bin/env python3
"""
StealthScore Factor — IC Analysis & Quintile Backtest (Real Returns)
=====================================================================
Uses CSMAR daily return data (daily_returns_cn_all.csv) for IC analysis
and quintile portfolio backtest.

Data flow:
  1. Load StealthScore factor (from validate_factor.py output)
  2. Map stock names → CSMAR stock codes via listing master
  3. Load daily returns for matched stocks (chunked, memory-safe)
  4. Daily Rank IC computation over target year
  5. Quintile portfolio cumulative returns

Requires:
  - results/<date>/stealth_score_factor.csv (from validate_factor.py)
  - CSMAR daily_returns_cn_all.csv (external, ~621MB)
  - CSMAR listing_master_by_year_clean.csv (external)
"""
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ═══════════════════════════════════════════════════════════
# Paths
# ═══════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"

# External CSMAR data (from genai_china_replication project)
CSMAR_ROOT = Path("/Users/leolee/Desktop/genai_china_replication")
DAILY_RETURNS = CSMAR_ROOT / "data/processed/daily_returns_cn_all.csv"
LISTING_MASTER = CSMAR_ROOT / "data/raw/csmar/listing_master_by_year_clean.csv"

# Find latest factor audit run
audit_dirs = sorted(RESULTS_ROOT.glob("20*"), reverse=True)
if not audit_dirs:
    print("ERROR: No audit results found. Run validate_factor.py first.")
    sys.exit(1)
IN_DIR = audit_dirs[0]
OUT_DIR = IN_DIR  # same output directory
FACTOR_FILE = IN_DIR / "stealth_score_factor.csv"

RED = "#E74C3C"
GREEN = "#27AE60"
BLUE = "#2980B9"
GRAY = "#7F8C8D"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10,
    "font.sans-serif": ["Heiti SC", "STHeiti", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


# ═══════════════════════════════════════════════════════════
# Step 1: Load factor + map names to codes
# ═══════════════════════════════════════════════════════════

def load_factor():
    df = pd.read_csv(FACTOR_FILE)
    df = df[df["coverage_flag"]].copy()
    df["StealthScore_rank"] = df["StealthScore"].rank(pct=True)
    df["quintile"] = pd.cut(
        df["StealthScore_rank"],
        bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=[1, 2, 3, 4, 5],
        include_lowest=True,
    ).astype(int)
    print(f"  Loaded {len(df)} stocks with valid StealthScore")
    return df


def map_name_to_code(factor_df, listing_year=2024):
    """Map stock names to CSMAR stock_ids using listing master.

    Uses the specified year's listing master. Falls back to exact-match
    if name not found in listing master (handles renamed stocks).
    """
    print(f"\n  Loading listing master (year={listing_year})...")
    master = pd.read_csv(LISTING_MASTER)
    master = master[master["year"] == listing_year].copy()
    master["name_stripped"] = master["ShortName"].str.replace("*", "").str.strip()

    # Build lookup: stripped name -> stock_id
    name_to_id = dict(zip(master["name_stripped"], master["stock_id"]))
    id_to_name = dict(zip(master["stock_id"], master["ShortName"]))

    # Also build: exact name -> stock_id (for ST stocks with asterisks)
    exact_name_to_id = dict(zip(master["ShortName"].str.strip(), master["stock_id"]))

    stock_names = factor_df["stock_name"].str.replace("*", "").str.strip().tolist()
    stock_names_orig = factor_df["stock_name"].str.strip().tolist()

    stock_ids = []
    n_exact, n_stripped, n_miss = 0, 0, 0
    for orig, stripped in zip(stock_names_orig, stock_names):
        sid = None
        # Try exact match first
        if orig in exact_name_to_id:
            sid = exact_name_to_id[orig]
            n_exact += 1
        # Try stripped match
        elif stripped in name_to_id:
            sid = name_to_id[stripped]
            n_stripped += 1
        else:
            n_miss += 1
        stock_ids.append(sid)

    factor_df["stock_id"] = stock_ids
    print(f"  Name mapping: {n_exact} exact, {n_stripped} stripped, {n_miss} unmatched")

    # Show some unmatched names for debugging
    if n_miss > 0:
        unmatched = factor_df[pd.isna(factor_df["stock_id"])]["stock_name"].head(10).tolist()
        print(f"  Sample unmatched: {unmatched}")

    return factor_df, id_to_name


# ═══════════════════════════════════════════════════════════
# Step 2: Load daily returns (memory-safe, chunked)
# ═══════════════════════════════════════════════════════════

def load_daily_returns(stock_ids, target_year=2025):
    """Chunked read of CSMAR daily returns, filter to target stocks and year.

    Returns: pivot table (stock_id × trade_date) with daily returns.
    """
    print(f"\n  Loading daily returns for {len(stock_ids)} stocks, year={target_year}...")
    stock_set = set(stock_ids)

    frames = []
    chunks_read = 0
    for chunk in pd.read_csv(DAILY_RETURNS, chunksize=200000, dtype={"stock_id": str}):
        chunks_read += 1
        # Filter to target stocks
        chunk = chunk[chunk["stock_id"].isin(stock_set)]
        if len(chunk) == 0:
            continue
        # Parse date and filter year
        chunk["trade_date"] = pd.to_datetime(chunk["trade_date"])
        chunk = chunk[chunk["trade_date"].dt.year == target_year]
        if len(chunk) == 0:
            continue
        frames.append(chunk[["stock_id", "trade_date", "ret"]])

        if chunks_read % 50 == 0:
            print(f"    Processed {chunks_read} chunks...")

    if not frames:
        print("  WARNING: No return data found for target stocks/year!")
        return None, None

    df_ret = pd.concat(frames, ignore_index=True)
    del frames

    n_stocks = df_ret["stock_id"].nunique()
    n_dates = df_ret["trade_date"].nunique()
    print(f"  Loaded returns: {len(df_ret)} rows, {n_stocks} stocks, {n_dates} dates")

    # Pivot to wide: rows=stock_id, cols=trade_date
    ret_matrix = df_ret.pivot_table(
        index="stock_id", columns="trade_date", values="ret", aggfunc="first"
    )
    dates_sorted = sorted(ret_matrix.columns)
    ret_matrix = ret_matrix[dates_sorted]

    print(f"  Pivot shape: {ret_matrix.shape[0]} stocks × {ret_matrix.shape[1]} dates")
    return ret_matrix


# ═══════════════════════════════════════════════════════════
# Step 3: IC Analysis (Daily)
# ═══════════════════════════════════════════════════════════

def ic_analysis(factor_df, return_matrix):
    """Daily Rank IC and Pearson IC analysis.

    For each trading day, compute cross-sectional IC between
    StealthScore and daily returns. Summarize with mean, std, IR, hit ratio.
    """
    print("\n" + "=" * 60)
    print("IC Analysis (Daily, 2025)")
    print("=" * 60)

    # Align: factor stocks that exist in return matrix
    mapped = factor_df.dropna(subset=["stock_id"]).copy()
    valid_ids = set(return_matrix.index)
    mapped = mapped[mapped["stock_id"].isin(valid_ids)].copy()

    if len(mapped) == 0:
        print("  ERROR: No matched stocks in return data!")
        return None, return_matrix.columns

    n_stocks = len(mapped)
    print(f"  Matched stocks for IC: {n_stocks}")

    # Sort both to ensure alignment
    mapped = mapped.set_index("stock_id").loc[return_matrix.index.intersection(mapped["stock_id"])]
    factor_values = mapped["StealthScore"].values

    rank_ics, pearson_ics = [], []
    dates_used = []

    for date in return_matrix.columns:
        rets = return_matrix.loc[mapped.index, date].values
        valid = ~np.isnan(rets)
        if valid.sum() < 30:  # minimum 30 stocks for reliable IC
            continue
        rank_ics.append(stats.spearmanr(factor_values[valid], rets[valid])[0])
        pearson_ics.append(stats.pearsonr(factor_values[valid], rets[valid])[0])
        dates_used.append(date)

    rank_ics = np.array(rank_ics)
    pearson_ics = np.array(pearson_ics)

    n_obs = len(rank_ics)
    print(f"  IC observations: {n_obs} trading days")

    results = {
        "n_days": n_obs,
        "n_stocks": n_stocks,
        "RankIC_mean": rank_ics.mean(),
        "RankIC_std": rank_ics.std(),
        "RankIC_IR": rank_ics.mean() / (rank_ics.std() + 1e-8),
        "RankIC_pos_ratio": (rank_ics > 0).mean(),
        "RankIC_tstat": rank_ics.mean() / (rank_ics.std() / np.sqrt(n_obs)) if n_obs > 1 else 0,
        "PearsonIC_mean": pearson_ics.mean(),
        "PearsonIC_std": pearson_ics.std(),
        "PearsonIC_IR": pearson_ics.mean() / (pearson_ics.std() + 1e-8),
    }

    for k, v in results.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # ── IC Time Series Plot ──
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = [RED if v > 0 else GREEN for v in rank_ics]
    ax.bar(range(n_obs), rank_ics, color=colors, alpha=0.7, width=1.0)
    ax.axhline(0, color=GRAY, linewidth=1)
    ax.axhline(rank_ics.mean(), color=BLUE, linestyle="--", linewidth=1.5,
               label=f"Mean RankIC={rank_ics.mean():.4f}")
    # Rolling 20-day mean
    if n_obs >= 20:
        from pandas import Series
        rolling = Series(rank_ics).rolling(20).mean()
        ax.plot(range(19, n_obs), rolling.values[19:], color=RED, linewidth=1.5, alpha=0.6, label="20d MA")

    ax.set_xlabel("Trading Day (2025)")
    ax.set_ylabel("Rank IC")
    ax.set_title(f"StealthScore Daily Rank IC — IR={results['RankIC_IR']:.3f}, "
                 f"t={results['RankIC_tstat']:.2f}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig_path = OUT_DIR / "ic_time_series.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart: {fig_path}")

    # ── IC Distribution Histogram ──
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    ax2.hist(rank_ics, bins=40, color=BLUE, alpha=0.7, edgecolor="white")
    ax2.axvline(0, color=GRAY, linewidth=1)
    ax2.axvline(rank_ics.mean(), color=RED, linestyle="--", linewidth=1.5,
                label=f"Mean={rank_ics.mean():.4f}")
    ax2.set_xlabel("Rank IC")
    ax2.set_ylabel("Frequency")
    ax2.set_title(f"Rank IC Distribution (skew={pd.Series(rank_ics).skew():.2f})")
    ax2.legend(fontsize=8)
    plt.tight_layout()
    fig_path2 = OUT_DIR / "ic_distribution.png"
    fig2.savefig(fig_path2, dpi=150, bbox_inches="tight")
    plt.close()

    # Tier assessment
    print("\n  Factor Tier Assessment (QMT conventions):")
    if abs(results["RankIC_mean"]) >= 0.02 and abs(results["RankIC_IR"]) >= 0.3:
        print(f"  ✅ CORE tier — |IC|>=0.02, |IR|>=0.3")
    elif abs(results["RankIC_mean"]) >= 0.01:
        print(f"  ⚡ RESERVE tier — |IC|>=0.01")
    elif abs(results["RankIC_mean"]) >= 0.005:
        print(f"  👀 WATCH tier — |IC|>=0.005")
    else:
        print(f"  ⚠️  OBSERVATION — |IC|<0.005, monitor only")

    # Direction check
    direction = "positive" if rank_ics.mean() > 0 else "negative"
    print(f"  Direction: {direction} ({results['RankIC_pos_ratio']*100:.0f}% days positive)")

    return results, rank_ics, dates_used


# ═══════════════════════════════════════════════════════════
# Step 4: Quintile Backtest (Daily)
# ═══════════════════════════════════════════════════════════

def quintile_backtest(factor_df, return_matrix):
    """Daily quintile portfolio backtest with cumulative return tracking."""
    print("\n" + "=" * 60)
    print("Quintile Portfolio Backtest (Daily, 2025)")
    print("=" * 60)

    # Align stocks
    mapped = factor_df.dropna(subset=["stock_id"]).copy()
    mapped = mapped[mapped["stock_id"].isin(return_matrix.index)].copy()
    mapped = mapped.set_index("stock_id")

    if len(mapped) == 0:
        print("  ERROR: No stocks for backtest!")
        return

    print(f"  Backtest stocks: {len(mapped)}")

    # Daily quintile returns: each day, equal-weight stocks by quintile
    quintile_daily = {}
    for q in sorted(mapped["quintile"].unique()):
        q_ids = mapped[mapped["quintile"] == q].index
        # Average return across stocks for each date
        q_ret = return_matrix.loc[q_ids].mean(axis=0)
        quintile_daily[q] = q_ret
        print(f"  Q{q}: {len(q_ids)} stocks")

    # Stats per quintile
    bt_stats = {}
    trading_days_per_year = 242
    for q, rets in quintile_daily.items():
        rets = rets.dropna().values
        cum = np.cumprod(1 + rets) - 1
        daily_mean = rets.mean()
        daily_vol = rets.std()
        annual_ret = (1 + daily_mean) ** trading_days_per_year - 1
        annual_vol = daily_vol * np.sqrt(trading_days_per_year)
        sharpe = (annual_ret - 0.025) / annual_vol if annual_vol > 0 else 0
        mdd = 0
        peak = cum[0]
        for v in cum:
            peak = max(peak, v)
            mdd = min(mdd, v - peak)
        bt_stats[q] = {
            "annual_return": annual_ret,
            "annual_vol": annual_vol,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "cum_return": cum[-1],
            "daily_mean_bps": daily_mean * 10000,
        }

    # Long-short: Q5 - Q1
    q_high = max(quintile_daily.keys())
    q_low = min(quintile_daily.keys())
    ls_rets = quintile_daily[q_high].dropna() - quintile_daily[q_low].dropna()
    ls_cum = np.cumprod(1 + ls_rets.values) - 1
    ls_daily_mean = ls_rets.mean()
    ls_annual = (1 + ls_daily_mean) ** trading_days_per_year - 1
    ls_vol = ls_rets.std() * np.sqrt(trading_days_per_year)
    ls_sharpe = (ls_annual - 0.025) / ls_vol if ls_vol > 0 else 0
    ls_t = ls_rets.mean() / (ls_rets.std() / np.sqrt(len(ls_rets))) if len(ls_rets) > 1 else 0

    print(f"\n  Long-Short (Q{q_high} - Q{q_low}):")
    print(f"    Annual Return: {ls_annual*100:.2f}%")
    print(f"    Annual Vol:    {ls_vol*100:.2f}%")
    print(f"    Sharpe:        {ls_sharpe:.3f}")
    print(f"    t-stat:        {ls_t:.3f}")
    print(f"    Max Drawdown:  {ls_cum.min()*100:.2f}%")

    for q in sorted(bt_stats.keys()):
        s = bt_stats[q]
        print(f"\n  Q{q}: AnnRet={s['annual_return']*100:.1f}%, "
              f"Vol={s['annual_vol']*100:.1f}%, Sharpe={s['sharpe']:.3f}, "
              f"MDD={s['max_drawdown']*100:.1f}%, Cum={s['cum_return']*100:.1f}%")

    # ── Cumulative Return Plot ──
    fig, ax = plt.subplots(figsize=(12, 6))
    colors_map = {1: GREEN, 2: "#7F8C8D", 3: "#34495E", 4: "#E67E22", 5: RED}

    for q in sorted(bt_stats.keys()):
        cum = np.cumprod(1 + quintile_daily[q].dropna().values) - 1
        c = colors_map.get(int(q), GRAY)
        ax.plot(cum, color=c, linewidth=1.5,
                label=f"Q{q} ({bt_stats[q]['annual_return']*100:.1f}%/yr, "
                      f"Sharpe={bt_stats[q]['sharpe']:.2f})", alpha=0.9)

    # Long-short line
    ls_color = RED if ls_cum[-1] > 0 else GREEN
    ax.plot(ls_cum, color=ls_color, linewidth=2, linestyle="--",
            label=f"LS Q{q_high}-Q{q_low} ({ls_annual*100:.1f}%/yr, t={ls_t:.2f})")

    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Trading Day (2025)")
    ax.set_ylabel("Cumulative Return")
    ax.set_title("StealthScore Quintile Cumulative Returns (Daily, 2025)")
    ax.legend(fontsize=7, loc="upper left")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    plt.tight_layout()
    fig_path = OUT_DIR / "quintile_cumulative.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart: {fig_path}")

    return bt_stats, ls_annual, ls_t, ls_sharpe


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main(target_year: int = 2025):
    print("=" * 60)
    print(f"StealthScore IC Analysis & Quintile Backtest")
    print(f"Return data year: {target_year}")
    print(f"Factor file: {FACTOR_FILE}")
    print("=" * 60)

    # Load factor
    df = load_factor()

    # Name → code mapping
    df, id_to_name = map_name_to_code(df, listing_year=min(target_year - 1, 2024))

    # Load daily returns
    valid_ids = df["stock_id"].dropna().unique().tolist()
    ret_matrix = load_daily_returns(valid_ids, target_year=target_year)
    if ret_matrix is None:
        print("FATAL: No return data available.")
        sys.exit(1)

    # IC Analysis
    ic_results, rank_ics, ic_dates = ic_analysis(df, ret_matrix)

    # Quintile Backtest
    bt_stats, ls_annual, ls_t, ls_sharpe = quintile_backtest(df, ret_matrix)

    # Save results
    if ic_results:
        # IC results CSV
        ic_summary = pd.DataFrame([{
            "target_year": target_year,
            **{k: v for k, v in ic_results.items() if isinstance(v, (int, float, np.floating))},
        }])
        ic_path = OUT_DIR / "ic_results.csv"
        ic_summary.to_csv(ic_path, index=False)
        print(f"\nIC results saved: {ic_path}")

        # Daily IC series CSV
        ic_daily_path = OUT_DIR / "ic_daily_series.csv"
        df_ic = pd.DataFrame({
            "date": ic_dates,
            "rank_ic": rank_ics,
        })
        df_ic.to_csv(ic_daily_path, index=False)
        print(f"Daily IC series saved: {ic_daily_path}")

    # Save backtest summary
    if bt_stats:
        bt_rows = []
        for q, s in bt_stats.items():
            bt_rows.append({"quintile": q, **{k: float(v) for k, v in s.items()}})
        bt_df = pd.DataFrame(bt_rows)
        bt_path = OUT_DIR / "quintile_backtest.csv"
        bt_df.to_csv(bt_path, index=False)
        print(f"Backtest summary saved: {bt_path}")

    print("\n" + "=" * 60)
    print("Done. Results in:", OUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main(target_year=2025)
