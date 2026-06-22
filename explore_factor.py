#!/usr/bin/env python3
"""
StealthScore Factor — Deep Exploration & Variant Mining
========================================================
Tests beyond the baseline IC analysis:
  1. Monthly IC (reduce noise)
  2. Alternative factor variants (sqrt, raw count, etc.)
  3. Size correlation & size-neutral IC
  4. Industry distribution check
  5. Second-order factors (concentration, breadth-only)
"""
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ══════════════════ Paths ══════════════════
PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_ROOT = PROJECT_ROOT / "results"
IN_DIR = sorted(RESULTS_ROOT.glob("20*"))[-1]
OUT_DIR = IN_DIR

CSMAR_ROOT = Path("/Users/leolee/Desktop/genai_china_replication")
DAILY_RETURNS = CSMAR_ROOT / "data/processed/daily_returns_cn_all.csv"
LISTING_MASTER = CSMAR_ROOT / "data/raw/csmar/listing_master_by_year_clean.csv"

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

# ═══════════════════════ Load Data ═══════════════════════

def load_all():
    print("Loading factor + returns + size...")

    # Factor
    df = pd.read_csv(FACTOR_FILE)
    df = df[df["coverage_flag"]].copy()

    # Name → code mapping
    master = pd.read_csv(LISTING_MASTER)
    master_2024 = master[master["year"] == 2024].copy()
    master_2024["name_clean"] = master_2024["ShortName"].str.replace("*", "").str.strip()
    name_to_id = dict(zip(master_2024["name_clean"], master_2024["stock_id"]))

    df["name_clean"] = df["stock_name"].str.replace("*", "").str.strip()
    df["stock_id"] = df["name_clean"].map(name_to_id)
    df = df.dropna(subset=["stock_id"])

    print(f"  {len(df)} stocks after name mapping")

    # Daily returns + market cap
    stock_set = set(df["stock_id"])
    frames = []
    for chunk in pd.read_csv(DAILY_RETURNS, chunksize=200000, dtype={"stock_id": str}):
        chunk = chunk[chunk["stock_id"].isin(stock_set)]
        if len(chunk) == 0:
            continue
        chunk["trade_date"] = pd.to_datetime(chunk["trade_date"])
        chunk = chunk[chunk["trade_date"].dt.year == 2025]
        if len(chunk) == 0:
            continue
        frames.append(chunk[["stock_id", "trade_date", "ret", "mkt_cap_float"]])
    df_ret = pd.concat(frames, ignore_index=True)
    del frames

    # Pivot returns
    ret_matrix = df_ret.pivot_table(index="stock_id", columns="trade_date", values="ret", aggfunc="first")
    dates_sorted = sorted(ret_matrix.columns)
    ret_matrix = ret_matrix[dates_sorted]

    # Get size (last available mkt_cap for each stock)
    size = df_ret.groupby("stock_id")["mkt_cap_float"].last()
    size.name = "mkt_cap"

    # Merge size into factor df
    df = df.merge(size, left_on="stock_id", right_index=True, how="left")

    print(f"  Return matrix: {ret_matrix.shape}")
    print(f"  Size data: {size.notna().sum()} stocks")

    return df, ret_matrix


# ═══════════════════════ Factor Variants ═══════════════════════

def build_variants(df):
    """Construct alternative factor definitions."""
    print("\n" + "=" * 60)
    print("Factor Variants")
    print("=" * 60)

    v = df.copy()

    # 1. StealthScore (baseline)
    v["F_stealth"] = v["StealthScore"]

    # 2. Raw hidden_count (breadth only)
    v["F_hidden_cnt"] = v["hidden_count"]

    # 3. Log hidden_count (breadth, diminishing)
    v["F_hidden_log"] = np.log1p(v["hidden_count"])

    # 4. HiddenRatio * sqrt(count) — alternative curvature
    v["F_stealth_sqrt"] = np.sqrt(v["hidden_count"]) * v["HiddenRatio"]

    # 5. HiddenRatio with minimum threshold (≥3 funds)
    v["F_hr_thresh"] = np.where(
        v["hidden_count"] >= 3,
        v["HiddenRatio"],
        np.nan
    )

    # 6. total_funds (sheer institutional breadth)
    v["F_total_funds"] = v["total_funds"]

    # 7. Log total_funds
    v["F_total_log"] = np.log1p(v["total_funds"])

    # 8. hidden_count / total_funds (hidden as fraction of ALL holders)
    v["F_hidden_pct_all"] = np.where(
        v["total_funds"] > 0,
        v["hidden_count"] / v["total_funds"],
        np.nan
    )

    # 9. Composite: hidden_count × HiddenRatio × log(total_funds)
    v["F_composite"] = (v["hidden_count"] * v["HiddenRatio"] * np.log1p(v["total_funds"]))

    # 10. "Surprise" — HiddenRatio weighted by total funds
    v["F_surprise"] = v["HiddenRatio"] * np.log1p(v["total_funds"])

    # 11. Z-score within industry? Skip for now (need industry data)
    # 12. Decile rank of StealthScore (non-parametric)
    v["F_rank"] = v["StealthScore"].rank(pct=True)

    variants = [c for c in v.columns if c.startswith("F_")]
    print(f"  Constructed {len(variants)} factor variants")
    for var in variants:
        valid = v[var].notna().sum()
        print(f"    {var:25s}  valid={valid:5d}  mean={v[var].mean():.4f}  std={v[var].std():.4f}")

    return v, variants


# ═══════════════════════ Correlation Matrix ═══════════════════════

def correlation_analysis(df, variants):
    """Correlation between variants and with size."""
    print("\n" + "=" * 60)
    print("Correlation Analysis")
    print("=" * 60)

    corr_cols = variants + ["mkt_cap", "hidden_count", "total_funds"]
    corr_cols = [c for c in corr_cols if c in df.columns]
    corr = df[corr_cols].corr()

    # Print key correlations with size
    print("\n  Correlation with log(mkt_cap):")
    df["log_mkt_cap"] = np.log(df["mkt_cap"] + 1)
    for var in variants:
        r = df[var].corr(df["log_mkt_cap"])
        print(f"    {var:25s}  r = {r:+.4f}")

    print(f"\n  StealthScore vs hidden_count: r = {df['StealthScore'].corr(df['hidden_count']):.4f}")
    print(f"  StealthScore vs HiddenRatio:  r = {df['StealthScore'].corr(df['HiddenRatio']):.4f}")
    print(f"  StealthScore vs total_funds:  r = {df['StealthScore'].corr(df['total_funds']):.4f}")

    return corr


# ═══════════════════════ IC Analysis (Monthly) ═══════════════════════

def monthly_ic(df, ret_matrix, variants):
    """Monthly RankIC — aggregate daily returns to monthly, compute IC."""
    print("\n" + "=" * 60)
    print("Monthly IC Analysis")
    print("=" * 60)

    aligned = df[df["stock_id"].isin(ret_matrix.index)].set_index("stock_id")
    common_ids = aligned.index.intersection(ret_matrix.index)
    aligned = aligned.loc[common_ids]
    ret_monthly = ret_matrix.loc[common_ids]

    # Resample returns to monthly (manual, cross-pandas-version compatible)
    ret_monthly.columns = pd.to_datetime(ret_monthly.columns)
    monthly_groups = ret_monthly.T.groupby(pd.Grouper(freq='ME')).prod()
    monthly_rets = monthly_groups.T - 1

    months = sorted(monthly_rets.columns)
    print(f"  Monthly returns: {len(months)} months ({months[0].date()} ~ {months[-1].date()})")

    results = {}
    for var in variants:
        fv = aligned[var].values
        valid_mask = ~np.isnan(fv)
        if valid_mask.sum() < 30:
            continue

        fv_clean = fv[valid_mask]
        monthly_ics = []
        for m in months:
            rets = monthly_rets.loc[common_ids[valid_mask], m].values
            rets_valid = ~np.isnan(rets)
            if rets_valid.sum() < 30:
                continue
            # Guard against constant arrays (all same return)
            if np.std(rets[rets_valid]) < 1e-10:
                continue
            ic, _ = stats.spearmanr(fv_clean[rets_valid], rets[rets_valid])
            if np.isnan(ic):
                continue
            monthly_ics.append(ic)

        monthly_ics = np.array(monthly_ics)
        if len(monthly_ics) < 3:
            continue

        results[var] = {
            "mean": monthly_ics.mean(),
            "std": monthly_ics.std(),
            "ir": monthly_ics.mean() / (monthly_ics.std() + 1e-8),
            "tstat": monthly_ics.mean() / (monthly_ics.std() / np.sqrt(len(monthly_ics))),
            "pos_ratio": (monthly_ics > 0).mean(),
            "n_months": len(monthly_ics),
        }

    print(f"\n  {'Variant':25s} {'MeanIC':>8s} {'IR':>7s} {'t-stat':>7s} {'Pos%':>6s} {'N':>4s}")
    print(f"  {'-'*25} {'-'*8} {'-'*7} {'-'*7} {'-'*6} {'-'*4}")
    for var, r in sorted(results.items(), key=lambda x: abs(x[1]["ir"]), reverse=True):
        sign = "+" if r["mean"] > 0 else ""
        print(f"  {var:25s} {sign}{r['mean']:7.4f} {r['ir']:7.3f} {r['tstat']:7.2f} {r['pos_ratio']:5.1%} {r['n_months']:4d}")

    return results


# ═══════════════════════ Size-Neutral IC ═══════════════════════

def size_neutral_ic(df, ret_matrix, variants):
    """IC analysis after orthogonalizing factor to log(size)."""
    print("\n" + "=" * 60)
    print("Size-Neutral IC (Daily, 2025)")
    print("=" * 60)

    aligned = df[df["stock_id"].isin(ret_matrix.index)].copy()
    aligned["log_mkt_cap"] = np.log(aligned["mkt_cap"] + 1)
    aligned = aligned.dropna(subset=["log_mkt_cap"]).set_index("stock_id")

    common_ids = aligned.index.intersection(ret_matrix.index)
    aligned = aligned.loc[common_ids]

    results = {}
    for var in variants:
        fv = aligned[var]
        sz = aligned["log_mkt_cap"]
        valid = fv.notna() & sz.notna()
        if valid.sum() < 30:
            continue

        # Orthogonalize: residual from regressing factor on log(size)
        fv_v = fv[valid].values
        sz_v = sz[valid].values
        # Simple OLS
        X = np.column_stack([np.ones(len(sz_v)), sz_v])
        beta = np.linalg.lstsq(X, fv_v, rcond=None)[0]
        fv_neutral = fv_v - (X @ beta)

        daily_ics = []
        for date in ret_matrix.columns:
            rets = ret_matrix.loc[common_ids[valid], date].values
            rets_v = ~np.isnan(rets)
            if rets_v.sum() < 30:
                continue
            ic, _ = stats.spearmanr(fv_neutral[rets_v], rets[rets_v])
            daily_ics.append(ic)

        daily_ics = np.array(daily_ics)
        if len(daily_ics) < 10:
            continue

        results[var] = {
            "mean": daily_ics.mean(),
            "ir": daily_ics.mean() / (daily_ics.std() + 1e-8),
            "tstat": daily_ics.mean() / (daily_ics.std() / np.sqrt(len(daily_ics))),
        }

    print(f"\n  {'Variant':25s} {'MeanIC':>8s} {'IR':>7s} {'t-stat':>7s}")
    print(f"  {'-'*25} {'-'*8} {'-'*7} {'-'*7}")
    for var, r in sorted(results.items(), key=lambda x: abs(x[1]["ir"]), reverse=True):
        sign = "+" if r["mean"] > 0 else ""
        print(f"  {var:25s} {sign}{r['mean']:7.4f} {r['ir']:7.3f} {r['tstat']:7.2f}")

    return results


# ═══════════════════════ Top/Bottom Characteristics ═══════════════════════

def examine_extremes(df):
    """Examine characteristics of top decile vs bottom decile stocks."""
    print("\n" + "=" * 60)
    print("Top/Bottom Decile Characteristics")
    print("=" * 60)

    df["decile"] = pd.qcut(df["StealthScore"].rank(method="first"), 10, labels=False) + 1
    top = df[df["decile"] == 10]
    bot = df[df["decile"] == 1]
    mid = df[(df["decile"] >= 5) & (df["decile"] <= 6)]

    for label, subset in [("Top (D10)", top), ("Mid (D5-6)", mid), ("Bot (D1)", bot)]:
        sz = subset["mkt_cap"].dropna() / 1e8  # in 亿元
        if len(sz) == 0:
            print(f"\n  {label}: no size data")
            continue
        print(f"\n  {label}: {len(subset)} stocks")
        print(f"    Size P50: ¥{sz.median():.0f}亿, P25: ¥{sz.quantile(0.25):.0f}亿, P75: ¥{sz.quantile(0.75):.0f}亿")
        print(f"    StealthScore: mean={subset['StealthScore'].mean():.3f}")
        print(f"    hidden_count: mean={subset['hidden_count'].mean():.1f}, median={subset['hidden_count'].median():.0f}")
        print(f"    HiddenRatio:  mean={subset['HiddenRatio'].mean():.3f}")
        print(f"    total_funds:  mean={subset['total_funds'].mean():.1f}")

    # StealthScore by size decile
    df["size_decile"] = pd.qcut(df["mkt_cap"].rank(method="first"), 10, labels=False) + 1
    print(f"\n  StealthScore by size decile:")
    for d in range(1, 11):
        sub = df[df["size_decile"] == d]
        sz_median = sub["mkt_cap"].median() / 1e8
        print(f"    Size D{d} (¥{sz_median:.0f}亿): StealthScore={sub['StealthScore'].mean():.3f}, "
              f"hidden_count={sub['hidden_count'].mean():.1f}, n={len(sub)}")


# ═══════════════════════ Plot ═══════════════════════

def plot_variant_comparison(monthly_results, size_neutral_results, df):
    """Create comparison charts for factor variants."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Monthly IC comparison
    ax = axes[0]
    vars_sorted = sorted(monthly_results.items(), key=lambda x: abs(x[1]["ir"]), reverse=True)
    names = [v[0].replace("F_", "") for v in vars_sorted]
    irs = [v[1]["ir"] for v in vars_sorted]
    colors = [RED if ir > 0 else GREEN for ir in irs]
    ax.barh(range(len(names)), irs, color=colors, alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Monthly IC IR")
    ax.set_title("Factor Variants — Monthly IC Information Ratio")

    # Size-neutral IC comparison
    ax = axes[1]
    vars_sorted2 = sorted(size_neutral_results.items(), key=lambda x: abs(x[1]["ir"]), reverse=True)
    names2 = [v[0].replace("F_", "") for v in vars_sorted2]
    irs2 = [v[1]["ir"] for v in vars_sorted2]
    colors2 = [RED if ir > 0 else GREEN for ir in irs2]
    ax.barh(range(len(names2)), irs2, color=colors2, alpha=0.8)
    ax.set_yticks(range(len(names2)))
    ax.set_yticklabels(names2, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Daily Size-Neutral IC IR")
    ax.set_title("Size-Orthogonalized — Daily IC IR")

    plt.tight_layout()
    path = OUT_DIR / "variant_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart: {path}")

    # StealthScore vs Size scatter
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    df_scatter = df.dropna(subset=["mkt_cap", "StealthScore"])
    cap_log = np.log(df_scatter["mkt_cap"] / 1e8)
    ax2.scatter(cap_log, df_scatter["StealthScore"], s=2, alpha=0.3, color=BLUE)
    ax2.set_xlabel("log(Market Cap / 100M)")
    ax2.set_ylabel("StealthScore")
    ax2.set_title(f"StealthScore vs Size (r={df_scatter['StealthScore'].corr(cap_log):.3f})")
    plt.tight_layout()
    path2 = OUT_DIR / "stealthscore_vs_size.png"
    fig2.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart: {path2}")

    return df


# ═══════════════════════ Main ═══════════════════════

def main():
    print("=" * 60)
    print("StealthScore — Deep Exploration & Variant Mining")
    print("=" * 60)

    # Load data
    df, ret_matrix = load_all()

    # Build variants
    df, variants = build_variants(df)

    # Correlation analysis
    corr = correlation_analysis(df, variants)

    # Monthly IC
    monthly = monthly_ic(df, ret_matrix, variants)

    # Size-neutral IC
    sz_neutral = size_neutral_ic(df, ret_matrix, variants)

    # Extreme characteristics
    examine_extremes(df)

    # Plot
    plot_variant_comparison(monthly, sz_neutral, df)

    # Save correlation matrix
    corr_path = OUT_DIR / "factor_correlation.csv"
    corr.to_csv(corr_path)
    print(f"\nCorrelation matrix saved: {corr_path}")

    print("\n" + "=" * 60)
    print("Exploration complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
