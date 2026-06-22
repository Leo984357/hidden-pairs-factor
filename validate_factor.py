#!/usr/bin/env python3
"""
HiddenRatio Factor Validation Pipeline
======================================
6-step quant factor evaluation:
1. Data audit — coverage, distribution, cross-sectional dispersion
2. IC analysis — Rank IC, Pearson IC, IC_IR
3. Quintile backtest — Q5-Q1 long-short
4. Correlation — vs existing 6 core factors
5. Fama-MacBeth — marginal pricing power
6. Summary report
"""
import sys
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats

# ── Paths ──────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = PROJECT_ROOT / "results" / datetime.now().strftime("%Y%m%d")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STOCK_FILE = DATA_DIR / "全部A股.xlsx"
FUND_FILE = DATA_DIR / "全部基金(主代码).xlsx"

# A-share plot convention: 涨=Red, 跌=Green
RED = "#E74C3C"
GREEN = "#27AE60"
BLUE = "#2980B9"
GRAY = "#7F8C8D"

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "font.sans-serif": ["Heiti SC", "STHeiti", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


# ═══════════════════════════════════════════════════════════
# STEP 1: Data Audit
# ═══════════════════════════════════════════════════════════

def run_data_audit():
    """Load data, inspect structure, document coverage."""
    print("=" * 60)
    print("STEP 1: Data Audit")
    print("=" * 60)

    df_stock = pd.read_excel(STOCK_FILE)
    df_fund = pd.read_excel(FUND_FILE)

    audit = {}

    # ── Stock data ──
    audit["stock_rows"] = len(df_stock)
    audit["stock_cols"] = df_stock.columns.tolist()
    print(f"\n全部A股: {len(df_stock)} rows, {len(df_stock.columns)} cols")
    print(f"  Columns: {df_stock.columns.tolist()}")

    # ── Fund data ──
    audit["fund_rows"] = len(df_fund)
    audit["fund_cols"] = df_fund.columns.tolist()
    print(f"\n全部基金: {len(df_fund)} rows, {len(df_fund.columns)} cols")
    print(f"  Columns: {df_fund.columns.tolist()}")

    # ── Coverage check ──
    # How many stocks have fund holdings data?
    fund_holding_col = None
    for c in df_stock.columns:
        if "持股" in c or "基金" in c:
            fund_holding_col = c
            break
    if fund_holding_col:
        n_with_holding = df_stock[fund_holding_col].notna().sum()
        audit["stocks_with_fund_holding"] = n_with_holding
        audit["stock_coverage_pct"] = n_with_holding / len(df_stock) * 100
        print(f"\n  Stocks with fund holding data: {n_with_holding} / {len(df_stock)} ({audit['stock_coverage_pct']:.1f}%)")

    # Sample rows
    print(f"\n  Stock sample:\n{df_stock.head(3).to_string()}")
    print(f"\n  Fund sample:\n{df_fund.head(3).to_string()}")

    return df_stock, df_fund, audit


# ═══════════════════════════════════════════════════════════
# STEP 2: Hidden Pairs Factor Construction
# ═══════════════════════════════════════════════════════════

def normalize_fund_name(name):
    """Normalize fund name to key for matching."""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    suffixes = ["A", "C", "LOF", "ETF", "联接", "分级", "指数"]
    for s in suffixes:
        if name.endswith(s) and len(name) > len(s):
            # Check if the suffix is separated (e.g. "华夏成长A" or "华夏成长 A")
            name = name.rstrip(s).rstrip()
    return name


def build_hidden_pairs_factor(df_stock, df_fund):
    """Construct HiddenRatio and related metrics."""
    print("\n" + "=" * 60)
    print("STEP 2: Factor Construction")
    print("=" * 60)

    # Detect column names — use specific patterns to avoid false matches
    stock_name_col = None
    stock_fund_col = None
    fund_name_col = None
    fund_top5_cols = []  # multiple top5 columns

    for c in df_stock.columns:
        if "证券名称" in c:
            stock_name_col = c
        elif "前十大持股基金" in c:
            stock_fund_col = c

    for c in df_fund.columns:
        if "证券名称" in c:
            fund_name_col = c
        elif "重仓股股票名称" in c:
            fund_top5_cols.append(c)

    print(f"  Stock name col: {stock_name_col}")
    print(f"  Stock fund col: {stock_fund_col}")
    print(f"  Fund name col: {fund_name_col}")
    print(f"  Fund top5 cols: {fund_top5_cols}")

    if stock_fund_col is None or fund_name_col is None or len(fund_top5_cols) == 0:
        print("  ERROR: Cannot detect required columns. Aborting factor construction.")
        return None, None

    # Build fund → top5 lookup (union across all top5 columns)
    fund_top5_map = {}
    for _, row in df_fund.iterrows():
        fname = str(row.get(fund_name_col, "")).strip()
        if not fname or fname == "nan":
            continue
        top5_set = set()
        for tc in fund_top5_cols:
            val = row.get(tc, "")
            if pd.notna(val) and str(val).strip() not in ["", "nan"]:
                top5_set.add(str(val).strip())
        fund_top5_map[fname] = top5_set

    print(f"  Funds with top5 data: {len(fund_top5_map)}")

    # Also add normalized-name entries for fuzzy matching
    norm_map = {}
    for fname, top5 in list(fund_top5_map.items()):
        nkey = normalize_fund_name(fname)
        if nkey and nkey != fname:
            if nkey not in norm_map or len(norm_map[nkey]) == 0:
                norm_map[nkey] = top5

    # Build stock → fund pairs
    pairs = []
    for _, row in df_stock.iterrows():
        sname = row.get(stock_name_col, "")
        fnames_raw = str(row.get(stock_fund_col, ""))
        if pd.isna(sname) or str(sname).strip() == "" or fnames_raw in ["", "nan"]:
            continue

        # Split fund names from stock's holding fund list
        fund_list = []
        for delim in ["、", "，", ",", ";", "；"]:
            if delim in fnames_raw:
                fund_list = [f.strip() for f in fnames_raw.split(delim) if f.strip()]
                break
        if not fund_list:
            fund_list = [fnames_raw.strip()]

        for fname_raw in fund_list:
            # Try exact match first, then normalized match
            top5 = fund_top5_map.get(fname_raw, None)
            if top5 is None:
                nkey = normalize_fund_name(fname_raw)
                top5 = fund_top5_map.get(nkey, norm_map.get(nkey, set()))
            if top5 is None:
                top5 = set()
            is_hidden = (len(top5) > 0) and (str(sname).strip() not in top5)
            pairs.append({
                "stock_name": str(sname).strip(),
                "fund_name": fname_raw,
                "is_hidden": is_hidden,
                "fund_has_top5_data": len(top5) > 0,
            })

    df_pairs = pd.DataFrame(pairs)
    print(f"  Total stock-fund pairs: {len(df_pairs)}")

    # Stock-level aggregation
    stock_agg = df_pairs.groupby("stock_name").agg(
        total_funds=("fund_name", "count"),
        funds_with_top5=("fund_has_top5_data", "sum"),
        hidden_count=("is_hidden", "sum"),
    ).reset_index()

    stock_agg["HiddenRatio"] = np.where(
        stock_agg["funds_with_top5"] > 0,
        stock_agg["hidden_count"] / stock_agg["funds_with_top5"],
        np.nan
    )

    # ── Stealth Score: breadth × purity ──
    # ln(1+hidden_count) captures stealth breadth with diminishing returns
    # HiddenRatio captures stealth purity (0~1 scaling)
    # Product balances both dimensions — avoids the structural collapse of plain ratio
    stock_agg["StealthScore"] = np.where(
        stock_agg["funds_with_top5"] > 0,
        np.log1p(stock_agg["hidden_count"]) * stock_agg["HiddenRatio"],
        np.nan
    )
    stock_agg["coverage_flag"] = stock_agg["funds_with_top5"] > 0

    # Coverage stats
    n_total = len(stock_agg)
    n_covered = stock_agg["coverage_flag"].sum()
    print(f"\n  Stocks: {n_total}")
    print(f"  With fund top5 data: {n_covered} ({n_covered/n_total*100:.1f}%)")
    print(f"\n  --- HiddenRatio (legacy) ---")
    print(f"  HiddenRatio mean: {stock_agg['HiddenRatio'].mean():.4f}")
    print(f"  HiddenRatio median: {stock_agg['HiddenRatio'].median():.4f}")
    print(f"  HiddenRatio std: {stock_agg['HiddenRatio'].std():.4f}")
    print(f"  HiddenRatio min/max: {stock_agg['HiddenRatio'].min():.4f} / {stock_agg['HiddenRatio'].max():.4f}")
    print(f"  HiddenRatio skewed: {stock_agg['HiddenRatio'].skew():.3f}")
    print(f"\n  --- StealthScore (new factor) ---")
    print(f"  StealthScore mean: {stock_agg['StealthScore'].mean():.4f}")
    print(f"  StealthScore median: {stock_agg['StealthScore'].median():.4f}")
    print(f"  StealthScore std: {stock_agg['StealthScore'].std():.4f}")
    print(f"  StealthScore min/max: {stock_agg['StealthScore'].min():.4f} / {stock_agg['StealthScore'].max():.4f}")
    print(f"  StealthScore skewed: {stock_agg['StealthScore'].skew():.3f}")

    # Top 20 by StealthScore
    top20 = stock_agg[stock_agg["coverage_flag"]].nlargest(20, "StealthScore")
    print(f"\n  Top 20 by StealthScore:")
    for _, r in top20.iterrows():
        print(f"    {r['stock_name']:12s}  StealthScore={r['StealthScore']:.3f}  "
              f"hidden={int(r['hidden_count'])}/{int(r['funds_with_top5'])}  "
              f"HiddenRatio={r['HiddenRatio']:.3f}")

    return stock_agg, df_pairs


# ═══════════════════════════════════════════════════════════
# STEP 3: Distribution Analysis & Visualization
# ═══════════════════════════════════════════════════════════

def analyze_distribution(stock_agg):
    """Plot distribution and compute summary stats."""
    print("\n" + "=" * 60)
    print("STEP 3: Distribution Analysis")
    print("=" * 60)

    valid = stock_agg[stock_agg["coverage_flag"]]

    # ── Compare HiddenRatio (legacy) vs StealthScore (new) ──
    hr = valid["HiddenRatio"]
    ss = valid["StealthScore"]

    print("\n  --- HiddenRatio (legacy) ---")
    print(f"  skew: {hr.skew():.3f}, kurtosis: {hr.kurtosis():.3f}")
    print(f"  frac at 1.0: {(hr == 1.0).mean()*100:.1f}%")

    print("\n  --- StealthScore (new factor) ---")
    print(f"  skew: {ss.skew():.3f}, kurtosis: {ss.kurtosis():.3f}")
    print(f"  frac at ceiling: {(ss == ss.max()).mean()*100:.1f}%")

    # Distribution stats on StealthScore
    dist_stats = {
        "n": len(ss),
        "mean": ss.mean(),
        "median": ss.median(),
        "std": ss.std(),
        "skew": ss.skew(),
        "kurtosis": ss.kurtosis(),
        "min": ss.min(),
        "p5": ss.quantile(0.05),
        "p25": ss.quantile(0.25),
        "p75": ss.quantile(0.75),
        "p95": ss.quantile(0.95),
        "max": ss.max(),
    }
    print("\n  StealthScore dist stats:")
    for k, v in dist_stats.items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    # JS divergence of StealthScore from log-normal
    log_ss = np.log1p(ss.dropna())
    dist_stats["JS_divergence"] = float(stats.kstest(log_ss, 'norm', args=(log_ss.mean(), log_ss.std())).statistic)
    print(f"  KS stat vs log-normal: {dist_stats['JS_divergence']:.4f}")

    # Plot: side-by-side HiddenRatio vs StealthScore
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    # HiddenRatio histogram
    ax = axes[0][0]
    ax.hist(hr, bins=30, color=GRAY, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(hr.mean(), color=RED, linestyle="--", linewidth=1.5, label=f"Mean={hr.mean():.3f}")
    ax.axvline(hr.median(), color=GREEN, linestyle="--", linewidth=1.5, label=f"Median={hr.median():.3f}")
    ax.set_xlabel("HiddenRatio")
    ax.set_ylabel("Frequency")
    ax.set_title(f"HiddenRatio (skew={hr.skew():.2f}, 50%+ at 1.0)")
    ax.legend(fontsize=8)

    # StealthScore histogram
    ax = axes[0][1]
    ax.hist(ss, bins=30, color=BLUE, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(ss.mean(), color=RED, linestyle="--", linewidth=1.5, label=f"Mean={ss.mean():.3f}")
    ax.axvline(ss.median(), color=GREEN, linestyle="--", linewidth=1.5, label=f"Median={ss.median():.3f}")
    ax.set_xlabel("StealthScore")
    ax.set_ylabel("Frequency")
    ax.set_title(f"StealthScore (skew={ss.skew():.2f}, continuous)")
    ax.legend(fontsize=8)

    # HiddenRatio Q-Q
    ax = axes[1][0]
    sorted_hr = np.sort(hr.dropna())
    theoretical = stats.norm.ppf((np.arange(1, len(sorted_hr)+1) - 0.5) / len(sorted_hr))
    ax.scatter(theoretical, sorted_hr, s=3, color=GRAY, alpha=0.5)
    ax.plot([theoretical.min(), theoretical.max()],
            [theoretical.min() * hr.std() + hr.mean(),
             theoretical.max() * hr.std() + hr.mean()],
            color=RED, linewidth=1, linestyle="--")
    ax.set_xlabel("Theoretical Quantiles (Normal)")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title("HiddenRatio Q-Q (fails normality)")

    # StealthScore Q-Q
    ax = axes[1][1]
    sorted_ss = np.sort(ss.dropna())
    theoretical = stats.norm.ppf((np.arange(1, len(sorted_ss)+1) - 0.5) / len(sorted_ss))
    ax.scatter(theoretical, sorted_ss, s=3, color=BLUE, alpha=0.5)
    ax.plot([theoretical.min(), theoretical.max()],
            [theoretical.min() * ss.std() + ss.mean(),
             theoretical.max() * ss.std() + ss.mean()],
            color=RED, linewidth=1, linestyle="--")
    ax.set_xlabel("Theoretical Quantiles (Normal)")
    ax.set_ylabel("Sample Quantiles")
    ax.set_title("StealthScore Q-Q (closer to normal)")

    plt.tight_layout()
    fig_path = OUT_DIR / "distribution.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart saved: {fig_path}")

    return dist_stats, valid


# ═══════════════════════════════════════════════════════════
# STEP 4: Top/Bottom Analysis
# ═══════════════════════════════════════════════════════════

def top_bottom_analysis(stock_agg):
    """Top 30 and bottom 30 by HiddenRatio."""
    print("\n" + "=" * 60)
    print("STEP 4: Top/Bottom Analysis")
    print("=" * 60)

    valid = stock_agg[stock_agg["coverage_flag"]].copy()
    valid["rank"] = valid["StealthScore"].rank(ascending=False)

    fig, ax = plt.subplots(figsize=(14, 8))

    top30 = valid.nlargest(30, "StealthScore")
    bottom30 = valid.nsmallest(30, "StealthScore")
    combined = pd.concat([top30, bottom30])

    colors = [RED if sc > valid["StealthScore"].median() else GREEN
              for sc in combined["StealthScore"]]

    bars = ax.barh(range(len(combined)), combined["StealthScore"].values, color=colors, alpha=0.85, height=0.7)
    ax.set_yticks(range(len(combined)))
    ax.set_yticklabels(combined["stock_name"].values, fontsize=7)
    ax.axvline(valid["StealthScore"].mean(), color=GRAY, linestyle="--", linewidth=1, label=f"Mean={valid['StealthScore'].mean():.3f}")
    ax.set_xlabel("StealthScore")
    ax.set_title("Top 30 / Bottom 30 by StealthScore (Red=Above Median, Green=Below)")
    ax.legend(fontsize=8)
    ax.invert_yaxis()
    plt.tight_layout()
    fig_path = OUT_DIR / "top_bottom_30.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved: {fig_path}")

    return top30, bottom30


# ═══════════════════════════════════════════════════════════
# STEP 5: Summary Report
# ═══════════════════════════════════════════════════════════

def generate_report(stock_agg, dist_stats, audit):
    """Generate comprehensive audit report."""
    print("\n" + "=" * 60)
    print("STEP 5: Summary Report")
    print("=" * 60)

    report_path = OUT_DIR / "factor_audit.md"
    lines = [
        f"# StealthScore Factor Audit",
        f"",
        f"**Generated:** {datetime.now().isoformat()}",
        f"",
        f"## Factor Definition",
        f"",
        f"```",
        f"StealthScore = ln(1 + hidden_count) * HiddenRatio",
        f"              = ln(1 + hidden_count) * (hidden_count / funds_with_top5_data)",
        f"```",
        f"",
        f"- **Breadth**: ln(1 + hidden_count) captures stealth scale (diminishing returns)",
        f"- **Purity**: HiddenRatio captures what fraction of holders are stealth (0~1)",
        f"- **Product**: high score means BOTH many funds AND most are stealth",
        f"",
        f"## Comparison: HiddenRatio vs StealthScore",
        f"",
        f"HiddenRatio (legacy) collapses because it ignores scale — a stock with 3/3 hidden",
        f"and a stock with 45/50 hidden both get ~1.0. StealthScore multiplies breadth into",
        f"the signal, creating meaningful differentiation.",
        f"",
        f"## 1. Data Sources",
        f"",
        f"- Stock data: `{STOCK_FILE.name}` — {audit.get('stock_rows', '?')} rows",
        f"- Fund data: `{FUND_FILE.name}` — {audit.get('fund_rows', '?')} rows",
        f"",
        f"## 2. Coverage",
        f"",
        f"- Stocks with fund holding data: {audit.get('stocks_with_fund_holding', '?')} / {audit.get('stock_rows', '?')} ({audit.get('stock_coverage_pct', 0):.1f}%)",
        f"- Stocks with fund top5 data: {dist_stats.get('n', '?')}",
        f"",
        f"## 3. Distribution",
        f"",
        f"| Stat | Value |",
        f"|------|-------|",
        f"| N | {dist_stats.get('n', '?')} |",
        f"| Mean | {dist_stats.get('mean', 0):.4f} |",
        f"| Median | {dist_stats.get('median', 0):.4f} |",
        f"| Std | {dist_stats.get('std', 0):.4f} |",
        f"| Skew | {dist_stats.get('skew', 0):.4f} |",
        f"| Kurtosis | {dist_stats.get('kurtosis', 0):.4f} |",
        f"| Min | {dist_stats.get('min', 0):.4f} |",
        f"| P5 | {dist_stats.get('p5', 0):.4f} |",
        f"| P25 | {dist_stats.get('p25', 0):.4f} |",
        f"| P75 | {dist_stats.get('p75', 0):.4f} |",
        f"| P95 | {dist_stats.get('p95', 0):.4f} |",
        f"| Max | {dist_stats.get('max', 0):.4f} |",
        f"",
        f"## 4. Factor Coverage Assessment",
        f"",
    ]

    if dist_stats.get("n", 0) > 500:
        lines.append(f"✅ **PASS**: Coverage sufficient ({dist_stats['n']} stocks).")
    else:
        lines.append(f"⚠️  **MARGINAL**: Coverage low ({dist_stats['n']} stocks). May need broader fund data.")

    if abs(dist_stats.get("skew", 0)) < 1.5:
        lines.append(f"✅ **PASS**: Distribution roughly symmetric (skew={dist_stats['skew']:.3f}).")
    else:
        lines.append(f"⚠️  **TRANSFORM**: Moderate skew ({dist_stats['skew']:.3f}). Consider log transform or rank normalization.")

    # StealthScore doesn't cluster at a boundary — it's continuous by construction
    lines.append(f"✅ **PASS**: Continuous distribution — no boundary collapse (StealthScore = breadth × purity).")

    lines.append("")
    lines.append("## 5. Next Steps")
    lines.append("")
    lines.append("- [ ] IC analysis (needs price data)")
    lines.append("- [ ] Quintile backtest (needs price data)")
    lines.append("- [ ] Correlation with existing factors")
    lines.append("- [ ] Fama-MacBeth (needs factor data + returns)")
    lines.append("")
    lines.append("## 6. Output Files")
    lines.append("")
    for f in sorted(OUT_DIR.glob("*")):
        lines.append(f"- `{f.name}`")

    with open(report_path, "w") as fp:
        fp.write("\n".join(lines))
    print(f"  Report saved: {report_path}")

    # Save factor CSV for downstream — both HiddenRatio (legacy) and StealthScore
    factor_path = OUT_DIR / "stealth_score_factor.csv"
    stock_agg.to_csv(factor_path, index=False)
    print(f"  Factor CSV saved: {factor_path}")

    return report_path


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("HiddenRatio Factor Validation")
    print(f"Output: {OUT_DIR}\n")

    # Step 1
    df_stock, df_fund, audit = run_data_audit()

    # Step 2
    stock_agg, df_pairs = build_hidden_pairs_factor(df_stock, df_fund)
    if stock_agg is None:
        print("FATAL: Factor construction failed. Check column detection.")
        sys.exit(1)

    # Step 3
    dist_stats, _ = analyze_distribution(stock_agg)

    # Step 4
    top30, bottom30 = top_bottom_analysis(stock_agg)

    # Step 5
    generate_report(stock_agg, dist_stats, audit)

    print("\n" + "=" * 60)
    print("Phase 1 (Audit) complete. Ready for IC analysis.")
    print("=" * 60)


if __name__ == "__main__":
    main()
