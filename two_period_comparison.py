"""
Two-Period HiddenRatio Comparison
═══════════════════════════════════════════════════════════════════════
Compare Excel (old) vs NeoData (2026Q1) fund holdings.
Build two-period panel and compute ΔHiddenRatio.

Period 1 (Excel): Top1-5 per fund + stock→fund mapping (前十大持股基金名称)
Period 2 (NeoData): Top10 per fund (2026Q1, sorted by weight)

For each stock in each period:
  total_funds = number of funds holding it in top10
  hidden_count = total_funds - funds where stock is in their top5
  HiddenRatio = hidden_count / total_funds
  CoverageBreadth = ln(1 + total_funds)

Output: results/20260622/two_period_panel.csv
        results/20260622/two_period_comparison.png
"""

import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────
PROJECT_DIR = Path(__file__).parent
RESULTS_DIR = PROJECT_DIR / "results/20260622"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_excel_period():
    """Load Excel data: fund Top1-5 + stock→fund mapping."""
    # Fund → Top5 stocks
    df_funds = pd.read_excel(PROJECT_DIR / "data/全部基金(主代码).xlsx")
    top_cols = [df_funds.columns[i] for i in range(2, 7)]

    fund_top5 = {}  # fund_code → [stock_name1, ..., stock_name5]
    fund_names = {}

    for _, row in df_funds.iterrows():
        code = str(row["证券代码"]).replace(".OF", "").replace(".SZ", "").replace(".SH", "").replace(".JJ", "")
        name = str(row["证券名称"])
        stocks = []
        for col in top_cols:
            val = row[col]
            if pd.notna(val) and str(val).strip():
                stocks.append(str(val).strip())
        if stocks:
            fund_top5[code] = stocks
            fund_names[code] = name

    # Stock → funds holding in top10 (from 全部A股.xlsx)
    df_stocks = pd.read_excel(PROJECT_DIR / "data/全部A股.xlsx")
    stock_funds_col = df_stocks.columns[2]  # 前十大持股基金名称

    stock_to_funds = {}  # stock_name → [fund_name1, ...]
    stock_codes = {}

    for _, row in df_stocks.iterrows():
        stock_name = str(row["证券名称"]).strip()
        stock_code = str(row["证券代码"]).strip()
        funds_raw = str(row[stock_funds_col]) if pd.notna(row[stock_funds_col]) else ""

        fund_list = [f.strip() for f in funds_raw.split(",") if f.strip()]
        if fund_list:
            stock_to_funds[stock_name] = fund_list
            stock_codes[stock_name] = stock_code

    # Build stock → fund mapping from fund perspective
    # For each stock, find which funds have it in their top5
    stock_fund_top5 = defaultdict(set)  # stock_name → set of fund_codes
    for fund_code, stocks in fund_top5.items():
        for s in stocks:
            stock_fund_top5[s].add(fund_code)

    # Build stock → fund mapping from stock perspective (top10)
    # The "前十大持股基金名称" gives fund NAMES, need to match to codes
    # Create name→code lookup
    name_to_code = {}
    for code, name in fund_names.items():
        name_to_code[name] = code

    stock_fund_top10 = defaultdict(set)  # stock_name → set of fund_codes
    for stock_name, fund_names_list in stock_to_funds.items():
        for fname in fund_names_list:
            # Try exact match first
            if fname in name_to_code:
                stock_fund_top10[stock_name].add(name_to_code[fname])
            else:
                # Try partial match
                for n, c in name_to_code.items():
                    if fname in n or n in fname:
                        stock_fund_top10[stock_name].add(c)
                        break

    return fund_top5, stock_fund_top5, stock_fund_top10, stock_codes


def load_neodata_period():
    """Load NeoData current holdings and build top5/top10 mappings."""
    neo = pd.read_csv(PROJECT_DIR / "data/neodata_current_holdings.csv")

    # Filter only valid report dates
    neo = neo[neo["report_date"].isin(["2026-03-31", "2025-12-31"])].copy()

    # For each fund, sort by weight descending and take top 10
    fund_top10 = {}  # fund_code → [(stock_name, weight), ...]
    fund_top5 = {}   # fund_code → [stock_name, ...]

    for fund_code, group in neo.groupby("fund_code"):
        group = group.sort_values("weight", ascending=False)
        top10 = group.head(10)
        stocks = top10["stock_name"].tolist()
        fund_top10[fund_code] = stocks
        fund_top5[fund_code] = stocks[:5]

    # Build stock → fund mappings
    stock_fund_top5 = defaultdict(set)
    for fund_code, stocks in fund_top5.items():
        for s in stocks:
            stock_fund_top5[s].add(fund_code)

    stock_fund_top10 = defaultdict(set)
    for fund_code, stocks in fund_top10.items():
        for s in stocks:
            stock_fund_top10[s].add(fund_code)

    return fund_top5, fund_top10, stock_fund_top5, stock_fund_top10


def compute_factors(stock_fund_top5, stock_fund_top10):
    """Compute HiddenRatio and CoverageBreadth for each stock."""
    results = []
    all_stocks = set(stock_fund_top10.keys()) | set(stock_fund_top5.keys())

    for stock in all_stocks:
        funds_top10 = stock_fund_top10.get(stock, set())
        funds_top5 = stock_fund_top5.get(stock, set())

        total_funds = len(funds_top10)
        if total_funds == 0:
            continue

        hidden_count = total_funds - len(funds_top5 & funds_top10)
        hidden_ratio = hidden_count / total_funds
        breadth = math.log(1 + total_funds)

        results.append({
            "stock_name": stock,
            "total_funds": total_funds,
            "hidden_count": hidden_count,
            "HiddenRatio": hidden_ratio,
            "CoverageBreadth": breadth,
        })

    return pd.DataFrame(results)


def main():
    print("=" * 60)
    print("Two-Period HiddenRatio Comparison")
    print("=" * 60)

    # ── Period 1: Excel ─────────────────────────────────────────────
    print("\n[1] Loading Excel (old) period...")
    excel_top5, excel_stock_top5, excel_stock_top10, stock_codes = load_excel_period()
    print(f"  Funds with Top5: {len(excel_top5)}")
    print(f"  Stocks in top10 fund lists: {len(excel_stock_top10)}")
    print(f"  Stocks in top5 fund lists: {len(excel_stock_top5)}")

    df_excel = compute_factors(excel_stock_top5, excel_stock_top10)
    df_excel["period"] = "Excel (old)"
    print(f"  Stocks with factors: {len(df_excel)}")
    print(f"  HiddenRatio: mean={df_excel['HiddenRatio'].mean():.3f}, median={df_excel['HiddenRatio'].median():.3f}")
    print(f"  CoverageBreadth: mean={df_excel['CoverageBreadth'].mean():.3f}")

    # ── Period 2: NeoData ──────────────────────────────────────────
    print("\n[2] Loading NeoData (2026Q1) period...")
    neo_top5, neo_top10, neo_stock_top5, neo_stock_top10 = load_neodata_period()
    print(f"  Funds with Top10: {len(neo_top10)}")
    print(f"  Stocks in top10: {len(neo_stock_top10)}")
    print(f"  Stocks in top5: {len(neo_stock_top5)}")

    df_neo = compute_factors(neo_stock_top5, neo_stock_top10)
    df_neo["period"] = "NeoData (2026Q1)"
    print(f"  Stocks with factors: {len(df_neo)}")
    print(f"  HiddenRatio: mean={df_neo['HiddenRatio'].mean():.3f}, median={df_neo['HiddenRatio'].median():.3f}")
    print(f"  CoverageBreadth: mean={df_neo['CoverageBreadth'].mean():.3f}")

    # ── Compare periods ────────────────────────────────────────────
    print("\n[3] Comparing periods...")
    merged = df_excel.merge(df_neo, on="stock_name", suffixes=("_old", "_new"), how="inner")
    print(f"  Overlapping stocks: {len(merged)}")

    if len(merged) < 10:
        print("  ⚠️ Too few overlapping stocks! Trying name-based matching...")
        # The issue might be stock name format differences
        # Let's check what names look like in each
        print(f"  Excel sample: {df_excel['stock_name'].head(10).tolist()}")
        print(f"  NeoData sample: {df_neo['stock_name'].head(10).tolist()}")

    merged["delta_HiddenRatio"] = merged["HiddenRatio_new"] - merged["HiddenRatio_old"]
    merged["delta_Breadth"] = merged["CoverageBreadth_new"] - merged["CoverageBreadth_old"]

    print(f"\n  ΔHiddenRatio: mean={merged['delta_HiddenRatio'].mean():.4f}, std={merged['delta_HiddenRatio'].std():.4f}")
    print(f"  ΔBreadth: mean={merged['delta_Breadth'].mean():.4f}, std={merged['delta_Breadth'].std():.4f}")

    # Correlation between periods
    corr_hr = merged["HiddenRatio_old"].corr(merged["HiddenRatio_new"])
    corr_br = merged["CoverageBreadth_old"].corr(merged["CoverageBreadth_new"])
    print(f"\n  Correlation HiddenRatio (old vs new): {corr_hr:.3f}")
    print(f"  Correlation Breadth (old vs new): {corr_br:.3f}")

    # ── Determine Excel period ────────────────────────────────────
    # If we can identify specific stocks that changed positions, we might infer the Excel date
    print("\n[4] Attempting to identify Excel report period...")

    # Check stocks where HiddenRatio changed significantly
    changed = merged[merged["delta_HiddenRatio"].abs() > 0.3].sort_values("delta_HiddenRatio", key=abs, ascending=False)
    print(f"  Stocks with |ΔHiddenRatio| > 0.3: {len(changed)}")
    if len(changed) > 0:
        print(f"  Top 10 changed:")
        for _, row in changed.head(10).iterrows():
            print(f"    {row['stock_name']}: {row['HiddenRatio_old']:.2f} → {row['HiddenRatio_new']:.2f} (Δ={row['delta_HiddenRatio']:+.2f})")

    # ── Save panel ────────────────────────────────────────────────
    panel = merged[["stock_name", "total_funds_old", "hidden_count_old", "HiddenRatio_old",
                     "CoverageBreadth_old", "total_funds_new", "hidden_count_new",
                     "HiddenRatio_new", "CoverageBreadth_new", "delta_HiddenRatio",
                     "delta_Breadth"]].copy()
    panel.columns = ["stock_name", "funds_old", "hidden_old", "HR_old", "Breadth_old",
                      "funds_new", "hidden_new", "HR_new", "Breadth_new",
                      "delta_HR", "delta_Breadth"]
    panel.to_csv(RESULTS_DIR / "two_period_panel.csv", index=False, encoding="utf-8-sig")
    print(f"\n  Panel saved: {RESULTS_DIR / 'two_period_panel.csv'}")

    # ── Visualization ─────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # 1. HiddenRatio distribution comparison
    ax = axes[0, 0]
    ax.hist(df_excel["HiddenRatio"], bins=30, alpha=0.5, label="Excel (old)", color="blue", density=True)
    ax.hist(df_neo["HiddenRatio"], bins=30, alpha=0.5, label="NeoData (2026Q1)", color="red", density=True)
    ax.set_xlabel("HiddenRatio")
    ax.set_ylabel("Density")
    ax.set_title("HiddenRatio Distribution")
    ax.legend()

    # 2. CoverageBreadth distribution comparison
    ax = axes[0, 1]
    ax.hist(df_excel["CoverageBreadth"], bins=30, alpha=0.5, label="Excel (old)", color="blue", density=True)
    ax.hist(df_neo["CoverageBreadth"], bins=30, alpha=0.5, label="NeoData (2026Q1)", color="red", density=True)
    ax.set_xlabel("CoverageBreadth")
    ax.set_ylabel("Density")
    ax.set_title("CoverageBreadth Distribution")
    ax.legend()

    # 3. Scatter: old vs new HiddenRatio
    ax = axes[0, 2]
    ax.scatter(merged["HiddenRatio_old"], merged["HiddenRatio_new"], alpha=0.3, s=10)
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5)
    ax.set_xlabel("HiddenRatio (Excel old)")
    ax.set_ylabel("HiddenRatio (NeoData 2026Q1)")
    ax.set_title(f"HR Correlation: r={corr_hr:.3f}")

    # 4. Scatter: old vs new Breadth
    ax = axes[1, 0]
    ax.scatter(merged["CoverageBreadth_old"], merged["CoverageBreadth_new"], alpha=0.3, s=10)
    lim = max(merged["CoverageBreadth_old"].max(), merged["CoverageBreadth_new"].max())
    ax.plot([0, lim], [0, lim], "r--", alpha=0.5)
    ax.set_xlabel("Breadth (Excel old)")
    ax.set_ylabel("Breadth (NeoData 2026Q1)")
    ax.set_title(f"Breadth Correlation: r={corr_br:.3f}")

    # 5. ΔHiddenRatio distribution
    ax = axes[1, 1]
    ax.hist(merged["delta_HiddenRatio"], bins=40, alpha=0.7, color="green", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("ΔHiddenRatio (new - old)")
    ax.set_ylabel("Count")
    ax.set_title("Change in HiddenRatio")

    # 6. ΔBreadth distribution
    ax = axes[1, 2]
    ax.hist(merged["delta_Breadth"], bins=40, alpha=0.7, color="purple", edgecolor="black")
    ax.axvline(0, color="red", linestyle="--")
    ax.set_xlabel("ΔBreadth (new - old)")
    ax.set_ylabel("Count")
    ax.set_title("Change in CoverageBreadth")

    plt.suptitle("Two-Period Comparison: Excel (old) vs NeoData (2026Q1)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "two_period_comparison.png", dpi=150, bbox_inches="tight")
    print(f"  Figure saved: {RESULTS_DIR / 'two_period_comparison.png'}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
