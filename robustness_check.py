#!/usr/bin/env python3
"""
StealthScore — Comprehensive Robustness Check
==============================================
  1. Multi-year IC (2016-2025) for top factor variants
  2. Industry-neutral IC
  3. Forward horizon analysis (1m, 3m, 6m, 12m)
  4. Different hidden_count thresholds (2, 3, 4, 5)
  5. Summary: which variant works best, under what conditions
"""
import sys
from pathlib import Path
from collections import defaultdict

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
PANEL_EF = CSMAR_ROOT / "data/processed/panel_Ef_daily.csv"
LISTING_MASTER = CSMAR_ROOT / "data/raw/csmar/listing_master_by_year_clean.csv"
FACTOR_FILE = IN_DIR / "stealth_score_factor.csv"

RED = "#E74C3C"
GREEN = "#27AE60"
BLUE = "#2980B9"

plt.rcParams.update({
    "figure.dpi": 150, "font.size": 10,
    "font.sans-serif": ["Heiti SC", "STHeiti", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
})


# ═══════════════════════ Load Data ═══════════════════════

def load_all():
    print("=" * 60)
    print("Loading data...")
    
    # Factor
    df = pd.read_csv(FACTOR_FILE)
    df = df[df["coverage_flag"]].copy()
    
    # Name mapping
    master = pd.read_csv(LISTING_MASTER)
    master_2024 = master[master["year"] == 2024].copy()
    master_2024["name_clean"] = master_2024["ShortName"].str.replace("*", "").str.strip()
    name_to_id = dict(zip(master_2024["name_clean"], master_2024["stock_id"]))
    
    df["name_clean"] = df["stock_name"].str.replace("*", "").str.strip()
    df["stock_id"] = df["name_clean"].map(name_to_id)
    df = df.dropna(subset=["stock_id"])
    print(f"  {len(df)} stocks after name mapping")
    
    # Industry mapping (extract quickly from panel)
    print("  Loading industry mapping...")
    stock_industry = {}
    for chunk in pd.read_csv(PANEL_EF, chunksize=300000, dtype={"stock_id": str},
                              usecols=["stock_id", "industry_cs2012"]):
        for _, row in chunk.dropna(subset=["industry_cs2012"]).iterrows():
            sid = row["stock_id"]
            if sid not in stock_industry:
                stock_industry[sid] = row["industry_cs2012"]
        if len(stock_industry) > 5000:
            break
    df["industry"] = df["stock_id"].map(stock_industry)
    n_industry = df["industry"].notna().sum()
    print(f"  Industry coverage: {n_industry}/{len(df)}")
    
    # Build factor variants
    v = df.copy()
    v["F_hr_thresh2"] = np.where(v["hidden_count"] >= 2, v["HiddenRatio"], np.nan)
    v["F_hr_thresh3"] = np.where(v["hidden_count"] >= 3, v["HiddenRatio"], np.nan)
    v["F_hr_thresh4"] = np.where(v["hidden_count"] >= 4, v["HiddenRatio"], np.nan)
    v["F_hr_thresh5"] = np.where(v["hidden_count"] >= 5, v["HiddenRatio"], np.nan)
    v["F_hidden_pct_all"] = np.where(v["total_funds"] > 0, v["hidden_count"] / v["total_funds"], np.nan)
    v["F_total_funds"] = v["total_funds"]
    v["F_total_log"] = np.log1p(v["total_funds"])
    v["F_stealth"] = v["StealthScore"]
    
    variants = ["F_hr_thresh2", "F_hr_thresh3", "F_hr_thresh4", "F_hr_thresh5",
                "F_hidden_pct_all", "F_total_log", "F_total_funds", "F_stealth"]
    
    # Filter: stocks with IDs in daily returns
    stock_set = set(df["stock_id"])
    
    # Load all years of daily returns
    print("  Loading daily returns (all years)...")
    all_ret = []
    for chunk in pd.read_csv(DAILY_RETURNS, chunksize=300000, dtype={"stock_id": str}):
        chunk = chunk[chunk["stock_id"].isin(stock_set)]
        if len(chunk) == 0:
            continue
        chunk["trade_date"] = pd.to_datetime(chunk["trade_date"])
        all_ret.append(chunk[["stock_id", "trade_date", "ret"]])
    
    df_ret = pd.concat(all_ret, ignore_index=True)
    del all_ret
    df_ret["year"] = df_ret["trade_date"].dt.year
    df_ret["month"] = df_ret["trade_date"].dt.to_period("M")
    
    years = sorted(df_ret["year"].unique())
    print(f"  Years: {years[0]}-{years[-1]} ({len(years)} years)")
    print(f"  Total rows: {len(df_ret):,}")
    
    return df, v, variants, df_ret, stock_industry


# ═══════════════════════ 1. Multi-Year IC ═══════════════════════

def multi_year_ic(v, variants, df_ret):
    """Compute monthly IC for each variant across all available years."""
    print("\n" + "=" * 60)
    print("1. Multi-Year Monthly IC")
    print("=" * 60)
    
    years = sorted(df_ret["year"].unique())
    
    # Build factor lookup
    factor = v.set_index("stock_id")[variants]
    
    results_by_year = {}
    for year in years:
        ret_year = df_ret[df_ret["year"] == year]
        months = sorted(ret_year["month"].unique())
        
        # Build return matrix for this year
        ret_pivot = ret_year.pivot_table(
            index="stock_id", columns="month", values="ret", aggfunc="first"
        )
        
        year_results = {}
        for var in variants:
            fv = factor[var].dropna()
            common = fv.index.intersection(ret_pivot.index)
            if len(common) < 30:
                continue
            
            fv_common = fv.loc[common].values
            monthly_ics = []
            for m in months:
                if m not in ret_pivot.columns:
                    continue
                rets = ret_pivot.loc[common, m].values
                valid = ~np.isnan(rets)
                if valid.sum() < 30 or np.std(rets[valid]) < 1e-10:
                    continue
                ic, _ = stats.spearmanr(fv_common[valid], rets[valid])
                if not np.isnan(ic):
                    monthly_ics.append(ic)
            
            if len(monthly_ics) >= 6:
                ics = np.array(monthly_ics)
                year_results[var] = {
                    "mean_ic": ics.mean(),
                    "ir": ics.mean() / (ics.std() + 1e-8),
                    "tstat": ics.mean() / (ics.std() / np.sqrt(len(ics))),
                    "n_months": len(ics),
                    "pos_ratio": (ics > 0).mean(),
                }
        
        results_by_year[year] = year_results
    
    # Print summary table
    print(f"\n  {'Year':6s}", end="")
    for var in variants:
        print(f" {var.replace('F_',''):>12s}", end="")
    print()
    print(f"  {'-'*6}", end="")
    for _ in variants:
        print(f" {'-'*12}", end="")
    print()
    
    for year in years:
        yr = year
        print(f"  {yr:<6d}", end="")
        for var in variants:
            if var in results_by_year.get(year, {}):
                ic = results_by_year[year][var]["mean_ic"]
                print(f" {ic:+10.4f}", end="")
            else:
                print(f" {'---':>12}", end="")
        print()
    
    # Summary: average IC across years
    print(f"\n  {'Summary IC':<10s}", end="")
    for var in variants:
        all_ics = []
        for year in years:
            if var in results_by_year.get(year, {}):
                all_ics.append(results_by_year[year][var]["mean_ic"])
        if all_ics:
            avg = np.mean(all_ics)
            print(f" {avg:+10.4f}", end="")
        else:
            print(f" {'---':>12}", end="")
    print()
    
    # Summary: IR across years (pooled)
    print(f"\n  {'Summary IR':<10s}", end="")
    for var in variants:
        all_ics = []
        for year in years:
            if var in results_by_year.get(year, {}):
                r = results_by_year[year][var]
                all_ics.append(r["mean_ic"] / (r["mean_ic"] + 1e-8))  # placeholder
        if all_ics:
            avg = np.mean(all_ics)
            print(f" {avg:+10.1f}", end="")
        else:
            print(f" {'---':>12}", end="")
    print()
    
    # Compute proper pooled IR
    print(f"\n  {'Pooled IR':<10s}", end="")
    for var in variants:
        all_ics = []
        for year in years:
            if var in results_by_year.get(year, {}):
                r = results_by_year[year][var]
                all_ics.append(r["ir"])
        if all_ics:
            print(f" {np.mean(all_ics):+10.2f}", end="")
        else:
            print(f" {'---':>12}", end="")
    print()
    
    return results_by_year


# ═══════════════════════ 2. Industry-Neutral IC ═══════════════════════

def industry_neutral_ic(v, variants, df_ret, stock_industry):
    """IC after subtracting industry mean of factor."""
    print("\n" + "=" * 60)
    print("2. Industry-Neutral IC (Monthly)")
    print("=" * 60)
    
    v_ind = v.copy()
    v_ind = v_ind[v_ind["industry"].notna()].copy()
    
    results = {}
    for var in variants:
        # Demean within industry
        v_valid = v_ind.dropna(subset=[var])
        if len(v_valid) < 100:
            continue
        ind_means = v_valid.groupby("industry")[var].transform("mean")
        v_valid[f"{var}_neutral"] = v_valid[var] - ind_means
        
        # Monthly IC over 2023-2025 (recent 3 years)
        ret_focus = df_ret[df_ret["year"].isin([2023, 2024, 2025])]
        months = sorted(ret_focus["month"].unique())
        
        factor_neutral = v_valid.set_index("stock_id")[f"{var}_neutral"]
        
        monthly_ics = []
        for m in months:
            ret_m = ret_focus[ret_focus["month"] == m]
            ret_pivot = ret_m.pivot_table(index="stock_id", values="ret", aggfunc="first")
            
            common = factor_neutral.index.intersection(ret_pivot.index)
            if len(common) < 30:
                continue
            
            fv_vals = factor_neutral.loc[common].values
            ret_vals = ret_pivot.loc[common, "ret"].values
            valid = ~np.isnan(ret_vals)
            if valid.sum() < 30 or np.std(ret_vals[valid]) < 1e-10:
                continue
            ic, _ = stats.spearmanr(fv_vals[valid], ret_vals[valid])
            if not np.isnan(ic):
                monthly_ics.append(ic)
        
        if len(monthly_ics) >= 10:
            ics = np.array(monthly_ics)
            results[var] = {
                "mean_ic": ics.mean(), "ir": ics.mean() / (ics.std() + 1e-8),
                "tstat": ics.mean() / (ics.std() / np.sqrt(len(ics))),
                "n": len(ics), "pos_ratio": (ics > 0).mean()
            }
        
        del v_valid
    
    print(f"\n  {'Variant':20s} {'MeanIC':>8s} {'IR':>7s} {'t-stat':>7s} {'Pos%':>6s} {'N':>4s}")
    print(f"  {'-'*20} {'-'*8} {'-'*7} {'-'*7} {'-'*6} {'-'*4}")
    for var in variants:
        if var in results:
            r = results[var]
            sign = "+" if r["mean_ic"] > 0 else ""
            print(f"  {var:20s} {sign}{r['mean_ic']:7.4f} {r['ir']:7.3f} {r['tstat']:7.2f} {r['pos_ratio']:5.1%} {r['n']:4d}")
    
    return results


# ═══════════════════════ 3. Forward Horizon Analysis ═══════════════════════

def forward_horizon_ic(v, variants, df_ret):
    """Compute IC with forward returns at different horizons (1m, 3m, 6m, 12m).
    
    Essentially: factor as of today → IC with cumulative return over next N months.
    Since we have a static factor (one observation), we test IC stability over calendar months.
    """
    print("\n" + "=" * 60)
    print("3. Forward Horizon IC Decay")
    print("=" * 60)
    
    factor = v.set_index("stock_id")[variants]
    
    # Use 2023-2025 for horizon analysis
    ret_focus = df_ret[df_ret["year"].isin([2023, 2024, 2025])]
    months_all = sorted(ret_focus["month"].unique())
    
    horizons = {"1m": 1, "3m": 3, "6m": 6, "12m": 12}
    
    results = {}
    for var in ["F_hr_thresh3", "F_hidden_pct_all", "F_total_log", "F_stealth"]:
        fv = factor[var].dropna()
        common_base = fv.index
        
        for h_name, h_months in horizons.items():
            monthly_ics = []
            for i in range(len(months_all) - h_months):
                start_m = months_all[i]
                # Get monthly returns for this period
                period_rets = []
                for j in range(h_months):
                    m = months_all[i + j]
                    ret_m = ret_focus[ret_focus["month"] == m]
                    rp = ret_m.pivot_table(index="stock_id", values="ret", aggfunc="first")["ret"]
                    # Rename column
                    period_rets.append(rp.rename(str(m)))
                
                if not period_rets:
                    continue
                ret_df = pd.concat(period_rets, axis=1)
                # Cumulative return over horizon
                cum_ret = (1 + ret_df).prod(axis=1) - 1
                
                common = common_base.intersection(cum_ret.index)
                if len(common) < 30:
                    continue
                
                fv_vals = fv.loc[common].values
                ret_vals = cum_ret.loc[common].values
                valid = ~np.isnan(ret_vals)
                if valid.sum() < 30 or np.std(ret_vals[valid]) < 1e-10:
                    continue
                ic, _ = stats.spearmanr(fv_vals[valid], ret_vals[valid])
                if not np.isnan(ic):
                    monthly_ics.append(ic)
            
            if len(monthly_ics) >= 10:
                ics = np.array(monthly_ics)
                key = (var, h_name)
                results[key] = {
                    "mean_ic": ics.mean(),
                    "ir": ics.mean() / (ics.std() + 1e-8),
                    "tstat": ics.mean() / (ics.std() / np.sqrt(len(ics))),
                    "n": len(ics),
                }
    
    # Print horizon IC table
    vars_show = ["F_hr_thresh3", "F_hidden_pct_all", "F_total_log", "F_stealth"]
    labels = [v.replace("F_", "") for v in vars_show]
    
    print(f"\n  {'Horizon':8s}", end="")
    for l in labels:
        print(f" {l:>14s}", end="")
    print()
    print(f"  {'-'*8}", end="")
    for _ in labels:
        print(f" {'-'*14}", end="")
    print()
    
    for h_name in ["1m", "3m", "6m", "12m"]:
        print(f"  {h_name:8s}", end="")
        for var in vars_show:
            key = (var, h_name)
            if key in results:
                ic = results[key]["mean_ic"]
                print(f" {ic:+13.4f}", end="")
            else:
                print(f" {'---':>14}", end="")
        print()
    
    # Also print IR
    print(f"\n  IR by horizon:")
    print(f"  {'Horizon':8s}", end="")
    for l in labels:
        print(f" {l:>14s}", end="")
    print()
    for h_name in ["1m", "3m", "6m", "12m"]:
        print(f"  {h_name:8s}", end="")
        for var in vars_show:
            key = (var, h_name)
            if key in results:
                ir = results[key]["ir"]
                print(f" {ir:+13.2f}", end="")
            else:
                print(f" {'---':>14}", end="")
        print()
    
    return results


# ═══════════════════════ Charts ═══════════════════════

def plot_results(multi_year, ind_neutral, horizon):
    """Summary charts."""
    
    # Multi-year IC heatmap
    years = sorted(multi_year.keys())
    variants_show = ["F_hr_thresh3", "F_hidden_pct_all", "F_total_log", "F_stealth"]
    labels = [v.replace("F_", "") for v in variants_show]
    
    data = np.zeros((len(variants_show), len(years)))
    data[:] = np.nan
    for vi, var in enumerate(variants_show):
        for yi, year in enumerate(years):
            if var in multi_year.get(year, {}):
                data[vi, yi] = multi_year[year][var]["mean_ic"]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Heatmap
    ax = axes[0][0]
    im = ax.imshow(data, aspect="auto", cmap="RdBu_r", vmin=-0.03, vmax=0.03)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    for vi in range(len(variants_show)):
        for yi in range(len(years)):
            if not np.isnan(data[vi, yi]):
                c = "white" if abs(data[vi, yi]) > 0.015 else "black"
                ax.text(yi, vi, f"{data[vi,yi]:+.3f}", ha="center", va="center",
                        fontsize=7, color=c, fontweight="bold")
    ax.set_title("Multi-Year Monthly IC Heatmap")
    plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Industry-neutral comparison
    ax = axes[0][1]
    if ind_neutral:
        vars_sorted = sorted(ind_neutral.items(), key=lambda x: abs(x[1]["ir"]), reverse=True)
        names = [v[0].replace("F_", "") for v in vars_sorted]
        irs = [v[1]["ir"] for v in vars_sorted]
        colors = [RED if ir > 0 else GREEN for ir in irs]
        bars = ax.barh(range(len(names)), irs, color=colors, alpha=0.8)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=8)
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Industry-Neutral IR")
        ax.set_title("Industry-Neutral Monthly IC IR (2023-2025)")
    
    # Horizon IC decay
    ax = axes[1][0]
    horizons = ["1m", "3m", "6m", "12m"]
    h_nums = [1, 3, 6, 12]
    for var, label, color in [("F_hr_thresh3", "hr_thresh3", GREEN),
                                ("F_hidden_pct_all", "hidden_pct_all", BLUE),
                                ("F_total_log", "total_log", "#E67E22"),
                                ("F_stealth", "stealth", RED)]:
        ics = []
        for h_name in horizons:
            key = (var, h_name)
            if key in horizon:
                ics.append(horizon[key]["mean_ic"])
            else:
                ics.append(np.nan)
        ax.plot(h_nums, ics, 'o-', color=color, linewidth=2, markersize=6, label=label, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Horizon (months)")
    ax.set_ylabel("Mean IC")
    ax.set_title("IC Decay by Forward Horizon")
    ax.legend(fontsize=8)
    
    # Yearly IC bars for best variant
    ax = axes[1][1]
    best_var = "F_hr_thresh3"
    yearly_ics = []
    yearly_labels = []
    for year in years:
        if best_var in multi_year.get(year, {}):
            yearly_ics.append(multi_year[year][best_var]["mean_ic"])
            yearly_labels.append(year)
    if yearly_ics:
        colors_bar = [RED if ic > 0 else GREEN for ic in yearly_ics]
        ax.bar(range(len(yearly_ics)), yearly_ics, color=colors_bar, alpha=0.8)
        ax.set_xticks(range(len(yearly_ics)))
        ax.set_xticklabels(yearly_labels, rotation=45)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axhline(np.mean(yearly_ics), color=BLUE, linestyle="--",
                   label=f"Mean={np.mean(yearly_ics):.4f}")
        ax.set_xlabel("Year")
        ax.set_ylabel("Monthly IC")
        ax.set_title(f"hr_thresh3 — Yearly Monthly IC")
        ax.legend(fontsize=8)
    
    plt.tight_layout()
    path = OUT_DIR / "robustness_summary.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Chart: {path}")


# ═══════════════════════ Main ═══════════════════════

def main():
    # Load
    df, v, variants, df_ret, stock_industry = load_all()
    
    # 1. Multi-year IC
    multi_year = multi_year_ic(v, variants, df_ret)
    
    # 2. Industry-neutral
    ind_neutral = industry_neutral_ic(v, variants, df_ret, stock_industry)
    
    # 3. Forward horizon
    horizon = forward_horizon_ic(v, variants, df_ret)
    
    # Charts
    plot_results(multi_year, ind_neutral, horizon)
    
    print("\n" + "=" * 60)
    print("Done.")
    print("=" * 60)


if __name__ == "__main__":
    main()
