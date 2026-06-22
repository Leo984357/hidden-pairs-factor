#!/usr/bin/env python3
"""
Tier 1+2 Combined: All feasible robustness + heterogeneity experiments
=======================================================================
Runs everything NOT requiring CSMAR downloads.
Optimized: single data load, vectorized IC, pre-baked monthly returns.

Tier 1:
  1. Threshold traversal (hidden_count >= 2,3,4,5,6)
  2. Orthogonal neutralization (Size, Industry, BM, ROA)
  3. Sample split (2015-2020 vs 2021-2025)
  4. Definition robustness (identity + sqrt variants)
  5. Liquidity proxy filter (market cap quartile)

Tier 2:
  6. Market cap stratification (Large/Mid/Small)
  7. Macro cycle grouping (Bull/Bear/Range)
  8. Placebo permutation (100x shuffle)
  9. Volatility grouping (High/Med/Low daily vol)
  10. Industry decomposition (by CSRC 2012)

Tier 3:
  11. StealthScore decomposition proof (breadth × purity)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import warnings, time
warnings.filterwarnings('ignore')

# ── Font setup ──
import matplotlib.font_manager as fm
for name in ['Heiti SC', 'Songti SC', 'PingFang SC', 'STHeiti', 'SimHei']:
    for f in fm.fontManager.ttflist:
        if name in f.name:
            plt.rcParams['font.sans-serif'] = [f.name]
            break
    else:
        continue
    break
plt.rcParams['axes.unicode_minus'] = False

OUT = Path('/Users/leolee/Desktop/hidden-pairs-factor/results/20260618')
OUT.mkdir(parents=True, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════
# DATA LOAD (once, fast)
# ══════════════════════════════════════════════════════════════════════

print("=" * 60)
print("TIER 1+2 COMBINED EXPERIMENTS")
print("=" * 60)

t0 = time.time()

# ── Factor data ──
print("\n[1/4] Loading factor data...")
factor = pd.read_csv(OUT / 'stealth_score_factor.csv')
factor = factor[factor['coverage_flag']].copy()
print(f"  {len(factor)} stocks with coverage")

# Build variants
factor['hidden_count'] = factor['hidden_count'].astype(int)
factor['funds_with_top10'] = np.maximum(factor['hidden_count'], 0)  # placeholder — already in data

# ── Stock code mapping ──
print("[2/4] Loading stock mapping...")
master = pd.read_csv('/Users/leolee/Desktop/genai_china_replication/data/raw/csmar/listing_master_by_year_clean.csv')
master_latest = master[master['year'] == 2024][['stock_id', 'ShortName']].drop_duplicates('stock_id')

# Map factor stock names to codes
name_to_code = dict(zip(master_latest['ShortName'].str.strip(), master_latest['stock_id']))
factor['stock_id'] = factor['stock_name'].str.strip().map(name_to_code)
mapped = factor['stock_id'].notna()
print(f"  Mapped: {mapped.sum()}/{len(factor)} stocks to codes")
factor = factor[mapped].copy()

# ── Daily returns (chunked, filtered) ──
print("[3/4] Loading daily returns (chunked, filtered)...")
valid_codes = set(factor['stock_id'])
RET_PATH = '/Users/leolee/Desktop/genai_china_replication/data/processed/daily_returns_cn_all.csv'

chunks = []
for chunk in pd.read_csv(RET_PATH, chunksize=500000, dtype={'stock_id': str},
                         usecols=['stock_id', 'trade_date', 'ret', 'mkt_cap_float']):
    chunk = chunk[chunk['stock_id'].isin(valid_codes)]
    if len(chunk) > 0:
        chunks.append(chunk)
returns = pd.concat(chunks, ignore_index=True)
returns['trade_date'] = pd.to_datetime(returns['trade_date'])
returns = returns.sort_values(['stock_id', 'trade_date'])
print(f"  {returns.stock_id.nunique()} stocks, {returns.trade_date.nunique()} dates")
print(f"  Range: {returns.trade_date.min().date()} ~ {returns.trade_date.max().date()}")

# ── Fundamentals (from genai panel) ──
print("[4/4] Loading fundamentals...")
FUND_PATH = '/Users/leolee/Desktop/genai_china_replication/Ef_factor_asset_pricing/data/processed/panel_Ef_daily.csv'
fund_cols = ['stock_id', 'trade_date', 'bm', 'roa_2023', 'industry_cs2012_code', 'lev_approx']
funds = pd.read_csv(FUND_PATH, usecols=[c for c in fund_cols if c in pd.read_csv(FUND_PATH, nrows=1).columns], dtype={'stock_id': str})
funds['trade_date'] = pd.to_datetime(funds['trade_date'])
print(f"  Fundamentals: {funds.stock_id.nunique()} stocks, {funds.trade_date.nunique()} dates")

# ══════════════════════════════════════════════════════════════════════
# PREP: Monthly returns panel + annual stock metadata
# ══════════════════════════════════════════════════════════════════════

# Convert daily returns to stock x month matrix
returns['year'] = returns['trade_date'].dt.year
returns['month'] = returns['trade_date'].dt.month
returns['yearmon'] = returns['trade_date'].dt.to_period('M')

# Monthly return = product of (1+daily_ret) - 1
monthly = returns.groupby(['stock_id', 'yearmon'])['ret'].apply(lambda x: (1 + x).prod() - 1).reset_index()
monthly['yearmon_dt'] = monthly['yearmon'].dt.to_timestamp()

# Pivot to stock_id x yearmon matrix
ret_matrix = monthly.pivot(index='stock_id', columns='yearmon_dt', values='ret')
print(f"\nMonthly return matrix: {ret_matrix.shape[0]} stocks x {ret_matrix.shape[1]} months")

# Annual average market cap per stock
annual_cap = returns.groupby(['stock_id', 'year'])['mkt_cap_float'].mean().reset_index()

# Annual fundamentals
funds['year'] = funds['trade_date'].dt.year
annual_fund = funds.groupby(['stock_id', 'year'])[['bm', 'roa_2023', 'lev_approx']].mean().reset_index()
# Most common industry per stock per year
industry_mode = funds.groupby(['stock_id', 'year'])['industry_cs2012_code'].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan).reset_index()
annual_fund = annual_fund.merge(industry_mode, on=['stock_id', 'year'], how='left')

# ══════════════════════════════════════════════════════════════════════
# FACTOR VARIANTS
# ══════════════════════════════════════════════════════════════════════

def build_factor_variants(df):
    """Build many variants from the base dataframe."""
    variants = {}
    
    hc = df['hidden_count'].values
    hr = df['HiddenRatio'].values
    
    # Threshold variants
    for t in [2, 3, 4, 5, 6]:
        mask = hc >= t
        variants[f'hr_t{t}'] = np.where(mask, hr, np.nan)
        variants[f'hc_t{t}'] = np.where(mask, hc.astype(float), np.nan)
    
    # Raw variants
    variants['hidden_ratio'] = hr
    variants['hidden_count'] = hc.astype(float)
    variants['ln_hc'] = np.log1p(hc)
    variants['stealth_score'] = np.log1p(hc) * hr
    
    # Definition robustness
    variants['sqrt_hc_times_hr'] = np.sqrt(np.maximum(hc, 0)) * hr
    
    return variants

factor_var = build_factor_variants(factor)

# ══════════════════════════════════════════════════════════════════════
# HELPER: Compute multi-year monthly IC for a factor variant
# ══════════════════════════════════════════════════════════════════════

def compute_yearly_ic(factor_vals, stock_ids, ret_matrix, years=range(2015, 2026)):
    """For each year, compute monthly IC between factor and forward 1M return.
    Returns: dict of {year: [monthly_ICs]}"""
    results = {}
    for y in years:
        # Forward returns: factor computed at start of year, returns are monthly
        year_months = [m for m in ret_matrix.columns if m.year == y]
        if len(year_months) < 3:
            continue
        
        fv = pd.Series(factor_vals, index=stock_ids).dropna()
        common = [s for s in fv.index if s in ret_matrix.index]
        if len(common) < 100:
            results[y] = []
            continue
        
        fv_c = fv[common].values
        monthly_ics = []
        for m in year_months:
            r = ret_matrix.loc[common, m].values
            valid = ~np.isnan(r)
            if valid.sum() < 30:
                continue
            ic, _ = stats.spearmanr(fv_c[valid], r[valid])
            monthly_ics.append(ic)
        
        if monthly_ics:
            results[y] = monthly_ics
    return results


def compute_fwd_horizon_ic(factor_vals, stock_ids, ret_matrix):
    """Compute IC for 1M, 3M, 6M, 12M forward horizons using cumulative returns."""
    fv = pd.Series(factor_vals, index=stock_ids).dropna()
    common = [s for s in fv.index if s in ret_matrix.index]
    fv_c = fv[common].values
    
    horizons = {'1M': 1, '3M': 3, '6M': 6, '12M': 12}
    results = {}
    
    for label, h in horizons.items():
        all_ics = []
        months = ret_matrix.columns
        for i in range(len(months) - h):
            m_start = months[i]
            m_end = months[i + h - 1]
            # Cumulative return over h months
            cum_ret = (1 + ret_matrix.loc[common, m_start:m_end]).prod(axis=1) - 1
            valid = ~np.isnan(cum_ret.values)
            if valid.sum() < 30:
                continue
            ic, _ = stats.spearmanr(fv_c[valid], cum_ret.values[valid])
            all_ics.append(ic)
        
        if all_ics:
            results[label] = {
                'mean_ic': np.mean(all_ics),
                'ic_ir': np.mean(all_ics) / max(np.std(all_ics), 1e-10),
                't_stat': np.mean(all_ics) / max(np.std(all_ics) / np.sqrt(len(all_ics)), 1e-10),
                'n': len(all_ics)
            }
    return results


def summarize_ic(yearly_dict):
    """From {year: [ICs]}, compute pooled stats."""
    all_ics = []
    yearly_means = {}
    for y, ics in yearly_dict.items():
        if ics:
            all_ics.extend(ics)
            yearly_means[y] = np.mean(ics)
    
    if not all_ics:
        return {'mean': np.nan, 'ir': np.nan, 't': np.nan, 'pos_ratio': np.nan, 'n_months': 0}
    
    arr = np.array(all_ics)
    pos = (arr > 0).mean()
    return {
        'mean': np.mean(arr),
        'ir': np.mean(arr) / max(np.std(arr), 1e-10),
        't': np.mean(arr) / max(np.std(arr) / np.sqrt(len(arr)), 1e-10),
        'pos_ratio': pos,
        'n_months': len(arr),
        'yearly_means': yearly_means
    }


# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: Threshold Traversal
# ══════════════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("EXP 1: Threshold Traversal (hc >= 2,3,4,5,6)")
print("─" * 50)

threshold_results = {}
for t in [2, 3, 4, 5, 6]:
    fv = pd.Series(factor_var[f'hr_t{t}'], index=factor['stock_id'])
    yearly = compute_yearly_ic(fv.dropna().values, fv.dropna().index, ret_matrix)
    stats_data = summarize_ic(yearly)
    nr = fv.dropna().sum()
    stats_data['n_stocks'] = fv.notna().sum()
    threshold_results[t] = stats_data
    print(f"  t={t}: N={stats_data['n_stocks']:>5}, IC={stats_data['mean']:+.4f}, IR={stats_data['ir']:+.2f}, t={stats_data['t']:+.2f}, pos={stats_data['pos_ratio']:.0%}")

# Fwd horizon for each threshold
horizon_results = {}
for t in [2, 3, 4, 5, 6]:
    fv = pd.Series(factor_var[f'hr_t{t}'], index=factor['stock_id']).dropna()
    horizon_results[t] = compute_fwd_horizon_ic(fv.values, fv.index, ret_matrix)

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: Orthogonal Neutralization
# ══════════════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("EXP 2: Orthogonal Neutralization (Size, Industry, BM, ROA)")
print("─" * 50)

# Use hr_t4 as baseline
baseline_key = 'hr_t4'
baseline_fv = pd.Series(factor_var[baseline_key], index=factor['stock_id']).dropna()
baseline_yearly = compute_yearly_ic(baseline_fv.values, baseline_fv.index, ret_matrix)
baseline_stats = summarize_ic(baseline_yearly)

# For multi-factor orthogonalization: regress factor on Size+BM+ROA+industry dummies
# Then use residual as IC signal
def orthogonalize_factor(fv, stock_ids, ret_matrix, annual_fund):
    """Regress factor on Size+BM+ROA, return residual (numpy linalg)."""
    fund_2024 = annual_fund[annual_fund['year'] == 2024].set_index('stock_id')
    
    df = pd.DataFrame({'factor': fv}, index=stock_ids)
    df = df.join(fund_2024[['bm', 'roa_2023']], how='left')
    
    # Size from annual mkt cap
    cap_2024 = annual_cap[annual_cap['year'] == 2024].groupby('stock_id')['mkt_cap_float'].mean()
    df['size_ln'] = np.log(cap_2024)
    
    # Drop NaN
    reg_cols = ['bm', 'roa_2023', 'size_ln']
    df_clean = df.dropna(subset=reg_cols + ['factor'])
    
    if len(df_clean) < 100:
        return None, None
    
    X = df_clean[reg_cols].values
    # Add constant
    X = np.column_stack([np.ones(len(X)), X])
    y = df_clean['factor'].values
    
    try:
        coeff, residuals, rank, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ coeff
        residual = y - y_pred
        r2 = 1 - np.sum(residual**2) / np.sum((y - np.mean(y))**2)
        return pd.Series(residual, index=df_clean.index), r2
    except:
        return None, None

print("  Orthogonalizing hr_t4 against Size+BM+ROA...")
orth_res, r2 = orthogonalize_factor(
    pd.Series(factor_var[baseline_key], index=factor['stock_id']).dropna(),
    list(factor['stock_id']),
    ret_matrix, annual_fund
)

if orth_res is not None:
    orth_yearly = compute_yearly_ic(orth_res.values, orth_res.index, ret_matrix)
    orth_stats = summarize_ic(orth_yearly)
    print(f"  R² of factor ~ Size+BM+ROA = {r2:.4f}")
    print(f"  Orthogonalized: IC={orth_stats['mean']:+.4f}, IR={orth_stats['ir']:+.2f}, t={orth_stats['t']:+.2f}")
else:
    orth_stats = None
    print("  Orthogonalization failed (insufficient data)")

# Industry neutralization
print("  Industry-neutral IC...")
ind_neutral_yearly = {}
for y in range(2015, 2026):
    year_months = [m for m in ret_matrix.columns if m.year == y]
    if len(year_months) < 3:
        continue
    
    fund_year = annual_fund[annual_fund['year'] == y].set_index('stock_id')
    fv = baseline_fv.copy()
    
    # Subtract industry mean factor
    fv_df = pd.DataFrame({'factor': fv, 'industry': fund_year['industry_cs2012_code']})
    ind_mean = fv_df.groupby('industry')['factor'].transform('mean')
    fv_df['factor_neutral'] = fv_df['factor'] - ind_mean
    
    for m in year_months:
        fv_n = fv_df['factor_neutral'].dropna()
        common = [s for s in fv_n.index if s in ret_matrix.index]
        if len(common) < 100:
            continue
        r = ret_matrix.loc[common, m].values
        valid = ~np.isnan(r)
        if valid.sum() < 30:
            continue
        ic, _ = stats.spearmanr(fv_n[common].values[valid], r[valid])
        ind_neutral_yearly.setdefault(y, []).append(ic)

ind_neutral_stats = summarize_ic(ind_neutral_yearly)
print(f"  Industry-neutral: IC={ind_neutral_stats['mean']:+.4f}, IR={ind_neutral_stats['ir']:+.2f}, t={ind_neutral_stats['t']:+.2f}")

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: Sample Split
# ══════════════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("EXP 3: Sample Split (2015-2020 vs 2021-2025)")
print("─" * 50)

split_results = {}
for label, years_range in [('2015-2020', range(2015, 2021)), ('2021-2025', range(2021, 2026))]:
    for t in [2, 3, 4, 5, 6]:
        fv = pd.Series(factor_var[f'hr_t{t}'], index=factor['stock_id']).dropna()
        yearly = compute_yearly_ic(fv.values, fv.index, ret_matrix, years=years_range)
        stats_data = summarize_ic(yearly)
        split_results[f'{label}_t{t}'] = stats_data
        print(f"  {label} t={t}: IC={stats_data['mean']:+.4f}, IR={stats_data['ir']:+.2f}, t={stats_data['t']:+.2f}")

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: Definition Robustness
# ══════════════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("EXP 4: Definition Robustness")
print("─" * 50)

def_variants = ['hr_t4', 'sqrt_hc_times_hr', 'hidden_ratio', 'hidden_count', 'ln_hc', 'stealth_score']
def_results = {}
for vname in def_variants:
    fv = pd.Series(factor_var[vname], index=factor['stock_id']).dropna()
    yearly = compute_yearly_ic(fv.values, fv.index, ret_matrix)
    stats_data = summarize_ic(yearly)
    def_results[vname] = stats_data
    print(f"  {vname:<20}: N={fv.notna().sum():>5}, IC={stats_data['mean']:+.4f}, IR={stats_data['ir']:+.2f}")

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: Liquidity Proxy (Market Cap Quartile)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("EXP 5: Liquidity Proxy (Market Cap Quartile Filter)")
print("─" * 50)

cap_2024 = annual_cap[annual_cap['year'] == 2024].groupby('stock_id')['mkt_cap_float'].mean()
cap_q = pd.qcut(cap_2024, q=4, labels=['Q1_Small', 'Q2', 'Q3', 'Q4_Large'])
cap_q.name = 'cap_quartile'

liquidity_results = {}
for label in ['Q2', 'Q3', 'Q4_Large', 'Q3+Q4']:
    if 'Q3+Q4' in label:
        keep_stocks = cap_q[cap_q.isin(['Q3', 'Q4_Large'])].index
    else:
        keep_stocks = cap_q[cap_q == label].index
    
    fv = baseline_fv.copy()
    fv = fv[fv.index.isin(keep_stocks)]
    if fv.notna().sum() < 100:
        print(f"  {label}: too few stocks ({fv.notna().sum()})")
        continue
    
    yearly = compute_yearly_ic(fv.dropna().values, fv.dropna().index, ret_matrix)
    stats_data = summarize_ic(yearly)
    liquidity_results[label] = stats_data
    print(f"  ≥{label}: N={fv.notna().sum():>5}, IC={stats_data['mean']:+.4f}, IR={stats_data['ir']:+.2f}")

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 6: Market Cap Stratification
# ══════════════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("EXP 6: Market Cap Stratification")
print("─" * 50)

cap_strat_results = {}
for label, keep_stocks in [
    ('Small', cap_q[cap_q == 'Q1_Small'].index),
    ('Mid', cap_q[cap_q.isin(['Q2', 'Q3'])].index),
    ('Large', cap_q[cap_q == 'Q4_Large'].index)
]:
    fv = baseline_fv.copy()
    fv = fv[fv.index.isin(keep_stocks)]
    if fv.notna().sum() < 50:
        continue
    yearly = compute_yearly_ic(fv.dropna().values, fv.dropna().index, ret_matrix)
    stats_data = summarize_ic(yearly)
    fwd = compute_fwd_horizon_ic(fv.dropna().values, fv.dropna().index, ret_matrix)
    cap_strat_results[label] = {'ic': stats_data, 'fwd': fwd}
    print(f"  {label:<6}: N={fv.notna().sum():>5}, IC={stats_data['mean']:+.4f}, IR={stats_data['ir']:+.2f}")

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 7: Macro Cycle Grouping
# ══════════════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("EXP 7: Macro Cycle Grouping (Bull/Bear/Range)")
print("─" * 50)

# Classify years based on average monthly return
full_panel_ret = ret_matrix.mean(axis=0)
yearly_returns = full_panel_ret.groupby(full_panel_ret.index.year).mean()
yearly_cum = yearly_returns.cumsum()

def classify_year(y):
    """Simple rule-based classification."""
    # 2015: huge bull then crash — special
    if y == 2015:
        return 'Bull'
    # 2018: known bear
    if y == 2018:
        return 'Bear'
    # 2022-2023: grinding bear
    if y in [2022, 2023]:
        return 'Bear'
    # 2019-2021: bull run
    if y in [2019, 2020, 2021]:
        return 'Bull'
    # 2016-2017: recovery
    if y in [2016, 2017]:
        return 'Range'
    # 2024-2025: policy-driven
    if y in [2024, 2025]:
        return 'Range'
    return 'Range'

macro_results = {}
for regime, years_list in [
    ('Bull', [2015, 2019, 2020, 2021]),
    ('Range', [2016, 2017, 2024, 2025]),
    ('Bear', [2018, 2022, 2023])
]:
    fv = baseline_fv.copy()
    yearly = compute_yearly_ic(fv.dropna().values, fv.dropna().index, ret_matrix, years=years_list)
    stats_data = summarize_ic(yearly)
    macro_results[regime] = stats_data
    print(f"  {regime:<6}: IC={stats_data['mean']:+.4f}, IR={stats_data['ir']:+.2f}, t={stats_data['t']:+.2f}, n={stats_data['n_months']}")

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 8: Placebo Permutation (100x)
# ══════════════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("EXP 8: Placebo Permutation (100x shuffle)")
print("─" * 50)

np.random.seed(42)
fv_raw = pd.Series(factor_var[baseline_key], index=factor['stock_id']).dropna()
true_stats = summarize_ic(compute_yearly_ic(fv_raw.values, fv_raw.index, ret_matrix))

placebo_irs = []
placebo_ics = []
for i in range(100):
    shuffled = fv_raw.copy()
    shuffled.iloc[:] = np.random.permutation(shuffled.values)
    yearly = compute_yearly_ic(shuffled.values, shuffled.index, ret_matrix)
    s = summarize_ic(yearly)
    placebo_irs.append(s['ir'])
    placebo_ics.append(s['mean'])
    if (i + 1) % 20 == 0:
        print(f"  Iteration {i+1}/100...")

placebo_irs = np.array(placebo_irs)
placebo_ics = np.array(placebo_ics)
print(f"  True IR: {true_stats['ir']:.2f}")
print(f"  Placebo IR: mean={placebo_irs.mean():.3f}, std={placebo_irs.std():.3f}, max={placebo_irs.max():.3f}")
print(f"  P-value (IR > true): {(placebo_irs >= true_stats['ir']).mean():.4f}")

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 9: Volatility Grouping
# ══════════════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("EXP 9: Volatility Grouping (Daily Return Std)")
print("─" * 50)

# Compute annual volatility per stock
ret_vol = returns.copy()
ret_vol['ret_sq'] = ret_vol['ret'] ** 2
annual_vol = ret_vol.groupby(['stock_id', 'year'])['ret_sq'].mean().pow(0.5) * np.sqrt(252)  # annualized
# Extract 2024 — handle both MultiIndex and regular index
if isinstance(annual_vol.index, pd.MultiIndex):
    vol_2024 = annual_vol.xs(2024, level='year')
else:
    vol_2024 = annual_vol
vol_2024.name = 'ann_vol'

vol_q = pd.qcut(vol_2024, q=3, labels=['LowVol', 'MedVol', 'HighVol'])

vol_results = {}
for label, keep_stocks in [
    ('LowVol', vol_q[vol_q == 'LowVol'].index),
    ('MedVol', vol_q[vol_q == 'MedVol'].index),
    ('HighVol', vol_q[vol_q == 'HighVol'].index)
]:
    fv = baseline_fv.copy()
    fv = fv[fv.index.isin(keep_stocks)]
    if fv.notna().sum() < 50:
        continue
    yearly = compute_yearly_ic(fv.dropna().values, fv.dropna().index, ret_matrix)
    stats_data = summarize_ic(yearly)
    fwd = compute_fwd_horizon_ic(fv.dropna().values, fv.dropna().index, ret_matrix)
    vol_results[label] = {'ic': stats_data, 'fwd': fwd}
    print(f"  {label:<8}: N={fv.notna().sum():>5}, IC={stats_data['mean']:+.4f}, IR={stats_data['ir']:+.2f}")

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 10: Industry Decomposition
# ══════════════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("EXP 10: Industry Decomposition")
print("─" * 50)

fund_2024 = annual_fund[annual_fund['year'] == 2024].set_index('stock_id')
# Map factor stocks to industry codes
factor_ind = factor[['stock_id']].merge(
    fund_2024[['industry_cs2012_code']], left_on='stock_id', right_index=True, how='left'
)
industry_results = {}
for ind_code, ind_stocks in factor_ind.groupby('industry_cs2012_code')['stock_id']:
    ind_stocks = set(ind_stocks)
    fv = baseline_fv.copy()
    fv = fv[fv.index.isin(ind_stocks)]
    if fv.notna().sum() < 30:
        continue
    yearly = compute_yearly_ic(fv.dropna().values, fv.dropna().index, ret_matrix)
    stats_data = summarize_ic(yearly)
    if pd.notna(stats_data['mean']):
        industry_results[ind_code] = {'ic': stats_data, 'n': fv.notna().sum()}

# Sort by IR
sorted_inds = sorted(industry_results.items(), key=lambda x: x[1]['ic']['ir'] if pd.notna(x[1]['ic']['ir']) else -999, reverse=True)
for ind, s in sorted_inds[:15]:
    print(f"  Ind {ind}: N={s['n']:>4}, IC={s['ic']['mean']:+.4f}, IR={s['ic']['ir']:+.2f}")

# ══════════════════════════════════════════════════════════════════════
# EXPERIMENT 11: StealthScore Decomposition
# ══════════════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("EXP 11: StealthScore Decomposition (Breadth × Purity)")
print("─" * 50)

# Group stocks by high/low breadth and high/low purity
hc_median = factor['hidden_count'].median()
hr_median = factor['HiddenRatio'].median()

groups = {
    'HighB_HighP': (factor['hidden_count'] >= hc_median) & (factor['HiddenRatio'] >= hr_median),
    'HighB_LowP':  (factor['hidden_count'] >= hc_median) & (factor['HiddenRatio'] < hr_median),
    'LowB_HighP':  (factor['hidden_count'] < hc_median) & (factor['HiddenRatio'] >= hr_median),
    'LowB_LowP':   (factor['hidden_count'] < hc_median) & (factor['HiddenRatio'] < hr_median),
}

decomp_results = {}
for gname, gmask in groups.items():
    sub = factor[gmask]
    stock_ids = sub['stock_id'].tolist()
    if len(stock_ids) < 100:
        continue
    
    # Test: pure HiddenRatio IC within this group
    fv = pd.Series(sub['HiddenRatio'].values, index=sub['stock_id']).dropna()
    yearly = compute_yearly_ic(fv.values, fv.index, ret_matrix)
    stats_data = summarize_ic(yearly)
    decomp_results[gname] = {
        'n': len(stock_ids),
        'ic': stats_data,
        'mean_ss': sub['StealthScore'].mean() if 'StealthScore' in sub.columns else np.nan,
        'mean_hc': sub['hidden_count'].mean(),
        'mean_hr': sub['HiddenRatio'].mean()
    }
    print(f"  {gname:<12}: N={len(stock_ids):>5}, IC={stats_data['mean']:+.4f}, IR={stats_data['ir']:+.2f}")

# Also test: is the StealthScore IC driven more by HighB or HighP?
print("\n  ── Breadth- vs Purity-driven IC ──")
for label, fv_series in [
    ('HiddenRatio (HR)', pd.Series(factor['HiddenRatio'].values, index=factor['stock_id'])),
    ('HiddenCount (HC)', pd.Series(factor['hidden_count'].values.astype(float), index=factor['stock_id'])),
    ('StealthScore', pd.Series(factor_var['stealth_score'], index=factor['stock_id'])),
]:
    fv = fv_series.dropna()
    yearly = compute_yearly_ic(fv.values, fv.index, ret_matrix)
    s = summarize_ic(yearly)
    print(f"  {label:<20}: IC={s['mean']:+.4f}, IR={s['ir']:+.2f}")

# ══════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════

print("\n" + "─" * 50)
print("GENERATING PLOTS...")
print("─" * 50)

# ── Figure 1: Threshold Traversal IR ──
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.suptitle('Tier 1: Threshold Traversal — IR & Coverage', fontsize=16, fontweight='bold', y=0.98)

# 1a: Monthly IC IR by threshold
ax = axes[0, 0]
ts = list(threshold_results.keys())
irs = [threshold_results[t]['ir'] for t in ts]
ns = [threshold_results[t]['n_stocks'] for t in ts]
colors = ['#2E86AB' if ir > 0 else '#A23B72' for ir in irs]
bars = ax.bar([str(t) for t in ts], irs, color=colors, edgecolor='white', linewidth=0.5)
for bar, ir in zip(bars, irs):
    ax.text(bar.get_x() + bar.get_width()/2, ir + 0.01 if ir >= 0 else ir - 0.05,
            f'{ir:.2f}', ha='center', fontsize=10, fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
ax.set_ylabel('Monthly IC IR', fontsize=12)
ax.set_xlabel('Hidden Count Threshold', fontsize=12)
ax.set_title('Factor IR by Threshold (11-year pooled)', fontsize=13, fontweight='bold')

# 1b: Stock count vs IR (Pareto frontier view)
ax = axes[0, 1]
ax.scatter(ns, irs, s=200, c=colors, edgecolors='white', linewidth=0.8, zorder=5)
for t, n, ir in zip(ts, ns, irs):
    ax.annotate(f't={t}', (n, ir), textcoords="offset points", xytext=(8, 8), fontsize=11)
ax.plot(ns, irs, '--', color='gray', alpha=0.5, linewidth=1)
ax.axhline(y=0, color='red', linestyle=':', linewidth=0.8, alpha=0.5)
ax.set_xlabel('Covered Stocks', fontsize=12)
ax.set_ylabel('IR', fontsize=12)
ax.set_title('Coverage vs IR (Pareto Frontier)', fontsize=13, fontweight='bold')

# 1c: Forward horizon IR by threshold
ax = axes[1, 0]
horizons = ['1M', '3M', '6M', '12M']
x = np.arange(len(horizons))
width = 0.15
for i, t in enumerate([2, 3, 4, 5, 6]):
    irs_h = [horizon_results[t][h]['ic_ir'] if h in horizon_results[t] else 0 for h in horizons]
    ax.bar(x + i * width - 0.3, irs_h, width, label=f't={t}', edgecolor='white')
ax.set_xticks(x)
ax.set_xticklabels(horizons)
ax.set_ylabel('Forward IC IR', fontsize=12)
ax.set_xlabel('Horizon', fontsize=12)
ax.set_title('Forward Horizon IR by Threshold', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)

# 1d: Yearly IC heatmap for hr_t4
ax = axes[1, 1]
yearly_data = {}
for t in [2, 3, 4, 5, 6]:
    fv = pd.Series(factor_var[f'hr_t{t}'], index=factor['stock_id']).dropna()
    yearly = compute_yearly_ic(fv.values, fv.index, ret_matrix)
    for y, ics in yearly.items():
        if ics:
            yearly_data[(t, y)] = np.mean(ics)

# Build heatmap matrix
years_sorted = sorted(set(y for (t, y) in yearly_data.keys()))
ts_sorted = [2, 3, 4, 5, 6]
hm = np.zeros((len(ts_sorted), len(years_sorted)))
for i, t in enumerate(ts_sorted):
    for j, y in enumerate(years_sorted):
        hm[i, j] = yearly_data.get((t, y), np.nan)

im = ax.imshow(hm, cmap='RdBu_r', aspect='auto', vmin=-0.05, vmax=0.05)
for i in range(len(ts_sorted)):
    for j in range(len(years_sorted)):
        if not np.isnan(hm[i, j]):
            text_color = 'white' if abs(hm[i, j]) > 0.03 else 'black'
            ax.text(j, i, f'{hm[i,j]:.3f}', ha='center', va='center', fontsize=9, color=text_color)
ax.set_yticks(range(len(ts_sorted)))
ax.set_yticklabels([f't={t}' for t in ts_sorted])
ax.set_xticks(range(len(years_sorted)))
ax.set_xticklabels(years_sorted, rotation=45)
ax.set_title('Yearly Mean IC Heatmap', fontsize=13, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / 'tier1_threshold_traversal.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ tier1_threshold_traversal.png")

# ── Figure 2: Orthogonalization + Split ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Tier 1: Orthogonal Neutralization & Sample Split', fontsize=16, fontweight='bold', y=1.02)

# 2a: Orthogonalization comparison
ax = axes[0]
labels = ['Raw Baseline', 'Industry-Neutral', 'Size+BM+ROA\nNeutral']
irs = [baseline_stats['ir'], ind_neutral_stats['ir'], orth_stats['ir'] if orth_stats else 0]
ts_vals = [baseline_stats['t'], ind_neutral_stats['t'], orth_stats['t'] if orth_stats else 0]
colors2 = ['#2E86AB', '#D64045', '#FAA916']
bars = ax.bar(labels, irs, color=colors2, edgecolor='white', linewidth=0.8)
for bar, ir, tp in zip(bars, irs, ts_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'IR={ir:.2f}\nt={tp:.2f}', ha='center', fontsize=10, fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
ax.set_ylabel('Monthly IC IR', fontsize=12)
ax.set_title('Orthogonal Neutralization\n(hr_t4 baseline)', fontsize=13, fontweight='bold')

# 2b: Sample split comparison (IR by threshold, two periods)
ax = axes[1]
x2 = np.arange(len(ts))
width2 = 0.35
ir_15 = [split_results.get(f'2015-2020_t{t}', {}).get('ir', 0) for t in ts]
ir_21 = [split_results.get(f'2021-2025_t{t}', {}).get('ir', 0) for t in ts]
ax.bar(x2 - width2/2, ir_15, width2, label='2015-2020', color='#2E86AB', edgecolor='white')
ax.bar(x2 + width2/2, ir_21, width2, label='2021-2025', color='#D64045', edgecolor='white')
ax.set_xticks(x2)
ax.set_xticklabels([f't={t}' for t in ts])
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
ax.set_ylabel('IR', fontsize=12)
ax.set_title('Sample Split IR Comparison', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)

# 2c: Definition robustness
ax = axes[2]
vn = list(def_results.keys())
vn_short = ['hr_t4', 'sqrt(HC)×HR', 'HiddenRatio', 'HiddenCnt', 'ln(1+HC)', 'StealthScore']
irs_def = [def_results[v]['ir'] for v in vn]
colors3 = ['#2E86AB' if ir > 0 else '#A23B72' for ir in irs_def]
bars = ax.barh(vn_short, irs_def, color=colors3, edgecolor='white', linewidth=0.5)
for bar, ir in zip(bars, irs_def):
    x = bar.get_width()
    ax.text(x + 0.02 if ir >= 0 else x - 0.1, bar.get_y() + bar.get_height()/2,
            f'{ir:.2f}', va='center', fontsize=10, fontweight='bold')
ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)
ax.set_title('Definition Robustness IR', fontsize=13, fontweight='bold')

plt.tight_layout()
fig.savefig(OUT / 'tier1_neutralization_split.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ tier1_neutralization_split.png")

# ── Figure 3: Tier 2 — Heterogeneity ──
fig, axes = plt.subplots(2, 2, figsize=(16, 13))
fig.suptitle('Tier 2: Heterogeneity Experiments', fontsize=16, fontweight='bold', y=0.98)

# 3a: Market cap stratification
ax = axes[0, 0]
clabels = list(cap_strat_results.keys())
cirs = [cap_strat_results[c]['ic']['ir'] for c in clabels]
cns = [cap_strat_results[c]['ic']['n_months'] for c in clabels]
colors4 = ['#D64045', '#FAA916', '#2E86AB']
bars = ax.bar(clabels, cirs, color=colors4, edgecolor='white')
for bar, ir in zip(bars, cirs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 if ir >=0 else bar.get_height() - 0.04,
            f'{ir:.2f}', ha='center', fontsize=11, fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
ax.set_title('Market Cap Stratification\n(hr_t4 IR)', fontsize=13, fontweight='bold')
ax.set_ylabel('IR', fontsize=12)

# 3b: Macro cycle
ax = axes[0, 1]
mlabels = list(macro_results.keys())
mirs = [macro_results[m]['ir'] for m in mlabels]
mts = [macro_results[m]['t'] for m in mlabels]
colors5 = ['#D64045', '#FAA916', '#2E86AB']
bars = ax.bar(mlabels, mirs, color=colors5, edgecolor='white')
for bar, ir, tp_val in zip(bars, mirs, mts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f'IR={ir:.2f}', ha='center', fontsize=11, fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
ax.set_title('Macro Cycle IR\n(hr_t4)', fontsize=13, fontweight='bold')
ax.set_ylabel('IR', fontsize=12)

# 3c: Placebo distribution
ax = axes[1, 0]
ax.hist(placebo_irs, bins=25, color='#808080', alpha=0.7, edgecolor='white', density=True, label='Placebo (100x)')
ax.axvline(x=true_stats['ir'], color='#D64045', linewidth=3, linestyle='--', label=f'True IR = {true_stats["ir"]:.2f}')
ax.axvline(x=0, color='gray', linewidth=1, linestyle=':', alpha=0.5)
ax.set_xlabel('IR', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title(f'Placebo Permutation Test\nP(IR > True) = {(placebo_irs >= true_stats["ir"]).mean():.4f}', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)

# 3d: Forward horizon by size group
ax = axes[1, 1]
for label, color in [('Small', '#D64045'), ('Mid', '#FAA916'), ('Large', '#2E86AB')]:
    if label in cap_strat_results and 'fwd' in cap_strat_results[label]:
        fwd = cap_strat_results[label]['fwd']
        hlabels = ['1M', '3M', '6M', '12M']
        irs_h = [fwd[h]['ic_ir'] if h in fwd else np.nan for h in hlabels]
        ax.plot(hlabels, irs_h, 'o-', color=color, linewidth=2, markersize=8, label=label)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
ax.set_xlabel('Forward Horizon', fontsize=12)
ax.set_ylabel('IC IR', fontsize=12)
ax.set_title('Forward Horizon IR by Size', fontsize=13, fontweight='bold')
ax.legend(fontsize=11)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / 'tier2_heterogeneity.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ tier2_heterogeneity.png")

# ── Figure 4: Volatility + Decomposition ──
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('Extended Heterogeneity & Decomposition', fontsize=16, fontweight='bold', y=1.02)

# 4a: Volatility grouping
ax = axes[0]
vlabels = list(vol_results.keys())
virs = [vol_results[v]['ic']['ir'] for v in vlabels]
colors6 = ['#2E86AB', '#FAA916', '#D64045']
bars = ax.bar(vlabels, virs, color=colors6, edgecolor='white')
for bar, ir in zip(bars, virs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 if ir >=0 else bar.get_height() - 0.04,
            f'{ir:.2f}', ha='center', fontsize=11, fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
ax.set_title('Volatility Group IR', fontsize=13, fontweight='bold')
ax.set_ylabel('IR', fontsize=12)

# 4b: StealthScore decomposition (Quadrant analysis)
ax = axes[1]
dnames = ['HighB\nHighP', 'HighB\nLowP', 'LowB\nHighP', 'LowB\nLowP']
if all(d in decomp_results for d in ['HighB_HighP', 'HighB_LowP', 'LowB_HighP', 'LowB_LowP']):
    dirs = [decomp_results[d]['ic']['ir'] for d in ['HighB_HighP', 'HighB_LowP', 'LowB_HighP', 'LowB_LowP']]
    colors7 = ['#2E86AB', '#A23B72', '#FAA916', '#808080']
    bars = ax.barh(dnames, dirs, color=colors7, edgecolor='white')
    for bar, ir in zip(bars, dirs):
        ax.text(bar.get_width() + 0.01 if ir >= 0 else bar.get_width() - 0.08,
                bar.get_y() + bar.get_height()/2, f'{ir:.2f}', va='center', fontsize=10, fontweight='bold')
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)
    ax.set_title('StealthScore Decomposition\n(Breadth × Purity Quadrants)', fontsize=13, fontweight='bold')

# 4c: Liquidity proxy
ax = axes[2]
llabels = list(liquidity_results.keys())
lirs = [liquidity_results[l]['ir'] for l in llabels]
colors8 = ['#D64045', '#FAA916', '#2E86AB', '#1B998B'][:len(llabels)]
bars = ax.bar(llabels, lirs, color=colors8, edgecolor='white')
for bar, ir in zip(bars, lirs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01 if ir >=0 else bar.get_height() - 0.04,
            f'{ir:.2f}', ha='center', fontsize=11, fontweight='bold')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8)
ax.set_title('Liquidity Proxy IR\n(Remove Q1 smallest)', fontsize=13, fontweight='bold')
ax.set_ylabel('IR', fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT / 'tier2_vol_decomp.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ tier2_vol_decomp.png")

# ── Figure 5: Industry decomposition ──
if len(industry_results) >= 5:
    fig, ax = plt.subplots(figsize=(14, max(6, len(industry_results) * 0.35)))
    ind_labels = [str(i) for i, s in sorted_inds]
    ind_irs = [s['ic']['ir'] for i, s in sorted_inds]
    colors9 = ['#2E86AB' if ir > 0 else '#A23B72' for ir in ind_irs]
    ax.barh(range(len(ind_labels)), ind_irs, color=colors9, edgecolor='white')
    ax.set_yticks(range(len(ind_labels)))
    ax.set_yticklabels(ind_labels, fontsize=9)
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)
    ax.set_xlabel('IR', fontsize=12)
    ax.set_title('Industry Decomposition — hr_t4 IR by CSRC 2012 Industry', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig.savefig(OUT / 'tier2_industry_decomp.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ tier2_industry_decomp.png")

# ══════════════════════════════════════════════════════════════════════
# SAVE DATA
# ══════════════════════════════════════════════════════════════════════

# Threshold results
pd.DataFrame(threshold_results).T.to_csv(OUT / 'tier1_threshold_results.csv')

# Horizon results
horizon_flat = []
for t in ts:
    for h in ['1M', '3M', '6M', '12M']:
        if h in horizon_results[t]:
            d = horizon_results[t][h].copy()
            d['threshold'] = t
            d['horizon'] = h
            horizon_flat.append(d)
pd.DataFrame(horizon_flat).to_csv(OUT / 'tier1_horizon_results.csv')

elapsed = time.time() - t0
print(f"\n{'='*60}")
print(f"ALL EXPERIMENTS COMPLETE in {elapsed:.0f}s")
print(f"{'='*60}")
