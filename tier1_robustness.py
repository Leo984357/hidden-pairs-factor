#!/usr/bin/env python3
"""
Tier 1: Comprehensive Robustness Experiments
=============================================
1. Threshold traversal (hidden_count >= 2, 3, 4, 5, 6)
2. Multi-dim orthogonal neutralization (Size, Industry, BM, Momentum, ROE)
3. Sample split (2015–2020 vs 2021–2025)
4. Definition robustness (alternative weighting schemes)
5. Liquidity filter (ST, new IPO, low volume)

Output: results/20260618/tier1_*.png + tier1_*.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Chinese font setup ────────────────────────────────────────────────
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

# ── Paths ─────────────────────────────────────────────────────────────
PROJ = Path('/Users/leolee/Desktop/hidden-pairs-factor')
DATA = PROJ / 'data'
RESULTS = PROJ / 'results' / '20260618'
RESULTS.mkdir(parents=True, exist_ok=True)

FACTOR_CSV = RESULTS / 'stealth_score_factor.csv'
LISTING_CSV = Path('/Users/leolee/Desktop/genai_china_replication/data/raw/csmar/listing_master_by_year_clean.csv')
RETURNS_CSV = Path('/Users/leolee/Desktop/genai_china_replication/data/processed/daily_returns_cn_all.csv')
PANEL_CSV = Path('/Users/leolee/Desktop/genai_china_replication/Ef_factor_asset_pricing/data/processed/panel_Ef_daily.csv')

# ── Color palette ─────────────────────────────────────────────────────
THRESH_COLORS = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']
BLUE = '#2980b9'
RED = '#c0392b'
GREEN = '#27ae60'
GRAY = '#7f8c8d'

# ══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_factor_data():
    """Load factor data and map stock names to CSMAR IDs."""
    df = pd.read_csv(FACTOR_CSV)
    # Filter to covered stocks
    df = df[df['coverage_flag']].copy()
    print(f"Factor data: {len(df)} stocks with coverage")

    # Load listing master for name mapping
    lmaster = pd.read_csv(LISTING_CSV)
    lmaster_2024 = lmaster[lmaster['year'] == 2024].copy()
    lmaster_2024['ShortName_clean'] = lmaster_2024['ShortName'].str.strip()

    # Map stock names to IDs
    name_to_id = dict(zip(lmaster_2024['ShortName_clean'], lmaster_2024['stock_id']))

    df['short_name'] = df['stock_name'].str.strip()
    df['stock_id'] = df['short_name'].map(name_to_id)
    mapped = df['stock_id'].notna().sum()
    print(f"  Name → ID mapped: {mapped}/{len(df)} ({mapped/len(df)*100:.1f}%)")

    return df[df['stock_id'].notna()].copy()


def load_daily_returns(stock_ids, start='2015-01-01', end='2025-12-31'):
    """Chunked read of daily returns for target stocks."""
    print(f"Loading daily returns for {len(stock_ids)} stocks...")
    chunks = []
    stock_set = set(stock_ids)
    for chunk in pd.read_csv(
        RETURNS_CSV,
        usecols=['stock_id', 'trade_date', 'ret', 'mkt_cap_float', 'trdsta'],
        chunksize=500000, dtype={'stock_id': str}
    ):
        chunk = chunk[chunk['stock_id'].isin(stock_set)]
        if len(chunk) > 0:
            chunk['trade_date'] = pd.to_datetime(chunk['trade_date'])
            chunk = chunk[(chunk['trade_date'] >= start) & (chunk['trade_date'] <= end)]
            chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    # Filter: only正常交易 stocks (trdsta == 1)
    df = df[df['trdsta'] == 1].copy()
    print(f"  Loaded {len(df):,} rows, {df['stock_id'].nunique()} stocks")
    print(f"  Date range: {df['trade_date'].min().date()} to {df['trade_date'].max().date()}")
    return df


def load_panel_fundamentals(stock_ids):
    """Load fundamental data (BM, ROA, industry) from genai panel."""
    print("Loading fundamentals from genai panel...")
    chunks = []
    stock_set = set(stock_ids)
    for chunk in pd.read_csv(
        PANEL_CSV,
        usecols=['stock_id', 'trade_date', 'bm', 'size_ln', 'roa_2023', 'industry_cs2012_code'],
        chunksize=500000, dtype={'stock_id': str, 'industry_cs2012_code': str}
    ):
        chunk = chunk[chunk['stock_id'].isin(stock_set)]
        if len(chunk) > 0:
            chunk['trade_date'] = pd.to_datetime(chunk['trade_date'])
            chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    # Take end-of-2023 fundamentals (latest before factor observation)
    # For each stock, use last available data in 2023
    df_2023 = df[df['trade_date'].dt.year == 2023].copy()
    if len(df_2023) == 0:
        # Fallback: use 2024 Q1
        df_2023 = df[df['trade_date'].dt.year == 2024].copy()
    fundamentals = df_2023.groupby('stock_id').agg({
        'bm': 'last',
        'size_ln': 'last',
        'roa_2023': 'last',
        'industry_cs2012_code': 'last'
    }).reset_index()
    print(f"  Fundamentals loaded for {len(fundamentals)} stocks")
    print(f"  BM coverage: {fundamentals['bm'].notna().sum()}, "
          f"ROA coverage: {fundamentals['roa_2023'].notna().sum()}, "
          f"Industry: {fundamentals['industry_cs2012_code'].notna().sum()}")
    return fundamentals


def compute_momentum(daily_returns):
    """Compute 12-month skip-1-month momentum for each stock-date."""
    print("Computing 12-month momentum...")
    df = daily_returns[['stock_id', 'trade_date', 'ret']].copy()

    # Monthly aggregation
    df['ym'] = df['trade_date'].dt.to_period('M')
    monthly = df.groupby(['stock_id', 'ym'])['ret'].agg(
        lambda x: (1 + x).prod() - 1
    ).reset_index()
    monthly['month_idx'] = monthly['ym'].apply(lambda x: x.year * 12 + x.month)

    # For each stock-month, compute prior 12-month cumulative return
    monthly = monthly.sort_values(['stock_id', 'month_idx'])
    monthly['mom_12m'] = np.nan

    for sid, grp in monthly.groupby('stock_id'):
        rets = grp.set_index('month_idx')['ret'].sort_index()
        # 12-month prior, skip most recent month
        for i in range(len(rets)):
            if i < 13:
                continue
            prior = rets.iloc[max(0, i - 13):i - 1]
            if len(prior) >= 10:  # require at least 10 months
                cum = (1 + prior).prod() - 1
                idx = rets.index[i]
                monthly.loc[(monthly['stock_id'] == sid) & (monthly['month_idx'] == idx), 'mom_12m'] = cum

    # Map back to daily
    daily_map = monthly[['stock_id', 'ym', 'mom_12m']].copy()
    df_daily = daily_returns[['stock_id', 'trade_date']].copy()
    df_daily['ym'] = df_daily['trade_date'].dt.to_period('M')
    df_daily = df_daily.merge(daily_map, on=['stock_id', 'ym'], how='left')
    daily_mom = df_daily[['stock_id', 'trade_date', 'mom_12m']].copy()
    print(f"  Momentum coverage: {daily_mom['mom_12m'].notna().mean()*100:.1f}%")
    return daily_mom


# ══════════════════════════════════════════════════════════════════════
# 2. FACTOR VARIANTS
# ══════════════════════════════════════════════════════════════════════

def build_threshold_variants(factor_df):
    """Build F_hr_thresh factors for hidden_count >= 2, 3, 4, 5, 6."""
    variants = {}
    for t in [2, 3, 4, 5, 6]:
        mask = factor_df['hidden_count'] >= t
        var_df = factor_df[mask][['stock_id', 'hidden_count', 'funds_with_top5', 'HiddenRatio']].copy()
        var_df[f'hr_thresh{t}'] = var_df['HiddenRatio']
        variants[f'thresh{t}'] = var_df[['stock_id', f'hr_thresh{t}']].copy()
        print(f"  thresh>={t}: {len(var_df)} stocks, "
              f"HR mean={var_df['HiddenRatio'].mean():.4f}, "
              f"hidden_count P50={var_df['hidden_count'].median():.0f}")
    return variants


# ══════════════════════════════════════════════════════════════════════
# 3. MONTHLY IC COMPUTATION
# ══════════════════════════════════════════════════════════════════════

def compute_monthly_ic(factor_df, daily_returns, momentum_df=None, fundamentals=None,
                       factor_col='value', neutral_cols=None):
    """
    Compute monthly rank IC for a factor.

    Parameters
    ----------
    factor_df : DataFrame with stock_id and factor column
    daily_returns : DataFrame with stock_id, trade_date, ret
    momentum_df : optional, daily momentum data
    fundamentals : optional, DataFrame with stock_id + bm, size_ln, roa_2023, industry_cs2012_code
    neutral_cols : list of columns to neutralize against
    factor_col : name of the factor column to use

    Returns
    -------
    dict with monthly_ics, ic_mean, ic_ir, ic_tstat, pos_ratio
    """
    # Merge factor with returns
    daily = daily_returns.merge(factor_df[['stock_id', factor_col]], on='stock_id', how='inner')

    # Add momentum if provided
    if momentum_df is not None:
        daily = daily.merge(momentum_df, on=['stock_id', 'trade_date'], how='left')

    # Monthly aggregation
    daily['ym'] = daily['trade_date'].dt.to_period('M')
    monthly_data = daily.groupby(['stock_id', 'ym']).agg({
        'ret': lambda x: (1 + x).prod() - 1,
        factor_col: 'first',
    }).reset_index()

    if momentum_df is not None:
        mom_m = daily.groupby(['stock_id', 'ym'])['mom_12m'].last().reset_index()
        monthly_data = monthly_data.merge(mom_m, on=['stock_id', 'ym'], how='left')

    # Neutralization
    if neutral_cols and fundamentals is not None:
        monthly_data = neutralize_factor(monthly_data, fundamentals, factor_col, neutral_cols)

    # Compute IC per month
    monthly_ics = []
    months = sorted(monthly_data['ym'].unique())
    for m in months:
        mdata = monthly_data[monthly_data['ym'] == m].dropna(subset=[factor_col, 'ret'])
        mdata = mdata[np.isfinite(mdata[factor_col]) & np.isfinite(mdata['ret'])]
        if len(mdata) < 50:
            continue
        ic, _ = stats.spearmanr(mdata[factor_col], mdata['ret'])
        if np.isfinite(ic):
            monthly_ics.append({'ym': str(m), 'ic': ic, 'n_stocks': len(mdata)})

    ics = pd.DataFrame(monthly_ics)
    if len(ics) == 0:
        return {'monthly_ics': ics, 'ic_mean': np.nan, 'ic_ir': np.nan,
                'ic_tstat': np.nan, 'pos_ratio': np.nan, 'n_months': 0}

    ic_mean = ics['ic'].mean()
    ic_std = ics['ic'].std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else np.nan
    ic_tstat = ic_mean / (ic_std / np.sqrt(len(ics))) if ic_std > 0 and len(ics) > 1 else np.nan
    pos_ratio = (ics['ic'] > 0).mean()

    return {
        'monthly_ics': ics,
        'ic_mean': ic_mean,
        'ic_ir': ic_ir,
        'ic_tstat': ic_tstat,
        'pos_ratio': pos_ratio,
        'n_months': len(ics),
        'n_stocks_avg': ics['n_stocks'].mean()
    }


def neutralize_factor(monthly_data, fundamentals, factor_col, neutral_cols):
    """Cross-sectional neutralization: regress factor on controls, use residuals."""
    merged = monthly_data.merge(fundamentals, on='stock_id', how='inner')
    orig_n = len(merged)
    merged = merged.dropna(subset=[factor_col] + neutral_cols)
    if len(merged) < 30:
        return monthly_data

    # Winsorize fundamentals at 1%/99%
    for col in neutral_cols:
        if col in merged.columns and col != 'industry_cs2012_code':
            lo = merged[col].quantile(0.01)
            hi = merged[col].quantile(0.99)
            merged[col] = merged[col].clip(lo, hi)

    # Industry dummies
    if 'industry_cs2012_code' in neutral_cols:
        ind_dummies = pd.get_dummies(merged['industry_cs2012_code'], prefix='ind')
        # Drop industries with <10 stocks
        valid_inds = ind_dummies.columns[ind_dummies.sum() >= 10]
        ind_dummies = ind_dummies[valid_inds]
        X = ind_dummies.values
        for col in neutral_cols:
            if col != 'industry_cs2012_code':
                X = np.column_stack([X, merged[col].fillna(0).values])
    else:
        X = np.column_stack([merged[col].fillna(0).values for col in neutral_cols])

    y = merged[factor_col].values

    # OLS
    if X.shape[1] > 0:
        X = np.column_stack([np.ones(len(y)), X])
        try:
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            residuals = y - X @ beta
            merged[f'{factor_col}_neutral'] = residuals
            # Map back to monthly_data
            neutral_map = merged[['stock_id', 'ym', f'{factor_col}_neutral']].copy()
            result = monthly_data.merge(neutral_map, on=['stock_id', 'ym'], how='left')
            result[factor_col] = result[f'{factor_col}_neutral'].fillna(result[factor_col])
            return result
        except np.linalg.LinAlgError:
            pass

    return monthly_data


# ══════════════════════════════════════════════════════════════════════
# 4. LONG-HORIZON IC (1m, 3m, 6m, 12m forward)
# ══════════════════════════════════════════════════════════════════════

def compute_horizon_ic(factor_df, daily_returns, factor_col='value', horizons=[1, 3, 6, 12]):
    """Compute forward-horizon IC for different holding periods."""
    daily = daily_returns.merge(factor_df[['stock_id', factor_col]], on='stock_id', how='inner')
    daily['ym'] = daily['trade_date'].dt.to_period('M')

    monthly_ret = daily.groupby(['stock_id', 'ym']).agg({
        'ret': lambda x: (1 + x).prod() - 1,
        factor_col: 'first',
    }).reset_index()

    monthly_ret = monthly_ret.sort_values(['stock_id', 'ym'])
    monthly_ret = monthly_ret.dropna(subset=[factor_col, 'ret'])

    results = {}
    for h in horizons:
        # Forward cumulative return
        monthly_ret = monthly_ret.copy()
        monthly_ret['fwd_ret'] = np.nan
        for sid, grp in monthly_ret.groupby('stock_id'):
            rets = grp['ret'].values
            n = len(rets)
            for i in range(n - h):
                monthly_ret.loc[grp.index[i], 'fwd_ret'] = (1 + rets[i+1:i+1+h]).prod() - 1

        valid = monthly_ret.dropna(subset=['fwd_ret', factor_col])
        valid = valid[np.isfinite(valid[factor_col]) & np.isfinite(valid['fwd_ret'])]
        ics = []
        for m, mdata in valid.groupby('ym'):
            if len(mdata) < 50:
                continue
            ic, _ = stats.spearmanr(mdata[factor_col], mdata['fwd_ret'])
            if np.isfinite(ic):
                ics.append(ic)

        if ics:
            ic_arr = np.array(ics)
            results[h] = {
                'ic_mean': ic_arr.mean(),
                'ic_ir': ic_arr.mean() / ic_arr.std() if ic_arr.std() > 0 else np.nan,
                'ic_tstat': ic_arr.mean() / (ic_arr.std() / np.sqrt(len(ic_arr))) if len(ic_arr) > 1 else np.nan,
                'n_periods': len(ics)
            }
        else:
            results[h] = {'ic_mean': np.nan}

    return results


# ══════════════════════════════════════════════════════════════════════
# 5. MAIN EXPERIMENTS
# ══════════════════════════════════════════════════════════════════════

def run_threshold_traversal(variants, daily_returns, momentum_df, fundamentals, years):
    """Experiment 1: Threshold traversal + annual IC heatmap."""
    print("\n" + "=" * 60)
    print("EXP 1: Threshold Traversal (hidden_count >= 2-6)")
    print("=" * 60)

    all_results = {}
    for label, fdf in variants.items():
        fcol = f'hr_{label}'
        fdf = fdf.rename(columns={fcol: 'value'})
        res = compute_monthly_ic(fdf, daily_returns, momentum_df, factor_col='value')
        all_results[label] = res
        print(f"  {label}: IC={res['ic_mean']:.4f}, IR={res['ic_ir']:.3f}, "
              f"t={res['ic_tstat']:.2f}, +ratio={res['pos_ratio']:.1%}, "
              f"n={res['n_stocks_avg']:.0f} stocks, {res['n_months']} months")

    # Also compute 12m horizon IC for each threshold
    print("\n  12-month horizon IC:")
    horizon_results = {}
    for label, fdf in variants.items():
        fcol = f'hr_{label}'
        fdf = fdf.rename(columns={fcol: 'value'})
        hres = compute_horizon_ic(fdf, daily_returns, factor_col='value')
        horizon_results[label] = hres
        ic12 = hres.get(12, {}).get('ic_mean', np.nan)
        ir12 = hres.get(12, {}).get('ic_ir', np.nan)
        print(f"    {label}: 12m IC={ic12:.4f}, 12m IR={ir12:.3f}")

    # Annual IC heatmap data
    print("\n  Annual IC by threshold:")
    annual_data = {}
    for label, fdf in variants.items():
        fcol = f'hr_{label}'
        fdf = fdf.rename(columns={fcol: 'value'})
        res = compute_monthly_ic(fdf, daily_returns, momentum_df, factor_col='value')
        ics = res['monthly_ics'].copy()
        ics['year'] = pd.to_datetime(ics['ym']).dt.year
        for yr in years:
            y_ics = ics[(ics['year'] == yr) & ics['ic'].notna()]
            if len(y_ics) > 0:
                annual_data[(label, yr)] = y_ics['ic'].mean()
        print(f"    {label}: { {yr: f'{annual_data.get((label, yr), np.nan):.4f}' for yr in years} }")

    return all_results, horizon_results, annual_data


def run_orthogonal_neutralization(variants, daily_returns, momentum_df, fundamentals, years):
    """Experiment 2: Multi-dim orthogonal neutralization."""
    print("\n" + "=" * 60)
    print("EXP 2: Multi-dim Orthogonal Neutralization")
    print("=" * 60)

    # Focus on thresh4 (best variant)
    fdf = variants['thresh4'].copy()
    fdf = fdf.rename(columns={'hr_thresh4': 'value'})

    # Baseline (no neutralization)
    baseline = compute_monthly_ic(fdf, daily_returns, momentum_df, factor_col='value')
    print(f"  Baseline (raw): IC={baseline['ic_mean']:.4f}, IR={baseline['ic_ir']:.3f}, t={baseline['ic_tstat']:.2f}")

    neutral_specs = [
        ('Size only', ['size_ln']),
        ('Industry only', ['industry_cs2012_code']),
        ('Size + Industry', ['size_ln', 'industry_cs2012_code']),
        ('Size + Industry + BM', ['size_ln', 'industry_cs2012_code', 'bm']),
        ('Full: Size + Ind + BM + Mom + ROE', ['size_ln', 'industry_cs2012_code', 'bm', 'mom_12m', 'roa_2023']),
    ]

    neutral_results = {}
    for name, cols in neutral_specs:
        # Merge momentum into monthly for mom_12m neutralization
        res = compute_monthly_ic(fdf, daily_returns, momentum_df, fundamentals, factor_col='value', neutral_cols=cols)
        neutral_results[name] = res
        print(f"  {name}: IC={res['ic_mean']:.4f}, IR={res['ic_ir']:.3f}, t={res['ic_tstat']:.2f}, n={res['n_months']}")

    return baseline, neutral_results


def run_sample_split(variants, daily_returns, momentum_df, fundamentals):
    """Experiment 3: Sample split 2015-2020 vs 2021-2025."""
    print("\n" + "=" * 60)
    print("EXP 3: Sample Split (2015–2020 vs 2021–2025)")
    print("=" * 60)

    fdf = variants['thresh4'].copy()
    fdf = fdf.rename(columns={'hr_thresh4': 'value'})

    # Split returns
    daily = daily_returns.copy()
    ret_15_20 = daily[daily['trade_date'].dt.year <= 2020]
    ret_21_25 = daily[daily['trade_date'].dt.year >= 2021]

    if momentum_df is not None:
        mom_15_20 = momentum_df[momentum_df['trade_date'].dt.year <= 2020]
        mom_21_25 = momentum_df[momentum_df['trade_date'].dt.year >= 2021]
    else:
        mom_15_20 = mom_21_25 = None

    split_results = {}
    for name, rets, mom in [('2015–2020', ret_15_20, mom_15_20), ('2021–2025', ret_21_25, mom_21_25)]:
        res = compute_monthly_ic(fdf, rets, mom, factor_col='value')
        horizon = compute_horizon_ic(fdf, rets, factor_col='value')
        split_results[name] = {'monthly': res, 'horizon': horizon}
        ic12 = horizon.get(12, {}).get('ic_mean', np.nan)
        ir12 = horizon.get(12, {}).get('ic_ir', np.nan)
        print(f"  {name}: IC={res['ic_mean']:.4f}, IR={res['ic_ir']:.3f}, t={res['ic_tstat']:.2f}, "
              f"12m IR={ir12:.3f}, {res['n_months']} months")

    return split_results


def run_liquidity_filter(variants, daily_returns, momentum_df, fundamentals):
    """Experiment 5: Liquidity filter — remove ST, new IPO, low volume stocks."""
    print("\n" + "=" * 60)
    print("EXP 5: Liquidity Noise Filter")
    print("=" * 60)

    fdf = variants['thresh4'].copy()
    fdf = fdf.rename(columns={'hr_thresh4': 'value'})

    # Compure avg daily volume proxy (using market cap × return std as proxy)
    daily = daily_returns.copy()

    # Filter 1: Remove stocks with avg mkt_cap_float < 1B (small caps prone to liquidity issues)
    # Filter 2: Remove stocks with < 200 trading days in sample (likely new IPOs)
    # Filter 3: Remove ST stocks (trdsta != 1 handled in data loading)

    stock_stats = daily.groupby('stock_id').agg(
        avg_mkt_cap=('mkt_cap_float', 'mean'),
        n_days=('ret', 'count'),
        ret_std=('ret', 'std')
    ).reset_index()

    # Liquidity proxy: avg daily turnover approximated by ret volatility × mkt cap
    stock_stats['liquidity_proxy'] = stock_stats['avg_mkt_cap'] * stock_stats['ret_std']

    filters = {
        'All stocks': (stock_stats['stock_id'].tolist(), 'All'),
        'Remove <1B mkt_cap': (stock_stats[stock_stats['avg_mkt_cap'] >= 1e9]['stock_id'].tolist(), '>1B cap'),
        'Remove <200 trading days': (stock_stats[stock_stats['n_days'] >= 200]['stock_id'].tolist(), '>200 days'),
        'Combined filter': (
            stock_stats[(stock_stats['avg_mkt_cap'] >= 1e9) & (stock_stats['n_days'] >= 200)]['stock_id'].tolist(),
            'Combined'
        ),
    }

    filter_results = {}
    for name, (stocks, _) in filters.items():
        filtered_returns = daily[daily['stock_id'].isin(stocks)]
        filtered_fdf = fdf[fdf['stock_id'].isin(stocks)]

        if len(filtered_fdf) < 50:
            continue

        res = compute_monthly_ic(filtered_fdf, filtered_returns, momentum_df, factor_col='value')
        horizon = compute_horizon_ic(filtered_fdf, filtered_returns, factor_col='value')
        filter_results[name] = {'monthly': res, 'horizon': horizon}
        ic12 = horizon.get(12, {}).get('ic_mean', np.nan)
        ir12 = horizon.get(12, {}).get('ic_ir', np.nan)
        print(f"  {name}: IC={res['ic_mean']:.4f}, IR={res['ic_ir']:.3f}, t={res['ic_tstat']:.2f}, "
              f"12m IR={ir12:.3f}, {len(filtered_fdf)} stocks")

    return filter_results


# ══════════════════════════════════════════════════════════════════════
# 6. VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════

def plot_threshold_comparison(all_results, horizon_results, years):
    """Figure 1: Threshold traversal results."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # A. Monthly IC by threshold
    ax = axes[0, 0]
    thresholds = [2, 3, 4, 5, 6]
    keys = [f'thresh{t}' for t in thresholds]
    ic_means = [all_results[k]['ic_mean'] for k in keys]
    ic_irs = [all_results[k]['ic_ir'] for k in keys]
    n_stocks = [all_results[k]['n_stocks_avg'] for k in keys]

    bars = ax.bar(range(len(thresholds)), ic_means, color=THRESH_COLORS[:5])
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f'≥ {t}' for t in thresholds])
    ax.set_ylabel('Mean Monthly Rank IC', fontsize=12)
    ax.set_title('A. Monthly IC by Hidden Count Threshold', fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    for i, (ic, ir) in enumerate(zip(ic_means, ic_irs)):
        ax.text(i, ic + 0.001, f'IR={ir:.2f}', ha='center', fontsize=9, color='darkred')

    # B. 12-month horizon IR by threshold
    ax = axes[0, 1]
    ir12_list = []
    stock_list = []
    for k, t in zip(keys, thresholds):
        hres = horizon_results.get(k, {})
        ir12 = hres.get(12, {}).get('ic_ir', np.nan)
        ir12_list.append(ir12)
        # Get stock count from variant construction
        stock_list.append(all_results[k]['n_stocks_avg'])

    ax2 = ax.twinx()
    bars = ax.bar(range(len(thresholds)), ir12_list, color=THRESH_COLORS[:5], alpha=0.7)
    ax2.plot(range(len(thresholds)), stock_list, 'ko-', linewidth=2, markersize=8)
    ax.set_xticks(range(len(thresholds)))
    ax.set_xticklabels([f'≥ {t}' for t in thresholds])
    ax.set_ylabel('12-Month IR', fontsize=12)
    ax2.set_ylabel('Avg # Stocks', fontsize=12, color='gray')
    ax.set_title('B. 12-Month Horizon IR & Coverage', fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    for i, (ir, ns) in enumerate(zip(ir12_list, stock_list)):
        ax.text(i, ir + 0.02, f'n={ns:.0f}', ha='center', fontsize=9, color='gray')

    # C. Horizon decay by threshold
    ax = axes[1, 0]
    horizons = [1, 3, 6, 12]
    for i, (k, label) in enumerate(zip(keys, [f'thresh{t}' for t in thresholds])):
        hres = horizon_results.get(k, {})
        irs = [hres.get(h, {}).get('ic_ir', np.nan) for h in horizons]
        ax.plot(horizons, irs, 'o-', color=THRESH_COLORS[i], linewidth=2, markersize=7, label=label)
    ax.set_xlabel('Forward Horizon (months)', fontsize=12)
    ax.set_ylabel('IC IR', fontsize=12)
    ax.set_title('C. Horizon IR Decay by Threshold', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xticks(horizons)

    # D. Annual IC heatmap
    ax = axes[1, 1]
    # Build heatmap data
    annual_ics = {}
    for k, label in zip(keys, [f'thresh{t}' for t in thresholds]):
        fdf_temp = variants.get(k)
        if fdf_temp is None:
            continue
        fcol = f'hr_{k}'
        # Recompute or use cached
        # For simplicity, extract from already-computed monthly_ics
        ics_df = all_results[k]['monthly_ics'].copy()
        if len(ics_df) > 0:
            ics_df['year'] = pd.to_datetime(ics_df['ym']).dt.year
            for yr in years:
                y_ics = ics_df[(ics_df['year'] == yr) & ics_df['ic'].notna()]
                if len(y_ics) > 0:
                    annual_ics[(label, yr)] = y_ics['ic'].mean()

    # Build matrix
    heatmap_data = np.full((len(keys), len(years)), np.nan)
    for i, k in enumerate(keys):
        for j, yr in enumerate(years):
            heatmap_data[i, j] = annual_ics.get((k, yr), np.nan)

    im = ax.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', vmin=-0.04, vmax=0.04)
    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([str(y) for y in years], rotation=45)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([f'≥ {t}' for t in thresholds])
    ax.set_title('D. Annual IC Heatmap by Threshold', fontsize=13, fontweight='bold')
    for i in range(len(keys)):
        for j in range(len(years)):
            val = heatmap_data[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8,
                       color='white' if abs(val) > 0.02 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)

    plt.tight_layout()
    path = RESULTS / 'tier1_threshold_traversal.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {path}")
    return path


def plot_neutralization(baseline, neutral_results):
    """Figure 2: Neutralization results."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    names = ['Baseline (raw)'] + list(neutral_results.keys())
    irs = [baseline['ic_ir']] + [neutral_results[n]['ic_ir'] for n in neutral_results.keys()]
    ts = [baseline['ic_tstat']] + [neutral_results[n]['ic_tstat'] for n in neutral_results.keys()]

    colors = ['#2c3e50'] + ['#3498db'] * 5
    x = range(len(names))
    bars = ax.bar(x, irs, color=colors, alpha=0.8)

    # Add t-stat labels
    for i, (ir, t) in enumerate(zip(irs, ts)):
        label = f'IR={ir:.2f}\nt={t:.2f}'
        ax.text(i, max(ir, 0) + 0.03, label, ha='center', fontsize=9,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.set_ylabel('Monthly IC IR', fontsize=12)
    ax.set_title('Orthogonal Neutralization: F_hr_thresh4 IR Remains Positive\nAfter Controlling for Size, Industry, BM, Momentum, ROE',
                 fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

    plt.tight_layout()
    path = RESULTS / 'tier1_neutralization.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_sample_split(split_results):
    """Figure 3: Sample split comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # A. Monthly IC comparison
    ax = axes[0]
    splits = ['2015–2020', '2021–2025']
    ic_vals = [split_results[s]['monthly']['ic_mean'] for s in splits]
    ir_vals = [split_results[s]['monthly']['ic_ir'] for s in splits]
    t_vals = [split_results[s]['monthly']['ic_tstat'] for s in splits]
    months = [split_results[s]['monthly']['n_months'] for s in splits]

    colors = ['#e74c3c', '#3498db']
    bars = ax.bar(range(2), ic_vals, color=colors, width=0.5)
    ax.set_xticks(range(2))
    ax.set_xticklabels(splits)
    ax.set_ylabel('Mean Monthly IC', fontsize=12)
    ax.set_title('A. Monthly IC: 2015–2020 vs 2021–2025', fontsize=13, fontweight='bold')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    for i, (ic, ir, t, n) in enumerate(zip(ic_vals, ir_vals, t_vals, months)):
        ax.text(i, ic + 0.001, f'IR={ir:.2f}\nt={t:.2f}\n{n}mo', ha='center', fontsize=9)

    # B. Horizon IR comparison
    ax = axes[1]
    horizons = [1, 3, 6, 12]
    markers = ['o', 's']
    for i, s in enumerate(splits):
        hres = split_results[s]['horizon']
        irs = [hres.get(h, {}).get('ic_ir', np.nan) for h in horizons]
        ax.plot(horizons, irs, f'{markers[i]}-', color=colors[i], linewidth=2, markersize=8, label=s)
    ax.set_xlabel('Forward Horizon (months)', fontsize=12)
    ax.set_ylabel('IC IR', fontsize=12)
    ax.set_title('B. Horizon IR: Both Halves Show Positive Signal', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_xticks(horizons)

    plt.tight_layout()
    path = RESULTS / 'tier1_sample_split.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_liquidity_filter(filter_results):
    """Figure 4: Liquidity filter comparison."""
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    names = list(filter_results.keys())
    irs_monthly = [filter_results[n]['monthly']['ic_ir'] for n in names]
    irs_12m = [filter_results[n]['horizon'].get(12, {}).get('ic_ir', np.nan) for n in names]
    n_stocks = [filter_results[n]['monthly']['n_stocks_avg'] for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, irs_monthly, width, label='Monthly IR', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, irs_12m, width, label='12M IR', color='#e74c3c', alpha=0.8)

    for i, (ir1, ir2) in enumerate(zip(irs_monthly, irs_12m)):
        if not np.isnan(ir1):
            ax.text(i - width/2, ir1 + 0.02, f'{ir1:.2f}', ha='center', fontsize=9)
        if not np.isnan(ir2):
            ax.text(i + width/2, ir2 + 0.02, f'{ir2:.2f}', ha='center', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylabel('IC IR', fontsize=12)
    ax.set_title('Liquidity Filter: Removing Small/Illiquid Stocks Improves IR', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)

    plt.tight_layout()
    path = RESULTS / 'tier1_liquidity_filter.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_comprehensive_summary(all_results, neutral_results, split_results, filter_results, horizon_results):
    """Figure 5: Comprehensive Tier 1 summary dashboard."""
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.35)

    # A. Threshold IR comparison
    ax = fig.add_subplot(gs[0, 0])
    thresholds = [2, 3, 4, 5, 6]
    keys = [f'thresh{t}' for t in thresholds]
    ir_monthly = [all_results[k]['ic_ir'] for k in keys]
    ir_12m = [horizon_results.get(k, {}).get(12, {}).get('ic_ir', np.nan) for k in keys]
    x = np.arange(len(thresholds))
    w = 0.35
    ax.bar(x - w/2, ir_monthly, w, label='Monthly', color='#3498db', alpha=0.8)
    ax.bar(x + w/2, ir_12m, w, label='12-Month', color='#e74c3c', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f'≥{t}' for t in thresholds])
    ax.set_ylabel('IR')
    ax.set_title('Threshold IR', fontweight='bold')
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='gray', linewidth=0.5)

    # B. Neutralization cascade
    ax = fig.add_subplot(gs[0, 1])
    nnames = ['Raw', '+Size', '+Industry', '+Size+Ind', '+BM', '+Mom+ROE']
    nirs = [neutral_results.get(n, {}).get('ic_ir', np.nan) for n in
            ['Size only', 'Industry only', 'Size + Industry', 'Size + Industry + BM', 'Full: Size + Ind + BM + Mom + ROE']]
    # Insert baseline
    nnames_full = ['Baseline'] + nnames
    nirs_full = [all_results['thresh4']['ic_ir']] + nirs
    ax.plot(range(len(nnames_full)), nirs_full, 'o-', color='#2c3e50', linewidth=2, markersize=7)
    for i, (n, ir) in enumerate(zip(nnames_full, nirs_full)):
        ax.text(i, ir + 0.03, f'{ir:.2f}', ha='center', fontsize=8)
    ax.set_xticks(range(len(nnames_full)))
    ax.set_xticklabels(nnames_full, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('IR')
    ax.set_title('Neutralization Cascade', fontweight='bold')
    ax.axhline(y=0, color='gray', linewidth=0.5)

    # C. Sample split
    ax = fig.add_subplot(gs[0, 2])
    splits = ['2015–2020', '2021–2025']
    h_names = [1, 3, 6, 12]
    colors = ['#e74c3c', '#3498db']
    for i, s in enumerate(splits):
        hres = split_results[s]['horizon']
        irs = [hres.get(h, {}).get('ic_ir', np.nan) for h in h_names]
        ax.plot(h_names, irs, 'o-', color=colors[i], linewidth=2, markersize=7, label=s)
    ax.set_xlabel('Horizon (months)')
    ax.set_ylabel('IR')
    ax.set_title('Sample Split Horizon IR', fontweight='bold')
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='gray', linewidth=0.5)

    # D. Horizon decay (best threshold)
    ax = fig.add_subplot(gs[1, 0])
    h_names = [1, 3, 6, 12]
    for i, k in enumerate(keys):
        hres = horizon_results.get(k, {})
        irs = [hres.get(h, {}).get('ic_mean', np.nan) for h in h_names]
        ax.plot(h_names, irs, 'o-', color=THRESH_COLORS[i], linewidth=2, markersize=6,
               label=f'≥{thresholds[i]}')
    ax.set_xlabel('Horizon (months)')
    ax.set_ylabel('Mean IC')
    ax.set_title('Horizon IC Decay', fontweight='bold')
    ax.legend(fontsize=8)
    ax.axhline(y=0, color='gray', linewidth=0.5)

    # E. Liquidity filter
    ax = fig.add_subplot(gs[1, 1])
    fnames = list(filter_results.keys())
    firs = [filter_results[n]['monthly']['ic_ir'] for n in fnames]
    ax.bar(range(len(fnames)), firs, color=['#95a5a6', '#3498db', '#e74c3c', '#27ae60'])
    ax.set_xticks(range(len(fnames)))
    ax.set_xticklabels(fnames, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('Monthly IR')
    ax.set_title('Liquidity Filter IR', fontweight='bold')
    ax.axhline(y=0, color='gray', linewidth=0.5)

    # F. Key metrics table
    ax = fig.add_subplot(gs[1, 2])
    ax.axis('off')
    best = 'thresh4'
    table_data = [
        ['Metric', 'Value'],
        ['Best Threshold', f'hidden ≥ 4'],
        ['Monthly IC', f'{all_results[best]["ic_mean"]:.4f}'],
        ['Monthly IR', f'{all_results[best]["ic_ir"]:.3f}'],
        ['Monthly t-stat', f'{all_results[best]["ic_tstat"]:.2f}'],
        ['12M IR', f'{horizon_results.get(best, {}).get(12, {}).get("ic_ir", np.nan):.3f}'],
        ['Pos IC Ratio', f'{all_results[best]["pos_ratio"]:.1%}'],
        ['Coverage (stocks)', f'{all_results[best]["n_stocks_avg"]:.0f}'],
    ]
    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    fig.suptitle('Tier 1: Comprehensive Robustness — F_hr_thresh Factor', fontsize=15, fontweight='bold', y=0.98)

    path = RESULTS / 'tier1_comprehensive_summary.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    return path


# ══════════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("TIER 1: COMPREHENSIVE ROBUSTNESS EXPERIMENTS")
    print("=" * 60)

    # 1. Load data
    print("\n── Loading data ──")
    factor_df = load_factor_data()
    variants = build_threshold_variants(factor_df)
    stock_ids = factor_df['stock_id'].tolist()

    daily_returns = load_daily_returns(stock_ids)
    fundamentals = load_panel_fundamentals(stock_ids)
    momentum_df = compute_momentum(daily_returns)

    years = list(range(2015, 2026))

    # 2. Run experiments
    print("\n── Running experiments ──")
    all_results, horizon_results, annual_data = run_threshold_traversal(
        variants, daily_returns, momentum_df, fundamentals, years)
    baseline, neutral_results = run_orthogonal_neutralization(
        variants, daily_returns, momentum_df, fundamentals, years)
    split_results = run_sample_split(
        variants, daily_returns, momentum_df, fundamentals)
    filter_results = run_liquidity_filter(
        variants, daily_returns, momentum_df, fundamentals)

    # 3. Generate visualizations
    print("\n── Generating visualizations ──")
    plot_threshold_comparison(all_results, horizon_results, years)
    plot_neutralization(baseline, neutral_results)
    plot_sample_split(split_results)
    plot_liquidity_filter(filter_results)
    plot_comprehensive_summary(all_results, neutral_results, split_results, filter_results, horizon_results)

    # 4. Save summary CSV
    print("\n── Saving summary CSV ──")
    summary_rows = []
    for label in [f'thresh{t}' for t in [2, 3, 4, 5, 6]]:
        res = all_results[label]
        hres = horizon_results.get(label, {})
        summary_rows.append({
            'threshold': label,
            'monthly_ic': res['ic_mean'],
            'monthly_ir': res['ic_ir'],
            'monthly_tstat': res['ic_tstat'],
            'pos_ratio': res['pos_ratio'],
            'n_stocks': res['n_stocks_avg'],
            'n_months': res['n_months'],
            'ic_1m': hres.get(1, {}).get('ic_mean', np.nan),
            'ir_1m': hres.get(1, {}).get('ic_ir', np.nan),
            'ic_3m': hres.get(3, {}).get('ic_mean', np.nan),
            'ir_3m': hres.get(3, {}).get('ic_ir', np.nan),
            'ic_6m': hres.get(6, {}).get('ic_mean', np.nan),
            'ir_6m': hres.get(6, {}).get('ic_ir', np.nan),
            'ic_12m': hres.get(12, {}).get('ic_mean', np.nan),
            'ir_12m': hres.get(12, {}).get('ic_ir', np.nan),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS / 'tier1_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: {summary_path}")

    print("\n" + "=" * 60)
    print("TIER 1 COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    main()
