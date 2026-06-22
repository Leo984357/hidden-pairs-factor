"""
Tier 1 Diagnostics — Hidden Pairs Factor Validation
═══════════════════════════════════════════════════════════════════════

Five checks on existing (single cross-section) data:
  1. Fama-MacBeth cross-sectional regressions
  2. Factor correlation matrix
  3. Double-sort: Size quintile × Breadth tertile
  4. Out-of-sample split: 2015-2020 train → 2021-2025 test
  5. Yearly IC heatmap with market regime annotation

Purpose: Determine if Breadth/RecognitionSpread signal survives
control for Size/BM/Momentum/Vol, and if there's any real OOS signal.

OUTPUT: results/20260622/tier1_diagnostics.png + tier1_diagnostics.csv
"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm
import time
import os

# ── Config ──────────────────────────────────────────────────────────
GENAI = '/Users/leolee/Desktop/genai_china_replication'
FACTOR_FILE = '/Users/leolee/Desktop/hidden-pairs-factor/results/20260618/stealth_score_factor.csv'
OUTDIR = '/Users/leolee/Desktop/hidden-pairs-factor/results/20260622'
os.makedirs(OUTDIR, exist_ok=True)

START = time.time()
print("=" * 70)
print("TIER 1 DIAGNOSTICS — Factor Validation")
print("=" * 70)

# ── Font ────────────────────────────────────────────────────────────
import matplotlib.font_manager as fm
font_names = [f.name for f in fm.fontManager.ttflist]
cn_fonts = [f for f in font_names if any(k in f for k in 
    ['Hei', 'Song', 'Ming', 'Fang', 'Kai', 'PingFang', 'Noto Sans CJK'])]
CN_FONT = cn_fonts[0] if cn_fonts else 'sans-serif'
plt.rcParams['font.family'] = CN_FONT

# ══════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════
print("\n[1/6] Loading and preparing data...")

# ── Factor data ──
factor_df = pd.read_csv(FACTOR_FILE)
factor_df = factor_df[factor_df['coverage_flag']].copy()

# Map names to CSMAR IDs
master = pd.read_csv(f'{GENAI}/data/raw/csmar/listing_master_by_year_clean.csv')
master_2024 = master[master['year'] == 2024][['stock_id', 'ShortName']].copy()
master_2024['ShortName_clean'] = master_2024['ShortName'].str.replace('*', '').str.replace(' ', '').str.strip()
name_map = dict(zip(master_2024['ShortName_clean'], master_2024['stock_id']))
factor_df['stock_id'] = factor_df['stock_name'].str.replace('*', '').str.replace(' ', '').str.strip().map(name_map)
factor_df = factor_df.dropna(subset=['stock_id']).copy()

# Derived factors
factor_df['hidden_count'] = factor_df['total_funds'] - factor_df['funds_with_top5']
factor_df['hidden_ratio'] = factor_df['hidden_count'] / factor_df['total_funds'].replace(0, np.nan)
factor_df['coverage_breadth'] = np.log1p(factor_df['total_funds'])
factor_df['recognition_spread'] = factor_df['coverage_breadth'] * (1 - 2 * factor_df['hidden_ratio'])

BREADTH = factor_df.set_index('stock_id')['coverage_breadth']
HR = factor_df.set_index('stock_id')['hidden_ratio']
RECOG = factor_df.set_index('stock_id')['recognition_spread']
VISIBLE = factor_df.set_index('stock_id')['coverage_breadth'] * (1 - factor_df.set_index('stock_id')['hidden_ratio'])
STOCK_IDS_ALL = factor_df['stock_id'].unique()
print(f"  Factor data: {len(factor_df)} stocks")

# ── Returns → monthly ──
print("  Loading returns...")
ret_data = []
for chunk in pd.read_csv(f'{GENAI}/data/processed/daily_returns_cn_all.csv',
    usecols=['stock_id', 'trade_date', 'ret', 'mkt_cap_float'],
    dtype={'stock_id': str}, chunksize=300000):
    chunk['trade_date'] = pd.to_datetime(chunk['trade_date'])
    chunk = chunk[(chunk['stock_id'].isin(STOCK_IDS_ALL)) & 
                  chunk['trade_date'].dt.year.between(2015, 2025)]
    if len(chunk):
        ret_data.append(chunk)
returns = pd.concat(ret_data, ignore_index=True)
returns['ym'] = returns['trade_date'].dt.to_period('M')

# Monthly returns
monthly = returns.groupby(['stock_id', 'ym'])['ret'].apply(lambda x: (1+x).prod()-1).reset_index()
monthly['ym_dt'] = monthly['ym'].dt.to_timestamp()
ret_matrix = monthly.pivot(index='stock_id', columns='ym_dt', values='ret')
ret_matrix.columns = pd.to_datetime(ret_matrix.columns)
print(f"  ret_matrix: {ret_matrix.shape}")

# ── Fundamentals panel ──
print("  Loading fundamentals...")
panel = pd.read_csv(f'{GENAI}/Ef_factor_asset_pricing/data/processed/panel_Ef_daily.csv',
    usecols=['stock_id', 'trade_date', 'bm', 'size_ln', 'roa_2023', 'ret',
             'mkt_cap_float'],
    dtype={'stock_id': str})
panel['trade_date'] = pd.to_datetime(panel['trade_date'])

# ── Build monthly firm characteristics ──
# For each year-month, get latest available char
print("  Building monthly characteristics panel...")
monthly_chars = {}
for (s, y), grp in panel.groupby(['stock_id', panel['trade_date'].dt.year]):
    grp_ym = grp.copy()
    grp_ym['month'] = grp_ym['trade_date'].dt.month
    latest = grp_ym.sort_values('trade_date').groupby('month').tail(1)
    for _, row in latest.iterrows():
        ym_key = pd.Timestamp(year=row['trade_date'].year, month=row['trade_date'].month, day=1)
        key = (row['stock_id'], ym_key)
        monthly_chars[key] = {
            'size_ln': row['size_ln'], 'bm': row['bm'],
            'roa_2023': row['roa_2023']
        }
chars_df = pd.DataFrame(monthly_chars).T
chars_df.index.names = ['stock_id', 'ym_dt']
chars_df = chars_df.reset_index()
chars_df['ym_dt'] = pd.to_datetime(chars_df['ym_dt'])
print(f"  Month-char rows: {len(chars_df):,}")

# ── Momentum ──
print("  Computing monthly momentum (12-1)...")
mom_dfs = []
for s in STOCK_IDS_ALL:
    if s in ret_matrix.index:
        row = ret_matrix.loc[s].dropna()
        for i in range(12, len(row)):
            mom = (1 + row.iloc[i-12:i]).prod() - 1
            ym = row.index[i]
            mom_dfs.append({'stock_id': s, 'ym_dt': ym, 'mom_12m': mom})
mom_df = pd.DataFrame(mom_dfs)
mom_df['ym_dt'] = pd.to_datetime(mom_df['ym_dt'])
print(f"  Momentum rows: {len(mom_df):,}")

# ── Volatility (trailing 12-month) ──
print("  Computing monthly volatility (trailing 12m)...")
vol_dfs = []
for s in STOCK_IDS_ALL:
    if s in ret_matrix.index:
        row = ret_matrix.loc[s].dropna()
        for i in range(12, len(row)):
            vol = row.iloc[i-12:i].std() * np.sqrt(12)
            ym = row.index[i]
            vol_dfs.append({'stock_id': s, 'ym_dt': ym, 'vol_12m': vol})
vol_df = pd.DataFrame(vol_dfs)
vol_df['ym_dt'] = pd.to_datetime(vol_df['ym_dt'])
print(f"  Vol rows: {len(vol_df):,}")

# ── Build monthly panel with ALL factors ──
print("  Merging monthly panel...")
panel_m = monthly.copy()
panel_m['ym_dt'] = pd.to_datetime(panel_m['ym_dt'])

# Attach factor data (static in this cross-section)
for name, series in [('breadth', BREADTH), ('hidden_ratio', HR), 
                      ('recognition_spread', RECOG), ('visible_breadth', VISIBLE)]:
    panel_m[name] = panel_m['stock_id'].map(series)

# Attach time-varying characteristics
panel_m = panel_m.merge(chars_df, on=['stock_id', 'ym_dt'], how='left')
panel_m = panel_m.merge(mom_df, on=['stock_id', 'ym_dt'], how='left')
panel_m = panel_m.merge(vol_df, on=['stock_id', 'ym_dt'], how='left')

panel_m = panel_m.dropna(subset=['breadth', 'size_ln', 'bm', 'mom_12m', 'vol_12m',
                                  'ret', 'recognition_spread'])
print(f"  Final panel: {len(panel_m):,} rows × {panel_m.shape[1]} cols")
print(f"  Date range: {panel_m['ym_dt'].min().date()} to {panel_m['ym_dt'].max().date()}")

# ── Rank-standardize for cross-section ──
print("  Rank-standardizing factors...")
def rank_std_col(df, col):
    """Rank standardize to [-1, 1] within each month."""
    return df.groupby('ym_dt')[col].transform(lambda x: (x.rank(pct=True) - 0.5) * 2)

for col in ['breadth', 'hidden_ratio', 'recognition_spread', 'visible_breadth',
            'size_ln', 'bm', 'mom_12m', 'vol_12m', 'roa_2023']:
    panel_m[f'{col}_z'] = rank_std_col(panel_m, col)

# ══════════════════════════════════════════════════════════════════════
# 2. FACTOR CORRELATION MATRIX
# ══════════════════════════════════════════════════════════════════════
print("\n[2/6] Factor correlation matrix...")

corr_cols = ['breadth', 'hidden_ratio', 'recognition_spread', 'visible_breadth',
             'size_ln', 'bm', 'mom_12m', 'vol_12m', 'roa_2023']
corr_labels = ['Breadth', 'HiddenRatio', 'RecogSpread', 'VisibleBreadth',
               'Size(ln)', 'BM', 'Mom(12m)', 'Vol(12m)', 'ROA']

corr_rets = panel_m.groupby('ym_dt').apply(
    lambda g: g[corr_cols].corr(), include_groups=False
).groupby(level=1).mean()

corr_data = corr_rets.values
corr_annot = np.array([[f'{v:.2f}' for v in row] for row in corr_data])

# ══════════════════════════════════════════════════════════════════════
# 3. FAMA-MACBETH REGRESSIONS
# ══════════════════════════════════════════════════════════════════════
print("\n[3/6] Fama-MacBeth cross-sectional regressions...")

fm_results = []

# Model specifications
specs = [
    ('M1: Breadth only', ['breadth_z']),
    ('M2: + Size', ['breadth_z', 'size_ln_z']),
    ('M3: + Size + BM', ['breadth_z', 'size_ln_z', 'bm_z']),
    ('M4: + Size + BM + Mom', ['breadth_z', 'size_ln_z', 'bm_z', 'mom_12m_z']),
    ('M5: + Size + BM + Mom + Vol', ['breadth_z', 'size_ln_z', 'bm_z', 'mom_12m_z', 'vol_12m_z']),
    ('M6: RecogSpread only', ['recognition_spread_z']),
    ('M7: RecogSpread + Size + BM + Mom + Vol', 
     ['recognition_spread_z', 'size_ln_z', 'bm_z', 'mom_12m_z', 'vol_12m_z']),
]

for model_name, factors in specs:
    coefs = []
    for ym, grp in panel_m.groupby('ym_dt'):
        if len(grp) < 50:
            continue
        y = grp['ret']
        X = sm.add_constant(grp[factors].values)
        try:
            res = sm.OLS(y, X, missing='drop').fit()
            coefs.append([ym] + list(res.params))
        except:
            continue
    
    if not coefs:
        continue
    
    coefs_df = pd.DataFrame(coefs, columns=['ym_dt'] + ['const'] + factors)
    # Time-series means and Newey-West t-stats
    row_result = {'model': model_name, 'n_months': len(coefs_df)}
    for col in ['const'] + factors:
        series = coefs_df[col].dropna()
        mean = series.mean()
        se = series.std(ddof=1) / np.sqrt(len(series))
        tstat = mean / se if se > 0 else np.nan
        row_result[f'{col}_coef'] = mean
        row_result[f'{col}_t'] = tstat
    fm_results.append(row_result)

fm_df = pd.DataFrame(fm_results)
print(fm_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════
# 4. DOUBLE-SORT: Size Quintile × Breadth Tertile
# ══════════════════════════════════════════════════════════════════════
print("\n[4/6] Double-sort: Size quintile × Breadth tertile...")

def double_sort_returns(panel, size_col, factor_col, ret_col='ret'):
    """Compute value-weighted returns for 5×3 portfolios."""
    panel = panel.dropna(subset=[size_col, factor_col, ret_col])
    results = []
    for ym, grp in panel.groupby('ym_dt'):
        if len(grp) < 30:
            continue
        grp['size_q'] = pd.qcut(grp[size_col], 5, labels=False, duplicates='drop') + 1
        grp['factor_t'] = pd.qcut(grp[factor_col], 3, labels=False, duplicates='drop') + 1
        for sq in range(1, 6):
            for ft in range(1, 4):
                sub = grp[(grp['size_q'] == sq) & (grp['factor_t'] == ft)]
                if len(sub) < 3:
                    continue
                eq_ret = sub[ret_col].mean()
                results.append({'ym_dt': ym, 'size_q': sq, 'factor_t': ft, 'ret': eq_ret})
    return pd.DataFrame(results)

ds_port = double_sort_returns(panel_m, 'size_ln', 'breadth')

# Compute stats for each portfolio
ds_stats = []
for sq in range(1, 6):
    for ft in range(1, 3):
        sub = ds_port[(ds_port['size_q'] == sq) & (ds_port['factor_t'] == ft)]
        if len(sub) < 10:
            continue
        mean_ret = sub['ret'].mean()
        vol = sub['ret'].std()
        sharpe = mean_ret / vol * np.sqrt(12) if vol > 0 else np.nan
        ds_stats.append({'size_q': sq, 'factor_t': ft, 'count': len(sub),
                         'mean_ret': mean_ret, 'vol': vol, 'sharpe': sharpe})
ds_stats_df = pd.DataFrame(ds_stats)

# Breadth premium within each size quintile
print("\n  Breadth premium (T3-T1) within each size quintile:")
for sq in range(1, 6):
    hi = ds_stats_df[(ds_stats_df['size_q'] == sq) & (ds_stats_df['factor_t'] == 2)]
    lo = ds_stats_df[(ds_stats_df['size_q'] == sq) & (ds_stats_df['factor_t'] == 1)]
    if len(hi) and len(lo):
        prem = hi['mean_ret'].values[0] - lo['mean_ret'].values[0]
        print(f"    Size Q{sq}: Breadth premium = {prem*100:.2f}%/month")

# ══════════════════════════════════════════════════════════════════════
# 5. OUT-OF-SAMPLE SPLIT
# ══════════════════════════════════════════════════════════════════════
print("\n[5/6] Out-of-sample split: 2015-2020 train → 2021-2025 test...")

train_mask = panel_m['ym_dt'].dt.year <= 2020
test_mask = panel_m['ym_dt'].dt.year >= 2021

oos_results = []
for factor_name in ['breadth_z', 'recognition_spread_z', 'hidden_ratio_z']:
    for period, mask in [('In-Sample\n(2015-2020)', train_mask), 
                          ('Out-of-Sample\n(2021-2025)', test_mask)]:
        sub = panel_m[mask].dropna(subset=[factor_name, 'ret'])
        ics = sub.groupby('ym_dt').apply(
            lambda g: stats.spearmanr(g[factor_name], g['ret'])[0] if len(g) > 30 else np.nan,
            include_groups=False
        ).dropna()
        if len(ics) < 3:
            continue
        mean_ic = ics.mean()
        ir = mean_ic / ics.std(ddof=1) if ics.std() > 0 else np.nan
        t_stat = mean_ic / (ics.std(ddof=1) / np.sqrt(len(ics)))
        oos_results.append({
            'factor': factor_name.replace('_z', ''),
            'period': period,
            'n_months': len(ics),
            'mean_ic': mean_ic,
            'ic_ir': ir,
            't_stat': t_stat,
            'pos_rate': (ics > 0).mean()
        })

oos_df = pd.DataFrame(oos_results)
print(oos_df.to_string(index=False))

# ══════════════════════════════════════════════════════════════════════
# 6. YEARLY IC HEATMAP WITH MARKET REGIME
# ══════════════════════════════════════════════════════════════════════
print("\n[6/6] Yearly IC heatmap...")

# Compute yearly IC for each factor
factors_to_check = ['breadth', 'hidden_ratio', 'recognition_spread', 'visible_breadth',
                    'size_ln', 'bm', 'mom_12m', 'vol_12m']
heatmap_data = {}
for factor in factors_to_check:
    col = f'{factor}_z'
    ics = panel_m.dropna(subset=[col, 'ret']).groupby('ym_dt').apply(
        lambda g: stats.spearmanr(g[col], g['ret'])[0] if len(g) > 30 else np.nan,
        include_groups=False
    ).dropna()
    yearly = {}
    for y in range(2015, 2026):
        yr_ics = ics[ics.index.year == y]
        if len(yr_ics) >= 3:
            yearly[y] = yr_ics.mean()
    heatmap_data[factor] = yearly

hm_df = pd.DataFrame(heatmap_data)
hm_labels = ['Breadth', 'HiddenRatio', 'RecogSpread', 'VisibleBreadth',
             'Size', 'BM', 'Momentum', 'Volatility']

# Market returns for annotation
market_yearly_ret = {}
for y in range(2015, 2026):
    yr_ret = panel_m[panel_m['ym_dt'].dt.year == y].groupby('ym_dt')['ret'].mean().dropna()
    if len(yr_ret) >= 3:
        market_yearly_ret[y] = (1 + yr_ret).prod() - 1

# ══════════════════════════════════════════════════════════════════════
# 7. VISUALIZATION
# ══════════════════════════════════════════════════════════════════════
print("\n[7] Generating figures...")

fig = plt.figure(figsize=(22, 14))
fig.suptitle('Tier 1 Diagnostics — Hidden Pairs Factor Validation\n'
             'Fama-MacBeth | Factor Correlations | Double-Sort | OOS Split | Yearly IC Heatmap',
             fontsize=13, fontweight='bold')

# ── PANEL 1: Fama-MacBeth Coefficients ──────────────────────────────
ax = fig.add_subplot(3, 3, 1)
# Plot breadth coefficient across models M1-M5
fm_plot = fm_df[fm_df['model'].isin(['M1: Breadth only', 'M2: + Size', 
                                      'M3: + Size + BM', 'M4: + Size + BM + Mom',
                                      'M5: + Size + BM + Mom + Vol'])]
breadth_coefs = fm_plot['breadth_z_coef'].values
breadth_t = fm_plot['breadth_z_t'].values

x = np.arange(len(fm_plot))
colors = ['#2E86AB' if t > 1.65 else '#D4A373' if abs(t) < 1.65 else '#A23B72' 
          for t in breadth_t]
bars = ax.bar(x, breadth_coefs, color=colors, alpha=0.8, edgecolor='white')
for i, (c, t) in enumerate(zip(breadth_coefs, breadth_t)):
    sig = '***' if abs(t) > 2.58 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.65 else ''
    ax.text(i, c + 0.0001 if c >= 0 else c - 0.0002,
            f't={t:+.2f}{sig}', ha='center', fontsize=7, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([m.replace('M', '') for m in fm_plot['model']], 
                    fontsize=7, rotation=15)
ax.set_title('1. Fama-MacBeth: Breadth Coefficients\n(Monthly CS Regressions)',
             fontsize=10, fontweight='bold')
ax.set_ylabel('Avg Coefficient')
ax.grid(axis='y', alpha=0.3)

# ── PANEL 2: Factor Correlation Heatmap ─────────────────────────────
ax = fig.add_subplot(3, 3, 2)
im = ax.imshow(corr_data, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
ax.set_xticks(range(len(corr_labels)))
ax.set_yticks(range(len(corr_labels)))
ax.set_xticklabels(corr_labels, rotation=45, fontsize=6.5, ha='right')
ax.set_yticklabels(corr_labels, fontsize=6.5)
for i in range(len(corr_labels)):
    for j in range(len(corr_labels)):
        val = corr_data[i, j]
        color = 'white' if abs(val) > 0.5 else 'black'
        ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                fontsize=6, color=color, fontweight='bold')
ax.set_title('2. Factor Correlation Matrix\n(Avg Monthly Cross-Sectional)',
             fontsize=10, fontweight='bold')
plt.colorbar(im, ax=ax, shrink=0.8)

# ── PANEL 3: Double-Sort Bar Chart ──────────────────────────────────
ax = fig.add_subplot(3, 3, 3)
x_q = np.arange(5)
width = 0.35
for ft, label, color in [(1, 'Low Breadth', '#A23B72'), (2, 'High Breadth', '#2E86AB')]:
    vals = []
    for sq in range(1, 6):
        sub = ds_stats_df[(ds_stats_df['size_q'] == sq) & (ds_stats_df['factor_t'] == ft)]
        vals.append(sub['mean_ret'].values[0] if len(sub) else 0)
    offset = -width/2 if ft == 1 else width/2
    ax.bar(x_q + offset, vals, width, label=label, color=color, alpha=0.8, edgecolor='white')

ax.set_xticks(x_q)
ax.set_xticklabels([f'Q{q}\n(Small→Large)' for q in range(1, 6)], fontsize=7)
ax.set_title('3. Double-Sort: Size Quintile × Breadth\n(Monthly Equal-Weighted Return)',
             fontsize=10, fontweight='bold')
ax.set_ylabel('Mean Monthly Return')
ax.legend(fontsize=7)
ax.grid(axis='y', alpha=0.3)

# ── PANEL 4: OOS IC Comparison ──────────────────────────────────────
ax = fig.add_subplot(3, 3, 4)
factors_plot = ['breadth', 'recognition_spread', 'hidden_ratio']
factor_labels_plot = ['Breadth', 'RecogSpread', 'HiddenRatio']
x_oos = np.arange(len(factors_plot))
width_oos = 0.35

for i, (period, color) in enumerate([('In-Sample\n(2015-2020)', '#888888'), 
                                       ('Out-of-Sample\n(2021-2025)', '#2E86AB')]):
    vals = []
    for f in factors_plot:
        row = oos_df[(oos_df['factor'] == f) & (oos_df['period'] == period)]
        vals.append(row['mean_ic'].values[0] if len(row) else 0)
    offset = -width_oos/2 if i == 0 else width_oos/2
    bars = ax.bar(x_oos + offset, vals, width_oos, label=period, color=color,
                  alpha=0.8, edgecolor='white')
    for j, v in enumerate(vals):
        ax.text(x_oos[j] + offset, v + 0.002 if v >= 0 else v - 0.005,
                f'{v:+.3f}', ha='center', fontsize=7, fontweight='bold')

ax.axhline(0, color='black', linewidth=0.8)
ax.set_xticks(x_oos)
ax.set_xticklabels(factor_labels_plot, fontsize=8)
ax.set_title('4. Out-of-Sample IC Split\n(Monthly Rank IC, Mean)',
             fontsize=10, fontweight='bold')
ax.legend(fontsize=7)
ax.grid(axis='y', alpha=0.3)

# ── PANEL 5: Yearly IC Heatmap ──────────────────────────────────────
ax = fig.add_subplot(3, 3, 5)
years = list(range(2015, 2026))
hm_array = np.array([[hm_df.loc[y, f] if y in hm_df.index else np.nan 
                       for f in factors_to_check] for y in years])
# Replace NaN with 0 for display
hm_array_display = np.nan_to_num(hm_array, nan=0)

im_hm = ax.imshow(hm_array_display.T, cmap='RdBu_r', vmin=-0.2, vmax=0.2, aspect='auto')
ax.set_xticks(range(len(years)))
ax.set_yticks(range(len(hm_labels)))
ax.set_xticklabels([str(y) for y in years], fontsize=6.5, rotation=45)
ax.set_yticklabels(hm_labels, fontsize=6.5)
for i in range(len(hm_labels)):
    for j in range(len(years)):
        val = hm_array[j, i]
        if not np.isnan(val):
            color = 'white' if abs(val) > 0.12 else 'black'
            ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                    fontsize=5, color=color, fontweight='bold')
ax.set_title('5. Yearly Rank IC Heatmap\n(Red = Positive, Blue = Negative)',
             fontsize=10, fontweight='bold')
plt.colorbar(im_hm, ax=ax, shrink=0.8)

# ── PANEL 6: Market Regime Context ──────────────────────────────────
ax = fig.add_subplot(3, 3, 6)
market_vals = [market_yearly_ret.get(y, np.nan) for y in years]
valid_years = [y for y, v in zip(years, market_vals) if not np.isnan(v)]
valid_vals = [v for v in market_vals if not np.isnan(v)]

colors_mkt = ['#D64045' if v > 0 else '#2E86AB' for v in valid_vals]
bars = ax.bar(valid_years, [v*100 for v in valid_vals], color=colors_mkt, alpha=0.8, edgecolor='white')
for y, v in zip(valid_years, valid_vals):
    ax.text(y, v*100 + 0.5 if v >= 0 else v*100 - 1.5,
            f'{v*100:+.0f}%', ha='center', fontsize=7, fontweight='bold')
ax.axhline(0, color='black', linewidth=0.8)
ax.set_title('6. Market Regime Context\n(Average Stock Yearly Return)',
             fontsize=10, fontweight='bold')
ax.set_ylabel('Avg. Yearly Return (%)')
ax.grid(axis='y', alpha=0.3)

# ── PANEL 7: RecognitionSpread FM Coefficients ──────────────────────
ax = fig.add_subplot(3, 3, 7)
fm_r = fm_df[fm_df['model'].isin(['M6: RecogSpread only', 
                                    'M7: RecogSpread + Size + BM + Mom + Vol'])]
if len(fm_r):
    x_r = np.arange(len(fm_r))
    coefs_r = fm_r['recognition_spread_z_coef'].values
    t_r = fm_r['recognition_spread_z_t'].values
    colors_r = ['#2E86AB' if t > 1.65 else '#D4A373' for t in t_r]
    ax.bar(x_r, coefs_r, color=colors_r, alpha=0.8, edgecolor='white')
    for i, (c, t) in enumerate(zip(coefs_r, t_r)):
        sig = '***' if abs(t) > 2.58 else '**' if abs(t) > 1.96 else '*' if abs(t) > 1.65 else ''
        ax.text(i, c + 0.0001 if c >= 0 else c - 0.0002,
                f't={t:+.2f}{sig}', ha='center', fontsize=8, fontweight='bold')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.set_xticks(x_r)
    ax.set_xticklabels([m.replace('M', '') for m in fm_r['model']], fontsize=7)
ax.set_title("7. Fama-MacBeth: RecogSpread\n(Monthly CS Regressions)",
             fontsize=10, fontweight='bold')
ax.set_ylabel('Avg Coefficient')
ax.grid(axis='y', alpha=0.3)

# ── PANEL 8: OOS Detailed ──────────────────────────────────────────
ax = fig.add_subplot(3, 3, 8)
row_labels = []
row_vals = []
for _, row in oos_df.iterrows():
    lbl = f"{row['factor']} | {row['period'].replace(chr(10), ' ')}"
    row_labels.append(lbl)
    row_vals.append(row['ic_ir'])

y_plot = np.arange(len(row_labels))
colors_oos = ['#2E86AB' if v > 0 else '#A23B72' for v in row_vals]
bars = ax.barh(y_plot, row_vals, color=colors_oos, alpha=0.8, edgecolor='white', height=0.6)
for i, (lbl, v) in enumerate(zip(row_labels, row_vals)):
    ax.text(v + 0.01 if v >= 0 else v - 0.05, i, f'{v:+.2f}',
            fontsize=7, fontweight='bold', va='center')
ax.axvline(0, color='black', linewidth=0.8)
ax.set_yticks(y_plot)
ax.set_yticklabels(row_labels, fontsize=6.5)
ax.set_title('8. OOS IC_IR Comparison', fontsize=10, fontweight='bold')
ax.set_xlabel('IC_IR')
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

# ── PANEL 9: Summary / Verdict ──────────────────────────────────────
ax = fig.add_subplot(3, 3, 9)
ax.axis('off')

# Collect key diagnostics
# 1. Breadth FM coefficient with full controls
m5_row = fm_df[fm_df['model'] == 'M5: + Size + BM + Mom + Vol']
breadth_m5_t = m5_row['breadth_z_t'].values[0] if len(m5_row) else np.nan
breadth_m5_coef = m5_row['breadth_z_coef'].values[0] if len(m5_row) else np.nan

# 2. RecogSpread FM coefficient with full controls
m7_row = fm_df[fm_df['model'] == 'M7: RecogSpread + Size + BM + Mom + Vol']
recog_m7_t = m7_row['recognition_spread_z_t'].values[0] if len(m7_row) else np.nan

# 3. OOS Breadth IC
oos_breadth = oos_df[(oos_df['factor'] == 'breadth') & 
                      (oos_df['period'] == 'Out-of-Sample\n(2021-2025)')]
oos_breadth_ir = oos_breadth['ic_ir'].values[0] if len(oos_breadth) else np.nan
oos_breadth_t = oos_breadth['t_stat'].values[0] if len(oos_breadth) else np.nan

# 4. Breadth-Size correlation
breadth_size_corr = corr_rets.loc['breadth', 'size_ln']

# 5. Double-sort premium summary
size_premiums = {}
for sq in range(1, 6):
    hi = ds_stats_df[(ds_stats_df['size_q'] == sq) & (ds_stats_df['factor_t'] == 2)]
    lo = ds_stats_df[(ds_stats_df['size_q'] == sq) & (ds_stats_df['factor_t'] == 1)]
    if len(hi) and len(lo):
        size_premiums[sq] = hi['mean_ret'].values[0] - lo['mean_ret'].values[0]

pass_count = sum(1 for p in size_premiums.values() if p > 0)

verdict = f"""
 TIER 1 DIAGNOSTICS — VERDICT
 ═══════════════════════════════════════

 1. FAMA-MACBETH:
    Breadth (full controls)  t = {breadth_m5_t:+.2f}
    RecogSpread (full ctrls) t = {recog_m7_t:+.2f}
    → Breadth {'SURVIVES' if abs(breadth_m5_t) > 1.65 else 'DIES'}
      after controlling for Size/BM/Mom/Vol.
    → {'Signal is INDEPENDENT of known factors' if abs(breadth_m5_t) > 1.65 else 'Signal IS SUBSUMED by known factors'}

 2. CORRELATION:
    Breadth–Size correlation = {breadth_size_corr:.2f}
    → {'Breadth is NOT just a Size proxy' if abs(breadth_size_corr) < 0.5 else 'Breadth IS highly correlated with Size'}

 3. DOUBLE-SORT:
    Breadth premium positive in {pass_count}/5 Size quintiles.
    → {'Breadth works ACROSS size spectrum' if pass_count >= 4 else 'Breadth only works in SOME size tiers (' + str(pass_count) + '/5)'}

 4. OUT-OF-SAMPLE:
    Breadth OOS IC_IR = {oos_breadth_ir:+.2f}  (t = {oos_breadth_t:+.2f})
    → {'OOS signal EXISTS — deployable' if oos_breadth_ir > 0.1 else 'OOS signal WEAK or ABSENT — data limitation'}

 5. HEATMAP:
    See panel 5 for year-by-year IC patterns.
    Best years for Breadth: {' '.join(str(y) for y in years if not np.isnan(hm_df.loc[y, 'breadth'] if y in hm_df.index else np.nan) and hm_df.loc[y, 'breadth'] > 0.05)}

 ═══ BOTTOM LINE ═══
 {'PASS: Breadth/RecognitionSpread has independent predictive power beyond Size/BM/Mom/Vol.' if abs(breadth_m5_t) > 1.65 and oos_breadth_ir > 0.1 else 'CAUTION: Signal exists but needs multi-period data to confirm robustness. Current results are suggestive, not conclusive.'}
"""

ax.text(0.02, 0.98, verdict, fontsize=5.5, family='monospace',
        verticalalignment='top', horizontalalignment='left',
        bbox=dict(boxstyle='round', facecolor='#F5F5F5', alpha=0.9, edgecolor='#CCCCCC'))

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(f'{OUTDIR}/tier1_diagnostics.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved: {OUTDIR}/tier1_diagnostics.png")

# ── Save CSR ────────────────────────────────────────────────────────
# Save all results to a single CSV with multiple sections
all_rows = []

# FM results
for _, r in fm_df.iterrows():
    d = r.to_dict()
    d['section'] = 'fama_macbeth'
    all_rows.append(d)

# OOS results
for _, r in oos_df.iterrows():
    d = r.to_dict()
    d['section'] = 'oos_split'
    all_rows.append(d)

# Correlation (top triangle only)
for i in range(len(corr_labels)):
    for j in range(i+1, len(corr_labels)):
        all_rows.append({
            'section': 'correlation',
            'var1': corr_labels[i],
            'var2': corr_labels[j],
            'correlation': corr_data[i, j]
        })

# Yearly IC
for y in years:
    if y in hm_df.index:
        for f in factors_to_check:
            if y in hm_df.index and not np.isnan(hm_df.loc[y, f]):
                all_rows.append({
                    'section': 'yearly_ic',
                    'year': y,
                    'factor': f,
                    'ic': hm_df.loc[y, f]
                })

results_df = pd.DataFrame(all_rows)
results_df.to_csv(f'{OUTDIR}/tier1_diagnostics.csv', index=False, encoding='utf-8-sig')
print(f"  Saved: {OUTDIR}/tier1_diagnostics.csv")

# ══════════════════════════════════════════════════════════════════════
# FINAL
# ══════════════════════════════════════════════════════════════════════
elapsed = time.time() - START
print(f"\n{'='*70}")
print(f"TIER 1 DIAGNOSTICS COMPLETE — {elapsed:.1f}s")
print(f"{'='*70}")

# Print key summary
print(f"\n═══ KEY RESULTS ═══")
print(f"Breadth FM (full controls) t-stat: {breadth_m5_t:+.2f}")
print(f"RecogSpread FM (full controls) t-stat: {recog_m7_t:+.2f}")
print(f"Breadth–Size correlation: {breadth_size_corr:.3f}")
print(f"Breadth OOS IC_IR: {oos_breadth_ir:+.3f} (t={oos_breadth_t:+.2f})")
print(f"Breadth premium positive in {pass_count}/5 Size quintiles")
print(f"\nOutput: {OUTDIR}/tier1_diagnostics.png")
print(f"        {OUTDIR}/tier1_diagnostics.csv")
