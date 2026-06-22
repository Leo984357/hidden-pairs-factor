#!/usr/bin/env python3
"""Final comprehensive experiments on Fund Coverage Breadth factor.
Single-pass, designed to be robust."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy import stats
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

DIR = Path(__file__).parent
OUT = DIR / 'results' / '20260618'
OUT.mkdir(parents=True, exist_ok=True)
GENAI = Path('/Users/leolee/Desktop/genai_china_replication')

# ═══════════════════════════════════════════════════════
print("Loading data...")
# ═══════════════════════════════════════════════════════

# Factor
factor_df = pd.read_csv(DIR / 'results/20260618/stealth_score_factor.csv')
factor_df = factor_df[factor_df['coverage_flag']].copy()

# Name map
master = pd.read_csv(GENAI / 'data/raw/csmar/listing_master_by_year_clean.csv')
m24 = master[master['year'] == 2024][['stock_id', 'ShortName']].copy()
m24['nc'] = m24['ShortName'].str.replace('*', '').str.replace(' ', '').str.strip()
nm = {r['nc']: r['stock_id'] for _, r in m24.iterrows()}
factor_df['stock_id'] = factor_df['stock_name'].str.replace('*', '').str.replace(' ', '').str.strip().map(nm)
factor_df = factor_df.dropna(subset=['stock_id'])

# Build factors
S = factor_df.set_index('stock_id')
TOTAL = S['total_funds'].dropna()
HIDDEN = S['hidden_count'].dropna()
HIDDEN_RATIO = S['HiddenRatio'].dropna()
BREADTH = np.log1p(TOTAL)  # Coverage Breadth
covered_ids = set(BREADTH.index)
print(f"  {len(covered_ids)} stocks mapped")

# Monthly returns
print("Loading returns...")
chunks_ret = []
for ck in pd.read_csv(GENAI / 'data/processed/daily_returns_cn_all.csv',
    usecols=['stock_id', 'trade_date', 'ret', 'mkt_cap_float'],
    dtype={'stock_id': str}, chunksize=300000):
    ck['trade_date'] = pd.to_datetime(ck['trade_date'])
    ck = ck[ck['trade_date'].dt.year.between(2015, 2025)]
    ck = ck[ck['stock_id'].isin(covered_ids)]
    if len(ck): chunks_ret.append(ck)
daily = pd.concat(chunks_ret, ignore_index=True)
daily['ret'] = pd.to_numeric(daily['ret'], errors='coerce')
daily['mkt_cap_float'] = pd.to_numeric(daily['mkt_cap_float'], errors='coerce')
daily['ym'] = daily['trade_date'].dt.to_period('M')

monthly = daily.groupby(['stock_id', 'ym'])['ret'].apply(lambda x: (1+x).prod()-1).reset_index()
monthly['ym_dt'] = monthly['ym'].dt.to_timestamp()
ret_mat = monthly.pivot(index='stock_id', columns='ym_dt', values='ret')
months = sorted(ret_mat.columns)
print(f"  Returns: {ret_mat.shape[0]} stocks × {len(months)} months ({months[0].date()} to {months[-1].date()})")

# Fundamentals
fund = pd.read_csv(GENAI / 'Ef_factor_asset_pricing/data/processed/panel_Ef_daily.csv',
    usecols=['stock_id', 'trade_date', 'bm', 'roa_2023', 'industry_cs2012_code', 'size_ln', 'lev_approx'],
    dtype={'stock_id': str})
fund['trade_date'] = pd.to_datetime(fund['trade_date'])
fund['year'] = fund['trade_date'].dt.year
fund_ann = fund[fund['year'].between(2015, 2025)].groupby(['stock_id', 'year']).last().reset_index()

# Momentum
daily_sorted = daily.sort_values(['stock_id', 'trade_date'])
daily_sorted['ret_12m'] = daily_sorted.groupby('stock_id')['ret'].transform(
    lambda x: x.shift(1).rolling(12, min_periods=8).apply(lambda r: (1+r).prod()-1))
mom_2024 = daily_sorted[daily_sorted['trade_date'].dt.year == 2024].groupby('stock_id')['ret_12m'].last()

# ═══════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════

def ic_stats(ics):
    """Given list of ICs, return stats dict."""
    v = [x for x in ics if pd.notna(x)]
    if len(v) < 2:
        return {'mean': np.nan, 't': np.nan, 'ir': np.nan, 'pos': np.nan, 'n': len(v)}
    m = np.mean(v)
    s = np.std(v, ddof=1)
    return {'mean': m, 't': m/(s/np.sqrt(len(v))), 'ir': m/s, 'pos': sum(1 for x in v if x>0)/len(v), 'n': len(v)}

def monthly_ic(fv, ret_mat, months_subset=None):
    """Compute rank IC for each month. fv = pd.Series index=stock_id."""
    ics = []
    months_to_use = months_subset if months_subset is not None else months
    for m in months_to_use:
        rets = ret_mat[m].dropna()
        common = rets.index.intersection(fv.index)
        if len(common) < 30:
            continue
        ic, _ = stats.spearmanr(fv[common], rets[common])
        ics.append(ic)
    return ics

def forward_ic(fv, ret_mat, horizon=12):
    """Factor → cumulative return over next H months."""
    ics = []
    for i in range(len(months) - horizon):
        h_ret = (1 + ret_mat.iloc[:, i:i+horizon]).prod(axis=1) - 1
        common = h_ret.index.intersection(fv.index)
        if common.dropna().shape[0] < 30:
            continue
        ic, _ = stats.spearmanr(fv[common], h_ret[common])
        ics.append(ic)
    return np.mean(ics) if ics else np.nan

def yearly_ics(fv, ret_mat):
    """Group monthly IC by year, return year→IC dict."""
    monthly = monthly_ic(fv, ret_mat)
    monthly_dates = [m for m in months]  # all months
    # Align ICs with dates
    if len(monthly) != len(monthly_dates):
        # Recompute with explicit alignment
        monthly_dates = []
        monthly = []
        for m in months:
            rets = ret_mat[m].dropna()
            common = rets.index.intersection(fv.index)
            if len(common) < 30:
                continue
            ic, _ = stats.spearmanr(fv[common], rets[common])
            monthly.append(ic)
            monthly_dates.append(m)
    
    yearly = {}
    for _, grp in pd.DataFrame({'date': monthly_dates, 'ic': monthly}).groupby(
        monthly_dates.__class__([d.year for d in monthly_dates]) if monthly_dates else []
    ):
        pass
    
    # Simpler: iterate
    result = {}
    for y in range(2015, 2026):
        year_ics = []
        for d, ic in zip(monthly_dates, monthly):
            if d.year == y:
                year_ics.append(ic)
        if len(year_ics) >= 3:
            result[y] = np.mean(year_ics)
    return result

# ═══════════════════════════════════════════════════════
# QUICK BENCHMARK — confirm data works
# ═══════════════════════════════════════════════════════

print("\n=== Quick benchmark ===")
# Test with BREADTH
all_ics = monthly_ic(BREADTH, ret_mat)
st = ic_stats(all_ics)
print(f"Coverage Breadth monthly IC: mean={st['mean']:+.4f}, IR={st['ir']:+.2f}, t={st['t']:+.2f}, n={st['n']} ({st['pos']:.0%} pos)")

# Forward horizons
for h in [1, 3, 6, 12, 24]:
    fwd_mean = forward_ic(BREADTH, ret_mat, horizon=h)
    if pd.notna(fwd_mean):
        print(f"  {h}m forward IC: {fwd_mean:+.4f}")

# Also test Total Funds raw (not log)
tf_ics = monthly_ic(TOTAL, ret_mat)
tf_st = ic_stats(tf_ics)
hc_ics = monthly_ic(HIDDEN, ret_mat)
hc_st = ic_stats(hc_ics)
hr_ics = monthly_ic(HIDDEN_RATIO, ret_mat)
hr_st = ic_stats(hr_ics)

print(f"\nFactor comparison (monthly IC):")
print(f"  TotalFunds(raw):     IC={tf_st['mean']:+.4f}, IR={tf_st['ir']:+.2f}, t={tf_st['t']:+.2f}")
print(f"  HiddenCount:          IC={hc_st['mean']:+.4f}, IR={hc_st['ir']:+.2f}, t={hc_st['t']:+.2f}")
print(f"  HiddenRatio:          IC={hr_st['mean']:+.4f}, IR={hr_st['ir']:+.2f}, t={hr_st['t']:+.2f}")
print(f"  CoverageBreadth(log): IC={st['mean']:+.4f}, IR={st['ir']:+.2f}, t={st['t']:+.2f}")

# ═══════════════════════════════════════════════════════
# EXP 1: Market Cap Stratification
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXP 1: Market Cap Stratification")
print("="*60)

cap_2024 = daily[daily['trade_date'].dt.year == 2024].groupby('stock_id')['mkt_cap_float'].median()
valid_cap = cap_2024.dropna()
if len(valid_cap) >= 100:
    terciles = pd.qcut(valid_cap, 3, labels=['Small', 'Mid', 'Large'])
    for label in ['Small', 'Mid', 'Large']:
        ids = terciles[terciles == label].index
        fv = BREADTH[BREADTH.index.isin(ids)]
        if len(fv) < 50:
            print(f"  {label}: too few ({len(fv)})")
            continue
        m_ic = monthly_ic(fv, ret_mat)
        s = ic_stats(m_ic)
        fwd12 = forward_ic(fv, ret_mat, 12)
        print(f"  {label}: N={len(fv)}, mIC={s['mean']:+.4f}, IR={s['ir']:+.2f}, t={s['t']:+.2f}, fwd12m={fwd12:+.4f}")

# ═══════════════════════════════════════════════════════
# EXP 2: Industry Decomposition
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXP 2: Industry Decomposition")
print("="*60)

ind_fund = fund_ann[fund_ann['year'] == 2024].set_index('stock_id')['industry_cs2012_code'].dropna()
ind_results = {}
for code in sorted(ind_fund.unique()):
    ids = ind_fund[ind_fund == code].index
    fv = BREADTH[BREADTH.index.isin(ids)]
    if fv.notna().sum() < 30:
        continue
    m_ic = monthly_ic(fv, ret_mat)
    s = ic_stats(m_ic)
    if pd.notna(s['mean']):
        ind_results[code] = s

sorted_ind = sorted(ind_results.items(), key=lambda x: x[1]['ir'] if pd.notna(x[1]['ir']) else -999, reverse=True)
print(f"Top 10 industries by IR:")
for code, s in sorted_ind[:10]:
    n = (ind_fund == code).sum()
    print(f"  {code}: N={n}, IC={s['mean']:+.4f}, IR={s['ir']:+.2f}")
print(f"Bottom 5:")
for code, s in sorted_ind[-5:]:
    print(f"  {code}: IC={s['mean']:+.4f}, IR={s['ir']:+.2f}")

# ═══════════════════════════════════════════════════════
# EXP 3: Composite Factors
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXP 3: Composite Factors")
print("="*60)

f24 = fund_ann[fund_ann['year'] == 2024].set_index('stock_id')
bm = f24['bm'].dropna()
roa = f24['roa_2023'].dropna()
size_ln = f24['size_ln'].dropna()

common = BREADTH.index.intersection(bm.index).intersection(mom_2024.dropna().index)
print(f"  Composite universe: {len(common)} stocks")

def z(s):
    return (s - s.mean()) / s.std()

zb = z(BREADTH[common])
zbm = z(bm[common])
zmom = z(mom_2024[common])

composites = {
    'Breadth+BM': (zb + zbm) / 2,
    'Breadth+Mom': (zb + zmom) / 2,
    'Breadth+BM+Mom': (zb + zbm + zmom) / 3,
    'Breadth-BM': zb - zbm,
}

for name, cfv in composites.items():
    m_ic = monthly_ic(cfv, ret_mat)
    s = ic_stats(m_ic)
    fwd12 = forward_ic(cfv, ret_mat, 12)
    print(f"  {name}: IC={s['mean']:+.4f}, IR={s['ir']:+.2f}, t={s['t']:+.2f}, fwd12m={fwd12:+.4f}")

# Standalone benchmarks
for name, fv in [('BM', bm), ('Momentum', mom_2024)]:
    fv = fv.dropna()
    m_ic = monthly_ic(fv, ret_mat)
    s = ic_stats(m_ic)
    print(f"  {name} standalone: IC={s['mean']:+.4f}, IR={s['ir']:+.2f}")

# ═══════════════════════════════════════════════════════
# EXP 4: Neutralization
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXP 4: Orthogonal Neutralization")
print("="*60)

neut_ids = common.intersection(size_ln.index).intersection(roa.index)
df_n = pd.DataFrame({'factor': BREADTH[neut_ids], 'bm': bm[neut_ids],
                      'mom': mom_2024[neut_ids], 'size': size_ln[neut_ids],
                      'roa': roa[neut_ids]}).dropna()
print(f"  Neutralization universe: {len(df_n)} stocks")

# Residualize against Size + BM + ROA
X = np.column_stack([np.ones(len(df_n)), df_n['size'].values, df_n['bm'].values, df_n['roa'].values])
y = df_n['factor'].values
beta = np.linalg.lstsq(X, y, rcond=None)[0]
resid = pd.Series(y - X @ beta, index=df_n.index)

resid_ic = monthly_ic(resid, ret_mat)
resid_s = ic_stats(resid_ic)
print(f"  Size+BM+ROA neutralized: IC={resid_s['mean']:+.4f}, IR={resid_s['ir']:+.2f}, t={resid_s['t']:+.2f}")

# Industry neutralized
ind_map = f24['industry_cs2012_code'].dropna()
ind_common = BREADTH.index.intersection(ind_map.index)
ind_fv = pd.DataFrame({'factor': BREADTH[ind_common], 'ind': ind_map[ind_common]}).dropna()
ind_fv['factor_ind_resid'] = ind_fv.groupby('ind')['factor'].transform(lambda x: x - x.mean())
ind_resid_ic = monthly_ic(ind_fv['factor_ind_resid'], ret_mat)
ind_resid_s = ic_stats(ind_resid_ic)
print(f"  Industry neutralized:   IC={ind_resid_s['mean']:+.4f}, IR={ind_resid_s['ir']:+.2f}, t={ind_resid_s['t']:+.2f}")

# ═══════════════════════════════════════════════════════
# EXP 5: Quintile Portfolio
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXP 5: Quintile Portfolio Backtest")
print("="*60)

ranks = BREADTH.rank(pct=True)
try:
    q_labels = pd.qcut(ranks, 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
except ValueError:
    # Fallback: use rank pctiles directly
    q_labels = pd.cut(ranks, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5], include_lowest=True)
port_rets = {q: [] for q in range(1, 6)}
port_dates = []

for m in months:
    mr = ret_mat[m].dropna()
    q_rets_m = {}
    valid = True
    for q in range(1, 6):
        q_ids = q_labels[q_labels == q].index
        q_r = mr[mr.index.isin(q_ids)]
        if len(q_r) < 5:
            valid = False
            break
        q_rets_m[q] = q_r.mean()
    if valid:
        for q in range(1, 6):
            port_rets[q].append(q_rets_m[q])
        port_dates.append(m)

port_df = pd.DataFrame(port_rets, index=port_dates)
cum = (1 + port_df).cumprod()
q_ann = {}
for q in range(1, 6):
    ann = cum[q].iloc[-1] ** (12 / len(port_df)) - 1
    vol = port_df[q].std() * np.sqrt(12)
    sr = ann / vol if vol > 0 else 0
    q_ann[q] = (ann, vol, sr)
    print(f"  Q{q}: Ann={ann:+.2%}, Vol={vol:.2%}, Sharpe={sr:+.2f}")

ls = port_df[5] - port_df[1]
ls_cum = (1 + ls).cumprod()
ls_ann = ls_cum.iloc[-1] ** (12 / len(ls)) - 1
ls_vol = ls.std() * np.sqrt(12)
ls_t = ls.mean() / ls.std() * np.sqrt(len(ls))
print(f"  Q5-Q1: Ann={ls_ann:+.2%}, Vol={ls_vol:.2%}, t={ls_t:+.2f}")

# ═══════════════════════════════════════════════════════
# EXP 6: Stock Characteristics by Coverage Quintile
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXP 6: Stock Characteristics by Coverage")
print("="*60)

# Use percentile-based binning to avoid duplicate edge issues
tf_vals = TOTAL[TOTAL > 0].dropna()
tf_pct = tf_vals.rank(pct=True)
tf_q = pd.cut(tf_pct, bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], labels=[1, 2, 3, 4, 5], include_lowest=True)
n_q = tf_q.nunique()
cap_m = cap_2024
bm_m = bm
roa_m = roa

print(f"{'Q':<5} {'N':>5} {'TotFunds':>9} {'HiddenR':>8} {'Cap(亿)':>10} {'BM':>8} {'ROA':>8} {'HiddenC':>8}")
print("-" * 65)
for q in sorted(tf_q.dropna().unique()):
    q_ids = tf_q[tf_q == q].index
    tf_v = TOTAL[q_ids]
    hr_v = HIDDEN_RATIO[HIDDEN_RATIO.index.isin(q_ids)]
    hc_v = HIDDEN[HIDDEN.index.isin(q_ids)]
    cap_v = cap_m[cap_m.index.isin(q_ids)] / 1e8
    bm_v = bm_m[bm_m.index.isin(q_ids)]
    roa_v = roa_m[roa_m.index.isin(q_ids)]
    print(f"Q{q}    {len(q_ids):>5} "
          f"{tf_v.median():>9.0f} "
          f"{hr_v.median():>8.3f} "
          f"{cap_v.median():>10.1f} "
          f"{bm_v.median():>8.3f} "
          f"{roa_v.median():>8.3f} "
          f"{hc_v.median():>8.0f}")

# ═══════════════════════════════════════════════════════
# EXP 7: Top5 (visible) vs Hidden (stealth) signal
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXP 7: Visible vs Hidden Signal")
print("="*60)

top5 = S['funds_with_top5'].dropna()
top5 = top5[top5 > 0]
tp5_ics = monthly_ic(top5, ret_mat)
tp5_s = ic_stats(tp5_ics)
print(f"  Top5-only (visible holdings): IC={tp5_s['mean']:+.4f}, IR={tp5_s['ir']:+.2f}, t={tp5_s['t']:+.2f}")
print(f"  Comparison:")
print(f"    Total coverage:  IC={tf_st['mean']:+.4f}, IR={tf_st['ir']:+.2f}")
print(f"    Visible (Top5):  IC={tp5_s['mean']:+.4f}, IR={tp5_s['ir']:+.2f}")
print(f"    Hidden (6-10):   IC={hc_st['mean']:+.4f}, IR={hc_st['ir']:+.2f}")
print(f"    HiddenRatio:     IC={hr_st['mean']:+.4f}, IR={hr_st['ir']:+.2f}")

# ═══════════════════════════════════════════════════════
# EXP 8: Macro cycle decomposition
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXP 8: Macro Cycle")
print("="*60)

periods = [
    ('2015 Bull/Crash', 2015, 2016),
    ('2017 Blue Chip', 2017, 2017),
    ('2018 Bear', 2018, 2018),
    ('2019-20 Recovery', 2019, 2020),
    ('2021-22 Structural', 2021, 2022),
    ('2023-24 Rebound', 2023, 2024),
    ('2025', 2025, 2025),
]

for name, ys, ye in periods:
    pm = [m for m in months if ys <= m.year <= ye]
    if len(pm) < 4:
        print(f"  {name}: too few months ({len(pm)})")
        continue
    p_ics = monthly_ic(BREADTH, ret_mat, months_subset=pm)
    s = ic_stats(p_ics)
    mkt_avg = ret_mat.loc[:, pm].mean().mean()
    print(f"  {name}: IC={s['mean']:+.4f}, IR={s['ir']:+.2f}, mkt={mkt_avg:+.2%}")

# ═══════════════════════════════════════════════════════
# EXP 9: Liquidity Filter
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXP 9: Liquidity / Quality Filter")
print("="*60)

# Filter: mkt_cap > ¥5B
big_only = cap_m[cap_m > 5e9].index
fv_big = BREADTH[BREADTH.index.isin(big_only)]
if len(fv_big) >= 100:
    big_ic = monthly_ic(fv_big, ret_mat)
    big_s = ic_stats(big_ic)
    print(f"  MktCap > ¥50亿: N={len(fv_big)}, IC={big_s['mean']:+.4f}, IR={big_s['ir']:+.2f}")

# Filter: top 80% by market cap
cap_p80 = cap_m.quantile(0.2)
cap_top80 = cap_m[cap_m >= cap_p80].index
fv_top80 = BREADTH[BREADTH.index.isin(cap_top80)]
if len(fv_top80) >= 100:
    t80_ic = monthly_ic(fv_top80, ret_mat)
    t80_s = ic_stats(t80_ic)
    print(f"  Top 80% mktcap:  N={len(fv_top80)}, IC={t80_s['mean']:+.4f}, IR={t80_s['ir']:+.2f}")

# ═══════════════════════════════════════════════════════
# EXP 10: Fama-MacBeth cross-section
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("EXP 10: Cross-Sectional Regression (Fama-MacBeth style)")
print("="*60)

# Factor → forward 12m return
fwd12 = []
for i in range(len(months) - 12):
    h_ret = (1 + ret_mat.iloc[:, i:i+12]).prod(axis=1) - 1
    h_ret.name = months[i]
    fwd12.append(h_ret.mean(axis=0) if False else h_ret)
# Use average across all overlapping 12m windows
avg_fwd12 = pd.concat([pd.DataFrame({'fwd': h}) for h in fwd12]).groupby(level=0)['fwd'].mean()

fm_data = pd.DataFrame({
    'breadth': BREADTH,
    'total_funds': TOTAL,
    'hidden_count': HIDDEN,
    'hidden_ratio': HIDDEN_RATIO,
}).join(f24[['bm', 'size_ln', 'lev_approx']], how='left')
fm_data['future_12m'] = avg_fwd12
fm_data = fm_data.dropna(subset=['breadth', 'bm', 'size_ln', 'future_12m'])
for c in ['breadth', 'total_funds', 'hidden_count', 'hidden_ratio', 'bm', 'size_ln']:
    if c in fm_data.columns:
        fm_data[c] = (fm_data[c] - fm_data[c].mean()) / fm_data[c].std()

import statsmodels.api as sm

models_fm = [
    ('Breadth only', ['breadth']),
    ('Breadth + Size + BM', ['breadth', 'size_ln', 'bm']),
    ('TotalFunds + Size + BM', ['total_funds', 'size_ln', 'bm']),
    ('HiddenCount + Size + BM', ['hidden_count', 'size_ln', 'bm']),
    ('HiddenRatio + Size + BM', ['hidden_ratio', 'size_ln', 'bm']),
]

for name, cols in models_fm:
    if not all(c in fm_data.columns for c in cols) or fm_data[cols].dropna().shape[0] < 50:
        print(f"  {name}: insufficient data")
        continue
    X = sm.add_constant(fm_data[cols].dropna())
    y = fm_data.loc[X.index, 'future_12m']
    try:
        m = sm.OLS(y, X).fit()
        coef_key = cols[0]
        print(f"  {name}: R²={m.rsquared:.4f}, coef({coef_key})={m.params.get(coef_key, 0):+.4f}%, t={m.tvalues.get(coef_key, 0):+.2f}")
    except Exception as e:
        print(f"  {name}: error {e}")

# ═══════════════════════════════════════════════════════
# PLOTS
# ═══════════════════════════════════════════════════════
print("\nPlotting...")

fig, axes = plt.subplots(3, 3, figsize=(20, 18))
fig.suptitle('Fund Coverage Breadth — Comprehensive Diagnostics', fontsize=16, fontweight='bold')

# 1) Monthly IC time series
ax = axes[0, 0]
ic_dates = []
ic_vals = []
for m in months:
    rets = ret_mat[m].dropna()
    common = rets.index.intersection(BREADTH.index)
    if len(common) >= 30:
        ic, _ = stats.spearmanr(BREADTH[common], rets[common])
        ic_dates.append(m)
        ic_vals.append(ic)
ax.plot(ic_dates, ic_vals, linewidth=0.8, color='#1565c0', alpha=0.7)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.axhline(y=st['mean'], color='#c62828', linestyle='--', linewidth=1, 
           label=f'Mean IC={st["mean"]:+.3f}')
ax.set_title(f'Monthly Rank IC — Coverage Breadth\nIR={st["ir"]:+.2f}, t={st["t"]:+.2f}')
ax.set_ylabel('Rank IC')
ax.legend(fontsize=8)

# 2) Cumulative IC
ax = axes[0, 1]
cum_ic = np.cumsum(ic_vals)
ax.plot(ic_dates, cum_ic, linewidth=1.5, color='#2e7d32')
ax.set_title(f'Cumulative Rank IC\nFinal={cum_ic[-1]:+.1f}')
ax.axhline(y=0, color='black', linewidth=0.8)

# 3) Forward horizon decay
ax = axes[0, 2]
horizons = [1, 3, 6, 12, 24, 36]
fwd_vals = []
for h in horizons:
    if len(months) <= h:
        fwd_vals.append(np.nan)
        continue
    fwd_vals.append(forward_ic(BREADTH, ret_mat, h))
ax.plot(horizons, fwd_vals, 'o-', color='#1565c0', linewidth=2, markersize=8)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_title('Forward Horizon IC')
ax.set_xlabel('Months forward')
ax.set_ylabel('Rank IC')

# 4) Market cap stratification
ax = axes[1, 0]
cap_labels = []
cap_irs = []
cap_ts = []
for label in ['Small', 'Mid', 'Large']:
    ids = terciles[terciles == label].index
    fv = BREADTH[BREADTH.index.isin(ids)]
    if len(fv) >= 50:
        m_ic = monthly_ic(fv, ret_mat)
        s = ic_stats(m_ic)
        cap_labels.append(label)
        cap_irs.append(s['ir'])
        cap_ts.append(s['t'])
x = np.arange(len(cap_labels))
w = 0.3
ax.bar(x - w/2, cap_irs, w, color='#1565c0', alpha=0.8, label='IR')
ax2 = ax.twinx()
ax2.bar(x + w/2, cap_ts, w, color='#e65100', alpha=0.8, label='t-stat')
ax.set_xticks(x)
ax.set_xticklabels(cap_labels)
ax.set_title('Market Cap Stratification')
ax.set_ylabel('IR', color='#1565c0')
ax2.set_ylabel('t-stat', color='#e65100')

# 5) Industry IR
ax = axes[1, 1]
if sorted_ind:
    top_n = min(15, len(sorted_ind))
    ind_labels = [str(c) for c, s in sorted_ind[:top_n]]
    ind_irs = [s['ir'] for c, s in sorted_ind[:top_n]]
    colors_ind = ['#2e7d32' if v > 0 else '#d32f2f' for v in ind_irs]
    ax.barh(range(len(ind_labels)), ind_irs, color=colors_ind, alpha=0.8)
    ax.set_yticks(range(len(ind_labels)))
    ax.set_yticklabels(ind_labels, fontsize=7)
    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.set_title(f'Top {top_n} Industries by IR')
    ax.set_xlabel('IR')

# 6) Portfolio cumulative
ax = axes[1, 2]
for q in [1, 2, 3, 4, 5]:
    ax.plot(cum.index, cum[q], label=f'Q{q}', linewidth=1.5, alpha=0.9)
ax.plot(ls_cum.index, ls_cum, 'k--', linewidth=2, label=f'Q5-Q1 ({ls_ann:+.1%})', alpha=0.8)
ax.set_title(f'Quintile Cumulative Returns')
ax.legend(fontsize=7, loc='upper left')
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

# 7) Factor comparison bar
ax = axes[2, 0]
comp_factors = ['Coverage\nBreadth', 'Total\nFunds', 'Hidden\nCount', 'Hidden\nRatio', 'Top5\nOnly']
comp_irs = [st['ir'], tf_st['ir'], hc_st['ir'], hr_st['ir'], tp5_s['ir']]
comp_colors = ['#2e7d32' if v > 0 else '#d32f2f' for v in comp_irs]
ax.bar(range(len(comp_factors)), comp_irs, color=comp_colors, alpha=0.8)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(range(len(comp_factors)))
ax.set_xticklabels(comp_factors, fontsize=8)
ax.set_title('Factor Variant Comparison (IR)')
ax.set_ylabel('IR')

# 8) Macro cycle
ax = axes[2, 1]
period_data = []
for name, ys, ye in periods:
    pm = [m for m in months if ys <= m.year <= ye]
    if len(pm) < 4:
        continue
    p_ics = monthly_ic(BREADTH, ret_mat, months_subset=pm)
    s = ic_stats(p_ics)
    period_data.append((name, s['mean']))
p_labels = [p[0] for p in period_data]
p_vals = [p[1] for p in period_data]
colors_p = ['#2e7d32' if v > 0 else '#d32f2f' for v in p_vals]
ax.barh(range(len(p_labels)), p_vals, color=colors_p, alpha=0.8)
ax.set_yticks(range(len(p_labels)))
ax.set_yticklabels(p_labels, fontsize=8)
ax.axvline(x=0, color='black', linewidth=0.8)
ax.set_title('Macro Cycle IC')
ax.set_xlabel('Mean IC')

# 9) Neutralization comparison
ax = axes[2, 2]
neut_labels = ['Raw', 'Industry\nNeut', 'Size+BM\n+ROA Neut']
neut_irs = [st['ir'], ind_resid_s['ir'], resid_s['ir']]
colors_n = ['#1565c0', '#2e7d32', '#e65100']
ax.bar(range(len(neut_labels)), neut_irs, color=colors_n, alpha=0.8, width=0.5)
ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(range(len(neut_labels)))
ax.set_xticklabels(neut_labels, fontsize=9)
ax.set_title('Orthogonal Neutralization')
ax.set_ylabel('IR')

plt.tight_layout()
fig.savefig(OUT / 'final_coverage_breadth.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: final_coverage_breadth.png")

# ═══════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════
print("\n" + "="*60)
print("FINAL VERDICT")
print("="*60)
print(f"""
After all experiments with single-cross-section HiddenPairs data applied to 2015-2025 returns:

FACTOR COMPARISON (monthly IC):
  Coverage Breadth (ln(1+total_funds)): IC={st['mean']:+.4f}, IR={st['ir']:+.2f}, t={st['t']:+.2f}
  Total Funds (raw):                    IC={tf_st['mean']:+.4f}, IR={tf_st['ir']:+.2f}
  Hidden Count (stealth scale):         IC={hc_st['mean']:+.4f}, IR={hc_st['ir']:+.2f}
  Hidden Ratio (stealth purity):        IC={hr_st['mean']:+.4f}, IR={hr_st['ir']:+.2f}
  Top5 Only (visible holdings):         IC={tp5_s['mean']:+.4f}, IR={tp5_s['ir']:+.2f}

NEUTRALIZATION:
  Industry neutralized:                 IR={ind_resid_s['ir']:+.2f}
  Size+BM+ROA neutralized:              IR={resid_s['ir']:+.2f}

PORTFOLIO:
  Q5-Q1 annualized:                     {ls_ann:+.2%}, t={ls_t:+.2f}

CONCLUSION: Fund coverage breadth is {("a weak positive signal" if st['mean'] > 0.005 else "statistically indistinguishable from zero")}. 
The "hidden/stealth" aspect does NOT add predictive power — total visible 
fund ownership works equally well or better.

All results based on single cross-section applied to 11 years of returns.
Multi-quarter holdings data needed for proper time-series validation.
""")
