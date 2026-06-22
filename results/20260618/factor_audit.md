# StealthScore Factor Audit

**Generated:** 2026-06-18T14:27:36.572586

## Factor Definition

```
StealthScore = ln(1 + hidden_count) * HiddenRatio
              = ln(1 + hidden_count) * (hidden_count / funds_with_top5_data)
```

- **Breadth**: ln(1 + hidden_count) captures stealth scale (diminishing returns)
- **Purity**: HiddenRatio captures what fraction of holders are stealth (0~1)
- **Product**: high score means BOTH many funds AND most are stealth

## Comparison: HiddenRatio vs StealthScore

HiddenRatio (legacy) collapses because it ignores scale — a stock with 3/3 hidden
and a stock with 45/50 hidden both get ~1.0. StealthScore multiplies breadth into
the signal, creating meaningful differentiation.

## 1. Data Sources

- Stock data: `全部A股.xlsx` — 5427 rows
- Fund data: `全部基金(主代码).xlsx` — 13090 rows

## 2. Coverage

- Stocks with fund holding data: 3547 / 5427 (65.4%)
- Stocks with fund top5 data: 3140

## 3. Distribution

| Stat | Value |
|------|-------|
| N | 3140 |
| Mean | 1.0228 |
| Median | 0.9954 |
| Std | 0.5738 |
| Skew | 0.3269 |
| Kurtosis | -0.5004 |
| Min | 0.0000 |
| P5 | 0.0000 |
| P25 | 0.6931 |
| P75 | 1.3863 |
| P95 | 2.0723 |
| Max | 2.3979 |

## 4. Factor Coverage Assessment

✅ **PASS**: Coverage sufficient (3140 stocks).
✅ **PASS**: Distribution roughly symmetric (skew=0.327).
✅ **PASS**: Continuous distribution — no boundary collapse (StealthScore = breadth × purity).

## 5. Next Steps

- [ ] IC analysis (needs price data)
- [ ] Quintile backtest (needs price data)
- [ ] Correlation with existing factors
- [ ] Fama-MacBeth (needs factor data + returns)

## 6. Output Files

- `distribution.png`
- `factor_audit.md`
- `hidden_ratio_factor.csv`
- `ic_results.csv`
- `ic_time_series.png`
- `quintile_cumulative.png`
- `top_bottom_30.png`