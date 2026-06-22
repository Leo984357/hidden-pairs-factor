# hidden-pairs-factor

股票视角隐形重仓对因子挖掘项目 · v0.2.0

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## 背景

基金季报仅披露前五大重仓股，但东方财富/同花顺数据可查到每只股票的前十大持股基金列表。
当一只股票出现在某基金的前十大持股中，却**不在**该基金的前五大重仓股里，
这只基金-股票对即为 **"隐形重仓对（Hidden Pair）"**。

这些 hidden pairs 反映了机构持仓的"水面下共识"——多家基金独立持有某股票，
但均未将其纳入公开披露阈值内的重仓位，构成潜在信息差选股信号。

---

## 因子演进

### v1.0 — HiddenRatio（弃用）

```
HiddenRatio = hidden_count / total_funds
```

**缺陷**：52% 股票 HiddenRatio = 1.0，pd.qcut 无法分组，分布严重左偏（skew = -1.21）。
根本原因：比率忽略规模，1/1 与 45/50 的 HiddenRatio 均接近 1.0。

### v2.0 — StealthScore（当前版本）

```
StealthScore = ln(1 + hidden_count) × HiddenRatio
```

| 组件 | 含义 |
|------|------|
| `ln(1 + hidden_count)` | 广度（Breadth）：隐形持有规模，对数递减 |
| `HiddenRatio` | 纯度（Purity）：隐形比例 0~1 |
| 乘积 | 两个条件同时满足才得高分 |

**分布改善**：偏度从 -1.21 → +0.33，边界堆积从 52% → 0.9%，可直接横截面排序。

---

## 数据来源

| 文件 | 说明 | 规模 |
|------|------|------|
| `全部A股.xlsx` | 东方财富/同花顺导出，含「前十大持股基金名称」列 | 5427 支股票 |
| `全部基金(主代码).xlsx` | 基金季报数据，含 Top1–Top5 重仓股列 | 13090 只基金 |
| CSMAR `daily_returns_cn_all.csv` | 日度收益率（1991-2025），IC 回测用 | 621MB，借自 genai_china_replication |

> ⚠️ **数据局限**：持仓数据为单一截面季报（2025年Q1附近），无多期时序。
> IC 回测将同一截面因子映射到11年（2015-2025）的时序收益率上，存在 look-back bias，
> 结论仅供探索性参考，不构成因子有效性的严格证明。

---

## 实证结果

### Phase 1：因子分布审计

| 指标 | HiddenRatio (v1) | StealthScore (v2) |
|------|------------------|-------------------|
| N | 3140 | 3140 |
| 偏度 | -1.21 | +0.33 |
| 峰度 | +0.43 | -0.50 |
| 边界堆积比例 | 52.0% | 0.9% |
| 均值 | 0.90 | 1.02 |
| 标准差 | 0.30 | 0.57 |
| 范围 | [0, 1] | [0, 2.40] |

✅ StealthScore 分布连续、无堆积、可分层排序。

### Phase 2：IC 分析（月频，2015-2025，CSMAR 真实收益率）

| 因子 | RankIC 均值 | IC_IR | t 统计量 | 结论 |
|------|------------|-------|---------|------|
| HiddenRatio | -0.019 | **-0.299** | -3.42\*\*\* | ❌ 显著负信号 |
| StealthScore | +0.001 | +0.012 | +0.04 | ❌ 无信号 |
| HiddenCount | +0.014 | +0.103 | +0.34 | ❌ 无显著信号 |
| **CoverageBreadth** = ln(1+total_funds) | +0.059 | **+0.406** | +1.35 | ⚠️ 弱正信号 |

> CoverageBreadth 是 StealthScore 的分母成分，单独使用反而优于复合因子。

### Phase 3：Tier1 稳健性检验

**预测期限（HiddenRatio，threshold=2）**：

| 期限 | IC 均值 | IC_IR | t |
|------|---------|-------|---|
| 1M | -0.015 | -0.30 | -3.43\*\*\* |
| 3M | -0.026 | -0.57 | -6.50\*\*\* |
| 6M | -0.035 | -0.72 | -8.07\*\*\* |
| 12M | -0.049 | -1.08 | -11.87\*\*\* |

**结论**：随期限延长负信号持续放大，非噪音——HiddenRatio 是反向因子或因子反转。

**安慰剂检验（Permutation Test）**：真实 IR < 100次随机 shuffle 的任何一次（P=1.00），
排除信息优势的可能性。

### Phase 4：Tier2 异质性分析（CoverageBreadth）

| 子群 | IC_IR | t | 正IC月比例 |
|------|-------|---|---------|
| 全样本 | +0.406 | +1.35 | 54.5% |
| 低波动子集 | +0.450 | +1.49 | 72.7% |
| **高波动子集** | **+0.936** | **+3.11\*\*** | **90.9%** |
| 牛市 | +0.157 | +1.34 | 55.6% |
| 熊市 | +0.458 | +1.59 | 75.0% |
| 横盘 | +0.287 | +1.99\* | 52.1% |

**高波动子集是最强条件**：IR=0.94，t=3.11，正IC月比例 90.9%。
信号集中在信息不对称更高的高波动股票中，符合"隐形共识"的经济逻辑。

### Phase 5：因子合成与Tier3

CoverageBreadth 与 BM 的合成因子（Breadth+BM）IR=-0.38，合成后反而变差。
BM（账面市值比）在本样本中本身为负向（IR=-0.96，t=-3.17），合成无法改善信号。

---

## 结论与局限

**核心发现**：
- StealthScore 在单截面场景下无预测力（IR≈0）
- HiddenRatio 是反向因子（IR=-0.30），统计显著但方向为负，经济解释存疑
- CoverageBreadth = ln(1+持股基金数) 有弱正信号，高波动子集中显著（IR=0.94）
- 隐形持仓比例（HiddenRatio）本身不含正向信息，关键是覆盖广度

**根本局限**：
1. **单截面数据**：仅一期持仓数据，用于11年时序回测存在 look-back bias
2. **需要 CSMAR FUN_PortfolioStock**：多期基金季报数据才能消除此问题
3. **因果识别**：无政策冲击工具变量，DID 不可行

**下一步**（待数据）：
- [ ] 接入 CSMAR FUN_PortfolioStock 多期季报，构建时序面板
- [ ] 时序 IC 序列的稳健性验证
- [ ] 与 QMT 因子池相关性分析（排除市值/动量/波动率 proxy）
- [ ] 针对高波动子集做专项回测

---

## 项目结构

```
hidden-pairs-factor/
├── src/hidden_pairs/
│   ├── __init__.py
│   ├── config.py          # 列名映射、常量
│   ├── data_loader.py     # Excel 解析（自动检测表头行）
│   ├── factor_builder.py  # 核心：hidden pairs 构建 + StealthScore
│   ├── visualizer.py      # matplotlib 图表（中文字体自动检测）
│   └── cli.py            # argparse CLI（--active-only 等参数）
├── validate_factor.py     # Phase 1 审计脚本
├── ic_backtest.py         # Phase 2 IC 分析（接入 CSMAR）
├── explore_factor.py      # 因子变体探索（11个变体）
├── robustness_check.py    # 早期稳健性检查（已被 tier1_2_combined 取代）
├── tier1_2_combined.py    # Tier1+2 合并，干净月频面板
├── tier1_robustness.py    # Tier1 独立稳健性脚本
├── tier2_3_remaining.py   # Tier2 剩余+Tier3（信息不对称、宏观周期、因子合成）
├── final_experiments.py   # CoverageBreadth 专项诊断（9张子图）
├── tests/
│   └── test_factor_builder.py
├── examples/
│   └── run_factor.py
├── results/
│   ├── 20260618/          # 主要结果（因子审计、IC图、Tier1-2图表）
│   └── 20260619/          # Tier2-3 补充实验
└── data/
    ├── 全部A股.xlsx
    └── 全部基金(主代码).xlsx
```

---

## 快速开始

```bash
pip install -r requirements.txt

# 因子构建（输出 CSV + 图表）
python validate_factor.py

# IC 回测（需提供 CSMAR 收益率 CSV 路径）
python ic_backtest.py --returns /path/to/daily_returns_cn_all.csv

# Tier1-2 稳健性实验
python tier1_2_combined.py --returns /path/to/daily_returns_cn_all.csv

# Tier2-3 异质性+合成实验
python tier2_3_remaining.py --returns /path/to/daily_returns_cn_all.csv
```

---

## QMT 适配

```python
import pandas as pd

# 加载因子
factor = pd.read_csv("results/20260618/stealth_score_factor.csv")
# 列：stock_name, StealthScore, hidden_count, total_funds, HiddenRatio, coverage_flag

# 合并到 QMT 因子表（列名符合 QMT 规范）
main_factor = main_factor.merge(
    factor[["stock_name", "StealthScore", "CoverageBreadth"]],
    on="stock_name", how="left"
)
```

---

## 版本记录

| 版本 | 日期 | 变更 |
|------|------|------|
| v0.1.0 | 2026-06-18 | 初始发布，HiddenRatio 原型 |
| v0.2.0 | 2026-06-18 | StealthScore 重设计，修复6处 Bug |
| v0.2.1 | 2026-06-19 | 接入 CSMAR 真实收益率，完成 Tier1-3 全实验 |

---

## 作者

Leo Li（李硕仁）  
广东外语外贸大学 金融工程 2023级

## License

MIT
