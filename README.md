# hidden-pairs-factor

股票视角隐形重仓对因子挖掘项目。

## 背景

基金季报披露前五大重仓股，但一只股票的前十大持股基金中，部分基金并未将该股票列入自身前五大重仓。
这些**"隐形重仓对"（hidden pairs）** 反映了基金真实持仓意图与市场可见信息之间的偏差，
可作为选股因子使用。

## 因子逻辑

| 口径 | 含义 |
|---|---|
| `HiddenPairsCount` | 该股票被多少只基金"隐形持有" |
| `HiddenRatio` | 隐形持有数 / 前十大持股基金总数 |
| `StockFundTop10Count` | 前十大持股基金总数 |

**假设**：HiddenRatio 越高，说明该股票被多只基金实质持有但未被市场充分认知，
可能存在信息差带来的定价偏离。

## 项目结构

```
hidden-pairs-factor/
├── src/hidden_pairs/
│   ├── __init__.py
│   ├── config.py          # 配置常量
│   ├── data_loader.py     # Excel 解析（自动识别表头）
│   ├── factor_builder.py  # 核心：构建 hidden pairs + 因子打分
│   ├── visualizer.py      # 可视化（TopN柱状、占比、散点）
│   └── cli.py            # 命令行入口
├── tests/
│   └── test_factor_builder.py
├── examples/
│   └── run_factor.py     # 使用示例
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── README.md
```

## 快速开始

```bash
pip install -r requirements.txt

# 基本运行（输出因子 CSV + 图表）
python -m hidden_pairs.cli \
  --fund 全部基金(主代码).xlsx \
  --stocks 全部A股.xlsx \
  --outdir ./output

# 只看主动型基金
python -m hidden_pairs.cli \
  --fund 全部基金(主代码).xlsx \
  --stocks 全部A股.xlsx \
  --active-only \
  --outdir ./output

# 跳过图表（服务器环境）
python -m hidden_pairs.cli \
  --fund 全部基金(主代码).xlsx \
  --stocks 全部A股.xlsx \
  --no-charts
```

## 输出文件

| 文件 | 说明 |
|---|---|
| `hidden_pairs_stage_YYYYMMDD.csv` | 每只股票-基金对是否为 hidden pair |
| `hidden_pairs_rank_YYYYMMDD.csv` | 按股票聚合的因子值（可直接用于 QMT） |
| `hidden_pairs_rank_YYYYMMDD.xlsx` | 同上（Excel 格式，含两页） |
| `charts/topN_hidden.png` | Top N 股票 HiddenPairsCount 柱状图 |
| `charts/ratio_dist.png` | HiddenRatio 分布直方图 |
| `charts/scatter_hidden.png` | StockFundTop10Count vs HiddenPairsCount 散点图 |

## QMT 适配

`hidden_pairs_rank_YYYYMMDD.csv` 可直接作为 QMT 因子文件使用：

```python
# QMT 因子加载示例
import pandas as pd
factor = pd.read_csv("hidden_pairs_rank_20250115.csv")
# 合并到主因子表
main_factor = main_factor.merge(factor[["stock_name","HiddenRatio"]], on="stock_name", how="left")
```

## 数据来源

- **全部A股.xlsx**：东方财富/同花顺导出，含「前十大持股基金名称」列
- **全部基金(主代码).xlsx**：基金季报数据，含 Top1–Top5 重仓股列

两份文件放入同一目录，运行 `cli.py` 即可。

## 因子表现（待更新）

| 回测区间 | IC | IC_IR | 多空收益（年化） |
|---|---|---|---|
| 2019-2024 | TBD | TBD | TBD |

## 开发计划

- [x] 修复原型 Bug（函数名拼写、参数顺序、Unicode 范围）
- [x] 补全图表生成
- [x] 输出文件加日期戳
- [x] 暴露 `--active-only` CLI 参数
- [ ] 接入 QMT 回测框架，计算 IC / 分层收益
- [ ] 增加时序维度（多期季报数据）
- [ ] 发布到 PyPI

## 作者

Leo Li（李硕仁）

## License

MIT
