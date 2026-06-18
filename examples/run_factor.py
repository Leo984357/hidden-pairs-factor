# -*- coding: utf-8 -*-
"""
使用示例：运行 hidden-pairs-factor 并查看结果
"""

import sys
import os

# 确保 src/ 在 Python 路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd
from datetime import datetime

# 方法一：通过 CLI 运行（推荐）
# =============================================================================
# 在终端执行：
#
#   cd /path/to/hidden-pairs-factor
#   pip install -r requirements.txt
#   python -m hidden_pairs.cli \
#       --fund  /path/to/全部基金(主代码).xlsx \
#       --stocks /path/to/全部A股.xlsx \
#       --outdir ./output
#
# 输出文件：
#   ./output/hidden_pairs_stage_YYYYMMDD.csv
#   ./output/hidden_pairs_rank_YYYYMMDD.csv   ← 因子文件
#   ./output/hidden_pairs_rank_YYYYMMDD.xlsx
#   ./output/charts/*.png
# =============================================================================


# 方法二：作为 Python 模块调用
# =============================================================================
from hidden_pairs.data_loader import (
    read_xls_with_header,
    parse_fund_top5_edges,
    parse_stock_fund_edges,
)
from hidden_pairs.factor_builder import (
    build_stage_df,
    build_rank_df,
    build_factor_csv,
)


def run_as_module():
    # 1. 加载数据
    fund_df = read_xls_with_header("全部基金(主代码).xlsx")
    stocks_df = read_xls_with_header("全部A股.xlsx")

    # 2. 解析关系
    edges_fund_to_stock = parse_fund_top5_edges(fund_df)
    edges_stock_to_fund = parse_stock_fund_edges(stocks_df)

    # 3. 构建隐形重仓对
    stage_df = build_stage_df(
        edges_stock_to_fund, edges_fund_to_stock, active_only=False
    )

    # 4. 聚合因子值
    rank_df = build_rank_df(stage_df, edges_stock_to_fund)

    # 5. 导出因子 CSV（可直接用于 QMT）
    today = datetime.now().strftime("%Y%m%d")
    out = f"./output/hidden_pairs_rank_{today}.csv"
    rank_df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"因子文件已导出：{out}")

    # 6. 查看 Top 10
    print("\nTop 10 股票（按 HiddenPairsCount 排序）：")
    print(rank_df.head(10).to_string(index=False))

    return rank_df


if __name__ == "__main__":
    run_as_module()
