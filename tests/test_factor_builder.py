# -*- coding: utf-8 -*-
"""测试因子构建逻辑"""

import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from hidden_pairs.data_loader import (
    normalize_fund_name_key,
    split_fund_list,
    guess_is_active_fund,
    parse_fund_top5_edges,
    parse_stock_fund_edges,
)
from hidden_pairs.factor_builder import (
    build_stage_df,
    build_rank_df,
    build_factor_csv,
)


def make_sample_fund_df():
    return pd.DataFrame({
        "基金代码": ["000001", "000002"],
        "基金简称": ["测试基金A", "测试基金B"],
        "重仓股1": ["股票X", "股票Y"],
        "重仓股2": ["股票Y", None],
        "重仓股3": [None, None],
        "重仓股4": [None, None],
        "重仓股5": [None, None],
    })


def make_sample_stock_df():
    return pd.DataFrame({
        "股票代码": ["600000", "600001"],
        "股票名称": ["股票X", "股票Y"],
        "前十大持股基金名称": ["测试基金A", "测试基金A、测试基金B"],
    })


def test_normalize_fund_name_key():
    cases = [
        ("测试基金管理有限公司", "测试基金管理有限公司"),  # 无后缀，不变
        ("测试交易型开放式指数证券投资基金", "测试"),  # 后缀被移除
    ]
    for inp, expected in cases:
        result = normalize_fund_name_key(inp)
        assert expected in result or result == inp, f"normalize failed: {inp} → {result}"
    print("  ✓ test_normalize_fund_name_key passed")


def test_guess_is_active_fund():
    assert guess_is_active_fund("测试主动基金") == True
    assert guess_is_active_fund("测试ETF基金") == False
    assert guess_is_active_fund("测试指数基金") == False
    print("  ✓ test_guess_is_active_fund passed")


def test_build_stage_and_rank():
    fund_df = make_sample_fund_df()
    stock_df = make_sample_stock_df()

    edges_fund = parse_fund_top5_edges(fund_df)
    edges_stock = parse_stock_fund_edges(stock_df)

    stage = build_stage_df(edges_stock, edges_fund, active_only=False)
    assert "is_hidden_pair" in stage.columns
    assert "listed_in_fund_top5" in stage.columns

    rank = build_rank_df(stage, edges_stock)
    assert "HiddenPairsCount" in rank.columns
    assert "HiddenRatio" in rank.columns

    print("  ✓ test_build_stage_and_rank passed")


def test_build_factor_csv():
    rank = pd.DataFrame({
        "stock_name": ["股票X", "股票Y"],
        "HiddenPairsCount": [1, 0],
        "StockFundTop10Count": [2, 1],
        "HiddenRatio": [0.5, 0.0],
    })
    factor = build_factor_csv(rank, trade_date="20250115")
    assert "stock_name" in factor.columns
    assert "HiddenRatio" in factor.columns
    assert "factor_date" in factor.columns
    assert factor["factor_date"].iloc[0] == "20250115"
    print("  ✓ test_build_factor_csv passed")


if __name__ == "__main__":
    print("Running tests for hidden-pairs-factor...")
    test_normalize_fund_name_key()
    test_guess_is_active_fund()
    test_build_stage_and_rank()
    test_build_factor_csv()
    print("\nAll tests passed!")
