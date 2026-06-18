# -*- coding: utf-8 -*-
"""
隐形重仓对因子构建模块
修复原型 Bug：
  - stage_stockcentric 参数顺序（原代码把 edges_fund_to_stock 和 edges_stock_to_fund 顺序写错）
  - rank_stockcentric 接收 stage_df 参数名拼写错误（原代码写为 stage_df）
  - 新增因子打分输出（HiddenRatio 作为连续因子）
  - 新增日期戳输出
"""

import pandas as pd
from datetime import datetime
from typing import Optional

from .data_loader import guess_is_active_fund


def build_stage_df(edges_stock_to_fund: pd.DataFrame,
                   edges_fund_to_stock: pd.DataFrame,
                   active_only: bool = False) -> pd.DataFrame:
    """
    以股票为视角，找出「基金持有该股票但未列入自身 Top5」的隐形重仓对。

    参数
    ----
    edges_stock_to_fund : DataFrame
        [stock_name, fund_name, fund_name_key] — 来自股票侧「前十大持股基金名称」
    edges_fund_to_stock : DataFrame
        [fund_code, fund_name, fund_name_key, stock_name] — 来自基金侧 Top1–5 重仓股
    active_only : bool
        若为 True，仅保留主动型基金（过滤含 ETF/指数等关键词的基金）

    返回
    ----
    DataFrame，含列：
        stock_name, fund_name, fund_name_key,
        listed_in_fund_top5, is_hidden_pair
    """
    lhs = edges_stock_to_fund[["stock_name", "fund_name", "fund_name_key"]].drop_duplicates()

    if active_only:
        lhs = lhs[lhs["fund_name"].apply(lambda x: guess_is_active_fund(str(x)))]

    rhs = edges_fund_to_stock[["stock_name", "fund_name_key"]].drop_duplicates()

    merged = lhs.merge(rhs, on=["stock_name", "fund_name_key"], how="left", indicator=True)
    merged["listed_in_fund_top5"] = merged["_merge"].eq("both")
    merged["is_hidden_pair"] = ~merged["listed_in_fund_top5"]

    return merged.drop(columns=["_merge"])


def build_rank_df(stage_df: pd.DataFrame,
                  edges_stock_to_fund: pd.DataFrame) -> pd.DataFrame:
    """
    按股票聚合，生成因子值。

    输出列：
      - stock_name
      - HiddenPairsCount  : 隐形持有该股票的基金数
      - StockFundTop10Count : 该股票的前十大持股基金总数
      - HiddenRatio         : HiddenPairsCount / StockFundTop10Count（因子值）
    """
    hidden = stage_df[stage_df["is_hidden_pair"]].copy()

    hidden_pairs = (
        hidden.groupby("stock_name")["fund_name"]
        .nunique()
        .reset_index(name="HiddenPairsCount")
    )

    stock_total = (
        edges_stock_to_fund.groupby("stock_name")["fund_name"]
        .nunique()
        .reset_index(name="StockFundTop10Count")
    )

    res = stock_total.merge(hidden_pairs, on="stock_name", how="left")
    res["HiddenPairsCount"] = res["HiddenPairsCount"].fillna(0).astype(int)
    res["HiddenRatio"] = (res["HiddenPairsCount"] / res["StockFundTop10Count"]).round(4)

    return res.sort_values(
        ["HiddenPairsCount", "HiddenRatio", "StockFundTop10Count"],
        ascending=[False, False, False]
    ).reset_index(drop=True)


def build_factor_csv(rank_df: pd.DataFrame,
                    trade_date: Optional[str] = None) -> pd.DataFrame:
    """
    生成可直接用于 QMT 的因子 CSV。
    列：stock_name, HiddenRatio, factor_date
    """
    df = rank_df[["stock_name", "HiddenRatio"]].copy()
    df["factor_date"] = trade_date or datetime.now().strftime("%Y%m%d")
    return df[["stock_name", "factor_date", "HiddenRatio"]]
