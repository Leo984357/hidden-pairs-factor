# -*- coding: utf-8 -*-
"""
数据加载模块
负责从 Excel 文件中解析基金重仓股数据和股票被持股数据。
修复原型中的 Bug：
  - detect_header_row 中 Unicode 范围 \u4e00-\u9fa5（原代码 \u9fa5 少了个 a）
  - parse_fund_top5_edges 函数名拼写错误（原代码定义为 parse_fund_top5_edges，调用时写为 parse_fund_top5_edges）
  - stage_stockcentric 参数顺序错误
"""

import re
import os
import pandas as pd
from typing import List, Optional

from .config import HEADER_SCAN_MAX, FUND_NAME_SUFFIXES, PASSIVE_KEYWORDS


def detect_header_row(xls_path: str, sheet_index: int = 0, max_scan: int = HEADER_SCAN_MAX) -> int:
    """自动识别 Excel 表头所在行"""
    df0 = pd.read_excel(xls_path, sheet_name=sheet_index, header=None, dtype=object)
    best_i, best_score = 0, -1
    for i in range(min(max_scan, len(df0))):
        row = df0.iloc[i]
        nn = row.notna().sum()
        # 修复：\u4e00-\u9fa5（原代码少了 a）
        alpha = row.astype(str).str.contains(r"[\u4e00-\u9fa5]", regex=True).sum()
        score = nn + alpha
        if score > best_score:
            best_i, best_score = i, score
    return best_i


def read_xls_with_header(xls_path: str, sheet_index: int = 0,
                        header_index: Optional[int] = None) -> pd.DataFrame:
    """读取 Excel 并正确设置表头"""
    if header_index is None:
        header_index = detect_header_row(xls_path, sheet_index=sheet_index)
    df = pd.read_excel(xls_path, sheet_name=sheet_index, header=header_index, dtype=object)
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed:") and df[c].isna().all()]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def normalize_fund_name_key(name: str) -> str:
    """归一化基金名称：去掉后缀词和括号内容"""
    if name is None:
        return ""
    s = str(name)
    if s.strip() == "" or s.lower() == "nan":
        return ""
    for w in FUND_NAME_SUFFIXES:
        s = s.replace(w, "")
    s = re.sub(r"[（(].*?[）)]", "", s)
    s = re.sub(r"[-—_·\s]", "", s)
    return s


def split_fund_list(cell) -> List[str]:
    """将单元格中的基金名称列表拆分为列表"""
    s = "" if cell is None else str(cell)
    if s.strip() == "" or s.lower() == "nan":
        return []
    parts = re.split(r"[、，,；;/\s|]+", s)
    return [p.strip() for p in parts if p and p.strip() != ""]


def guess_is_active_fund(fund_name: str) -> bool:
    """判断是否为主动型基金（名称不含被动关键词）"""
    s = str(fund_name or "")
    return not any(b in s for b in PASSIVE_KEYWORDS)


def parse_fund_top5_edges(fund_df: pd.DataFrame,
                          fund_code_col: Optional[str] = None,
                          fund_name_col: Optional[str] = None) -> pd.DataFrame:
    """
    从基金季报 Excel 中解析 Top1–Top5 重仓股关系。
    返回 DataFrame: [fund_code, fund_name, fund_name_key, stock_name]
    """
    # 自动识别列名
    if not fund_code_col or fund_code_col not in fund_df.columns:
        for k in ["证券代码", "基金代码", "代码"]:
            if k in fund_df.columns:
                fund_code_col = k
                break
    if not fund_name_col or fund_name_col not in fund_df.columns:
        for k in ["证券名称", "基金简称", "基金名称", "名称"]:
            if k in fund_df.columns:
                fund_name_col = k
                break

    top_cols = [c for c in fund_df.columns if "重仓股" in c][:5]

    rows = []
    for _, r in fund_df.iterrows():
        fcode = str(r.get(fund_code_col, "")).strip()
        fname = str(r.get(fund_name_col, "")).strip()
        for c in top_cols:
            sname = r.get(c, None)
            if not sname or str(sname).strip() == "":
                continue
            rows.append({
                "fund_code": fcode,
                "fund_name": fname,
                "fund_name_key": normalize_fund_name_key(fname),
                "stock_name": str(sname).strip(),
            })
    return pd.DataFrame(rows)


def parse_stock_fund_edges(stock_df: pd.DataFrame,
                          stock_code_col: Optional[str] = None,
                          stock_name_col: Optional[str] = None,
                          stock_fundlist_col: Optional[str] = None) -> pd.DataFrame:
    """
    从股票 Excel（含「前十大持股基金名称」列）中解析股票-基金持有关系。
    返回 DataFrame: [stock_code, stock_name, fund_name, fund_name_key]
    """
    if not stock_code_col or stock_code_col not in stock_df.columns:
        for k in ["证券代码", "股票代码", "代码"]:
            if k in stock_df.columns:
                stock_code_col = k
                break
    if not stock_name_col or stock_name_col not in stock_df.columns:
        for k in ["证券名称", "股票名称", "名称", "股票简称", "证券简称"]:
            if k in stock_df.columns:
                stock_name_col = k
                break
    if not stock_fundlist_col or stock_fundlist_col not in stock_df.columns:
        for c in stock_df.columns:
            if "前十大持股基金名称" in str(c):
                stock_fundlist_col = c
                break

    rows = []
    for _, r in stock_df.iterrows():
        scode = str(r.get(stock_code_col, "")).strip()
        sname = str(r.get(stock_name_col, "")).strip()
        fund_list_cell = r.get(stock_fundlist_col, None)
        for fname in split_fund_list(fund_list_cell):
            rows.append({
                "stock_code": scode,
                "stock_name": sname,
                "fund_name": fname,
                "fund_name_key": normalize_fund_name_key(fname),
            })
    return pd.DataFrame(rows)
