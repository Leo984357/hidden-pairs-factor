# -*- coding: utf-8 -*-
"""
配置常量
"""

# 基金名称归一化：需要去掉的后缀词
FUND_NAME_SUFFIXES = [
    "交易型开放式指数证券投资基金",
    "证券投资基金",
    "混合型",
    "股票型",
    "债券型",
    "指数",
    "联接",
    "增强",
    "发起式",
    "发起式证券投资基金",
]

# 主动型基金判断：名称含以下词则视为被动/指数型
PASSIVE_KEYWORDS = ["ETF", "交易型开放式指数", "指数", "联接", "增强"]

# Excel 解析：自动识别表头时最多扫描行数
HEADER_SCAN_MAX = 30

# 输出文件名模板（{} 处填入 YYYYMMDD）
OUTPUT_STAGE_CSV = "hidden_pairs_stage_{}.csv"
OUTPUT_RANK_CSV = "hidden_pairs_rank_{}.csv"
OUTPUT_RANK_XLSX = "hidden_pairs_rank_{}.xlsx"
CHART_DIR = "charts"
