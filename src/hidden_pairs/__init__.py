# -*- coding: utf-8 -*-
"""hidden-pairs-factor package."""

__version__ = "0.1.0"
__author__ = "Leo Li (李硕仁)"

from .data_loader import (
    detect_header_row,
    read_xls_with_header,
    normalize_fund_name_key,
    split_fund_list,
    guess_is_active_fund,
    parse_fund_top5_edges,
    parse_stock_fund_edges,
)
from .factor_builder import (
    build_stage_df,
    build_rank_df,
    build_factor_csv,
)

__all__ = [
    "detect_header_row",
    "read_xls_with_header",
    "normalize_fund_name_key",
    "split_fund_list",
    "guess_is_active_fund",
    "parse_fund_top5_edges",
    "parse_stock_fund_edges",
    "build_stage_df",
    "build_rank_df",
    "build_factor_csv",
]
