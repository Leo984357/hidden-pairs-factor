# -*- coding: utf-8 -*-
"""
命令行入口
修复原型 Bug：
  - --no-charts 参数未实现（现已实现）
  - --active-only 参数未暴露（现已暴露）
  - 输出文件无日期戳（现已加日期戳）
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

from .data_loader import parse_fund_top5_edges, parse_stock_fund_edges, read_xls_with_header
from .factor_builder import build_stage_df, build_rank_df, build_factor_csv
from .visualizer import generate_all_charts


def _banner(msg: str):
    print(f"[hidden-pairs] {msg}", file=sys.stderr)


def main(argv: list = None):
    ap = argparse.ArgumentParser(
        description="hidden-pairs-factor: 隐形重仓对因子挖掘工具"
    )
    ap.add_argument(
        "--fund", default="全部基金(主代码).xlsx",
        help="基金重仓股 Excel 路径（默认：当前目录下的 全部基金(主代码).xlsx）"
    )
    ap.add_argument(
        "--stocks", default="全部A股.xlsx",
        help="股票前十大持股基金 Excel 路径（默认：当前目录下的 全部A股.xlsx）"
    )
    ap.add_argument(
        "--outdir", default="./output",
        help="输出目录（默认：./output）"
    )
    ap.add_argument(
        "--active-only", action="store_true",
        help="仅保留主动型基金（过滤 ETF/指数/联接/增强）"
    )
    ap.add_argument(
        "--no-charts", action="store_true",
        help="不生成图表（适用于无 GUI 的服务器环境）"
    )
    ap.add_argument(
        "--top-n", type=int, default=20,
        help="Top N 柱状图显示股票数（默认：20）"
    )
    args = ap.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    date_str = datetime.now().strftime("%Y%m%d")
    _banner(f"读取基金数据：{args.fund}")
    fund_df = read_xls_with_header(args.fund)
    _banner(f"读取股票数据：{args.stocks}")
    stocks_df = read_xls_with_header(args.stocks)

    _banner("解析基金 Top5 重仓关系...")
    edges_fund_to_stock = parse_fund_top5_edges(fund_df)
    _banner("解析股票→基金持有关系...")
    edges_stock_to_fund = parse_stock_fund_edges(stocks_df)

    _banner(f"构建隐形重仓对（active_only={args.active_only}）...")
    stage_df = build_stage_df(
        edges_stock_to_fund, edges_fund_to_stock, active_only=args.active_only
    )
    rank_df = build_rank_df(stage_df, edges_stock_to_fund)

    # 输出 Stage CSV
    stage_path = outdir / f"hidden_pairs_stage_{date_str}.csv"
    stage_df.to_csv(stage_path, index=False, encoding="utf-8-sig")
    _banner(f"已导出 Stage CSV：{stage_path}")

    # 输出 Rank CSV（因子文件，可直接用于 QMT）
    rank_path_csv = outdir / f"hidden_pairs_rank_{date_str}.csv"
    rank_df.to_csv(rank_path_csv, index=False, encoding="utf-8-sig")
    _banner(f"已导出 Rank CSV（因子文件）：{rank_path_csv}")

    # 输出 Rank Excel（含 Stage 和 Rank 两页）
    rank_path_xlsx = outdir / f"hidden_pairs_rank_{date_str}.xlsx"
    with pd.ExcelWriter(rank_path_xlsx, engine="openpyxl") as writer:
        stage_df.to_excel(writer, index=False, sheet_name="Stage")
        rank_df.to_excel(writer, index=False, sheet_name="Rank")
    _banner(f"已导出 Rank Excel：{rank_path_xlsx}")

    # 生成图表（除非 --no-charts）
    if not args.no_charts:
        _banner("生成图表...")
        try:
            chart_paths = generate_all_charts(stage_df, rank_df, str(outdir))
            for cp in chart_paths:
                _banner(f"已生成图表：{cp}")
        except Exception as e:
            _banner(f"图表生成失败（可加 --no-charts 跳过）：{e}")

    # 输出因子 CSV（QMT 适配格式）
    factor_df = build_factor_csv(rank_df, trade_date=date_str)
    factor_path = outdir / f"hidden_pairs_factor_{date_str}.csv"
    factor_df.to_csv(factor_path, index=False, encoding="utf-8-sig")
    _banner(f"已导出 QMT 因子文件：{factor_path}")

    _banner("全部完成！")
    return 0


if __name__ == "__main__":
    sys.exit(main())
