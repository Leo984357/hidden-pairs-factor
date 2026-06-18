"""
可视化模块
生成三张分析图表：
  1. Top N 股票 HiddenPairsCount 柱状图
  2. HiddenRatio 分布直方图
  3. StockFundTop10Count vs HiddenPairsCount 散点图
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from .config import CHART_DIR


def _ensure_chart_dir(outdir: str):
    path = os.path.join(outdir, CHART_DIR)
    os.makedirs(path, exist_ok=True)
    return path


def plot_top_hidden(stage_df: pd.DataFrame, rank_df: pd.DataFrame,
                    outdir: str, top_n: int = 20):
    """Top N 股票 HiddenPairsCount 柱状图"""
    chart_dir = _ensure_chart_dir(outdir)
    top = rank_df.head(top_n)
    plt.figure(figsize=(12, 6))
    plt.bar(top["stock_name"].astype(str), top["HiddenPairsCount"], color="#4A90D6")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.xlabel("Stock Name")
    plt.ylabel("HiddenPairsCount")
    plt.title(f"Top {top_n} Stocks by HiddenPairsCount")
    plt.tight_layout()
    path = os.path.join(chart_dir, "topN_hidden.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_ratio_distribution(rank_df: pd.DataFrame, outdir: str):
    """HiddenRatio 分布直方图"""
    chart_dir = _ensure_chart_dir(outdir)
    plt.figure(figsize=(10, 5))
    plt.hist(rank_df["HiddenRatio"].dropna(), bins=50, color="#50E3C2", edgecolor="black")
    plt.xlabel("HiddenRatio")
    plt.ylabel("Frequency")
    plt.title("Distribution of HiddenRatio")
    plt.tight_layout()
    path = os.path.join(chart_dir, "ratio_dist.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_scatter(stage_df: pd.DataFrame, rank_df: pd.DataFrame, outdir: str):
    """StockFundTop10Count vs HiddenPairsCount 散点图"""
    chart_dir = _ensure_chart_dir(outdir)
    merged = stage_df.merge(rank_df[["stock_name", "StockFundTop10Count"]],
                           on="stock_name", how="left")
    hidden = merged[merged["is_hidden_pair"]]
    plt.figure(figsize=(10, 6))
    plt.scatter(hidden["StockFundTop10Count"], hidden["fund_name"].apply(lambda x: 1),
                alpha=0.5, s=20)
    plt.xlabel("StockFundTop10Count")
    plt.ylabel("Hidden Pair (count)")
    plt.title("Hidden Pairs vs Total Fund Holdings")
    plt.tight_layout()
    path = os.path.join(chart_dir, "scatter_hidden.png")
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def generate_all_charts(stage_df: pd.DataFrame, rank_df: pd.DataFrame, outdir: str):
    """一键生成全部图表"""
    paths = []
    paths.append(plot_top_hidden(stage_df, rank_df, outdir))
    paths.append(plot_ratio_distribution(rank_df, outdir))
    paths.append(plot_scatter(stage_df, rank_df, outdir))
    return paths
