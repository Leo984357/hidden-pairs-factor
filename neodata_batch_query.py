"""
NeoData Batch Fund Holdings Query
═══════════════════════════════════════════════════════════════════════
Batch-query NeoData for the latest fund holdings of 200 sample funds.
Parse the markdown table response and save to CSV.

Output: data/neoda_current_holdings.csv
        columns: fund_code, fund_name, report_date, stock_name, stock_code,
                 weight, market_value, industry_l1, industry_l2
"""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

# ── Config ──────────────────────────────────────────────────────────────
SCRIPT_PATH = Path.home() / ".workbuddy/skills/skill_2053082432761950208/scripts/query.py"
PYTHON_BIN = "/Users/leolee/.workbuddy/binaries/python/envs/default/bin/python3"
SAMPLE_FILE = Path(__file__).parent / "data/neodata_sample_funds.csv"
OUTPUT_FILE = Path(__file__).parent / "data/neodata_current_holdings.csv"
DELAY = 0.5  # seconds between queries
BATCH_SIZE = 200


def clean_fund_code(code: str) -> str:
    """Strip exchange suffixes for NeoData query."""
    code = str(code).strip()
    for suffix in [".OF", ".SZ", ".SH", ".JJ"]:
        if code.endswith(suffix):
            code = code[: -len(suffix)]
    return code


def query_neodata(fund_code: str) -> dict | None:
    """Query NeoData for a single fund's holdings."""
    query = f"{fund_code} 基金重仓资产"
    try:
        result = subprocess.run(
            [PYTHON_BIN, str(SCRIPT_PATH), "--query", query, "--data-type", "api"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception):
        return None


def parse_holdings(response: dict) -> list[dict]:
    """Parse NeoData markdown table response into structured records."""
    if not response:
        return []

    api_data = response.get("data", {}).get("apiData", {})
    recalls = api_data.get("apiRecall", [])

    records = []
    for recall in recalls:
        if "基金重仓" not in recall.get("type", "") and "基金重仓" not in recall.get("desc", ""):
            continue

        content = recall.get("content", "")

        # Extract fund code, name, report date
        fund_code = ""
        fund_name = ""
        report_date = ""

        for line in content.split("\n"):
            if "基金代码" in line and "**" in line:
                m = re.search(r"\*\*基金代码\*\*:\s*(\S+)", line)
                if m:
                    fund_code = m.group(1)
            elif "基金名称" in line and "**" in line:
                m = re.search(r"\*\*基金名称\*\*:\s*(.+)", line)
                if m:
                    fund_name = m.group(1).strip()
            elif "报告期" in line and "**" in line:
                m = re.search(r"\*\*报告期\*\*:\s*(\S+)", line)
                if m:
                    report_date = m.group(1)

        # Parse stock table
        lines = content.split("\n")
        in_stock_table = False
        for line in lines:
            if "| 股票名称 |" in line or "|:---:" in line:
                in_stock_table = True
                continue
            if in_stock_table:
                if line.strip().startswith("|") and "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    parts = [p for p in parts if p]  # remove empty
                    if len(parts) >= 5 and parts[0] not in ["股票名称", "---"]:
                        try:
                            record = {
                                "fund_code": fund_code,
                                "fund_name": fund_name,
                                "report_date": report_date,
                                "stock_name": parts[0],
                                "stock_code": parts[1],
                                "weight": float(parts[8]) if len(parts) > 8 and parts[8] != "--" else None,
                                "market_value": float(parts[6].replace(",", "")) if len(parts) > 6 and parts[6] != "--" else None,
                                "industry_l1": parts[7] if len(parts) > 7 else "",
                                "industry_l2": parts[3] if len(parts) > 3 else "",
                            }
                            records.append(record)
                        except (ValueError, IndexError):
                            continue
                elif line.strip() == "" or "债券列表" in line or "基金列表" in line:
                    in_stock_table = False

    return records


def main():
    # Load sample funds
    sample = pd.read_csv(SAMPLE_FILE)
    print(f"Loaded {len(sample)} sample funds")

    all_records = []
    success = 0
    fail = 0

    for i, row in sample.head(BATCH_SIZE).iterrows():
        fund_code = clean_fund_code(row["fund_code"])
        fund_name = row.get("fund_name", "")

        # Query NeoData
        response = query_neodata(fund_code)

        if response is None:
            fail += 1
            print(f"  [{i+1}/{BATCH_SIZE}] {fund_code} ({fund_name[:15]}) → FAIL (no response)")
            time.sleep(DELAY)
            continue

        records = parse_holdings(response)

        if records:
            success += 1
            n_stocks = len(records)
            report_date = records[0].get("report_date", "?")
            print(f"  [{i+1}/{BATCH_SIZE}] {fund_code} ({fund_name[:15]}) → OK ({n_stocks} stocks, {report_date})")
            all_records.extend(records)
        else:
            fail += 1
            print(f"  [{i+1}/{BATCH_SIZE}] {fund_code} ({fund_name[:15]}) → EMPTY")

        time.sleep(DELAY)

    # Save results
    if all_records:
        df = pd.DataFrame(all_records)
        df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
        print(f"\n{'='*60}")
        print(f"SUCCESS: {success}, FAIL: {fail}")
        print(f"Total records: {len(all_records)}")
        print(f"Unique funds: {df['fund_code'].nunique()}")
        print(f"Report dates: {df['report_date'].unique()}")
        print(f"Saved to: {OUTPUT_FILE}")
    else:
        print("\nNo records collected!")


if __name__ == "__main__":
    main()
