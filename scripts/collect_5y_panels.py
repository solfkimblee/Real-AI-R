"""采集 THS 行业板块5年历史数据，生成多时间尺度回测面板。

Usage:
    python scripts/collect_5y_panels.py
"""

from __future__ import annotations

import time
import sys
from datetime import datetime, timedelta

import akshare as ak
import numpy as np
import pandas as pd


# 90 THS industry boards (from existing panel_train.parquet)
BOARDS = [
    "IT服务", "专用设备", "中药", "互联网电商", "保险", "元件",
    "光伏设备", "光学光电子", "公路铁路运输", "其他电子", "其他电源设备",
    "其他社会服务", "养殖业", "军工电子", "军工装备", "农产品加工",
    "农化制品", "包装印刷", "化学制品", "化学制药", "化学原料",
    "化学纤维", "医疗器械", "医疗服务", "医药商业", "半导体",
    "厨卫电器", "塑料制品", "多元金融", "家居用品", "小家电",
    "小金属", "工业金属", "工程机械", "建筑材料", "建筑装饰",
    "影视院线", "房地产", "教育", "文化传媒", "旅游及酒店",
    "服装家纺", "机场航运", "橡胶制品", "汽车整车", "汽车服务及其他",
    "汽车零部件", "油气开采及服务", "消费电子", "港口航运", "游戏",
    "煤炭开采加工", "燃气", "物流", "环保设备", "环境治理",
    "生物制品", "电力", "电子化学品", "电机", "电池",
    "电网设备", "白色家电", "白酒", "石油加工贸易", "种植业与林业",
    "纺织制造", "综合", "美容护理", "能源金属", "自动化设备",
    "计算机设备", "证券", "贵金属", "贸易", "轨交设备",
    "软件开发", "通信服务", "通信设备", "通用设备", "造纸",
    "金属新材料", "钢铁", "银行", "零售", "非金属材料",
    "风电设备", "食品加工制造", "饮料制造", "黑色家电",
]

# Simple sector grouping for rise_count/fall_count derivation
SECTOR_GROUPS = {
    "tech": ["IT服务", "半导体", "光学光电子", "消费电子", "电子化学品",
             "元件", "其他电子", "计算机设备", "软件开发", "通信设备",
             "通信服务"],
    "new_energy": ["光伏设备", "电池", "风电设备", "其他电源设备",
                   "电网设备", "能源金属", "电力", "自动化设备"],
    "pharma": ["中药", "化学制药", "医疗器械", "医疗服务", "医药商业",
               "生物制品"],
    "manufacturing": ["专用设备", "工程机械", "通用设备", "轨交设备",
                      "汽车整车", "汽车零部件", "汽车服务及其他", "电机"],
    "materials": ["化学制品", "化学原料", "化学纤维", "建筑材料",
                  "塑料制品", "橡胶制品", "金属新材料", "非金属材料",
                  "钢铁", "小金属", "工业金属", "造纸", "包装印刷"],
    "consumer": ["白酒", "饮料制造", "食品加工制造", "家居用品",
                 "小家电", "白色家电", "黑色家电", "厨卫电器",
                 "美容护理", "服装家纺", "纺织制造"],
    "energy_resource": ["煤炭开采加工", "油气开采及服务", "石油加工贸易",
                        "燃气", "贵金属"],
    "finance": ["银行", "保险", "证券", "多元金融"],
    "service": ["互联网电商", "游戏", "文化传媒", "教育", "影视院线",
                "旅游及酒店", "其他社会服务", "物流", "零售", "贸易"],
    "infra": ["公路铁路运输", "港口航运", "机场航运", "建筑装饰",
              "房地产", "环保设备", "环境治理"],
    "agri": ["农产品加工", "农化制品", "养殖业", "种植业与林业"],
    "other": ["综合"],
}

# Build board -> group mapping
BOARD_TO_GROUP: dict[str, str] = {}
for group, members in SECTOR_GROUPS.items():
    for m in members:
        BOARD_TO_GROUP[m] = group


def fetch_board_data(board_name: str, start: str, end: str) -> pd.DataFrame:
    """Fetch one board's THS industry index data."""
    try:
        df = ak.stock_board_industry_index_ths(
            symbol=board_name, start_date=start, end_date=end,
        )
    except Exception as e:
        print(f"  [warn] {board_name}: {e}")
        return pd.DataFrame()

    if df.empty:
        return df

    col_map = {
        "日期": "date", "开盘价": "open", "最高价": "high",
        "最低价": "low", "收盘价": "close",
        "成交量": "volume", "成交额": "amount",
    }
    df = df.rename(columns=col_map)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["name"] = board_name

    # Compute change_pct from close prices
    df["change_pct"] = df["close"].pct_change() * 100
    df = df.dropna(subset=["change_pct"]).reset_index(drop=True)

    # Compute turnover proxy: normalized daily volume
    vol_ma20 = df["volume"].rolling(20, min_periods=1).mean()
    df["turnover_rate"] = (df["volume"] / vol_ma20).clip(0.01, 10.0)

    # Lead stock proxy: intraday range relative to open
    df["lead_stock_pct"] = ((df["high"] - df["open"]) / df["open"]) * 100

    # Momentum 5d
    df["momentum_5d"] = df["change_pct"].rolling(5, min_periods=1).sum()

    return df[["date", "name", "change_pct", "turnover_rate",
               "lead_stock_pct", "momentum_5d", "volume"]].copy()


def build_panel(start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch all 90 boards and build panel DataFrame."""
    print(f"\n{'='*60}")
    print(f"Fetching data: {start_date} ~ {end_date}")
    print(f"{'='*60}")

    all_dfs = []
    for i, board in enumerate(BOARDS):
        print(f"  [{i+1}/{len(BOARDS)}] {board}...", end="", flush=True)
        df = fetch_board_data(board, start_date, end_date)
        if not df.empty:
            all_dfs.append(df)
            print(f" {len(df)} days")
        else:
            print(" SKIP")
        time.sleep(0.35)  # Rate limiting

    if not all_dfs:
        raise RuntimeError("No data fetched!")

    panel = pd.concat(all_dfs, ignore_index=True)

    # Compute rise_count / fall_count per board per day
    # For each board, count how many boards in the same sector group went up/down
    print("\nComputing rise_count / fall_count...")
    daily_changes = panel.pivot_table(
        index="date", columns="name", values="change_pct",
    )

    rise_counts = {}
    fall_counts = {}

    for date in daily_changes.index:
        row = daily_changes.loc[date]
        for board in row.index:
            if pd.isna(row[board]):
                continue
            group = BOARD_TO_GROUP.get(board, "other")
            group_boards = SECTOR_GROUPS.get(group, [board])
            group_vals = row[row.index.isin(group_boards)].dropna()
            rc = int((group_vals > 0).sum())
            fc = int((group_vals <= 0).sum())
            rise_counts[(date, board)] = rc
            fall_counts[(date, board)] = fc

    panel["rise_count"] = panel.apply(
        lambda r: rise_counts.get((r["date"], r["name"]), 5), axis=1,
    )
    panel["fall_count"] = panel.apply(
        lambda r: fall_counts.get((r["date"], r["name"]), 5), axis=1,
    )

    # Select final columns
    panel = panel[["date", "name", "change_pct", "turnover_rate",
                   "rise_count", "fall_count", "lead_stock_pct",
                   "momentum_5d"]].copy()

    # Sort
    panel = panel.sort_values(["date", "name"]).reset_index(drop=True)

    dates = sorted(panel["date"].unique())
    boards = panel["name"].nunique()
    print(f"\nPanel: {len(dates)} days, {boards} boards, "
          f"{len(panel)} rows")
    print(f"Range: {dates[0]} ~ {dates[-1]}")

    return panel


def main():
    # Today is 2026-04-16, fetch from 2020-01 to cover 5+ years + warmup
    # We need:
    #   5Y test:  2021-04-16 ~ 2026-04-16  (warmup: before 2021-04)
    #   3Y test:  2023-04-16 ~ 2026-04-16  (warmup: before 2023-04)
    #   3M test:  2026-01-16 ~ 2026-04-16  (warmup: before 2026-01)
    #   1M test:  2026-03-16 ~ 2026-04-16  (warmup: before 2026-03)

    # Fetch ALL data from 2020-01 to 2026-04
    full_panel = build_panel("20200101", "20260416")

    dates = sorted(full_panel["date"].unique())
    print(f"\n=== Full dataset: {len(dates)} days ===")
    print(f"Range: {dates[0]} ~ {dates[-1]}")

    # Define test periods and warmup cutoffs
    today = pd.Timestamp("2026-04-16")
    periods = {
        "5y": {
            "test_start": today - pd.DateOffset(years=5),
            "test_end": today,
            "desc": "5年",
        },
        "3y": {
            "test_start": today - pd.DateOffset(years=3),
            "test_end": today,
            "desc": "3年",
        },
        "3m": {
            "test_start": today - pd.DateOffset(months=3),
            "test_end": today,
            "desc": "3个月",
        },
        "1m": {
            "test_start": today - pd.DateOffset(months=1),
            "test_end": today,
            "desc": "1个月",
        },
    }

    for key, p in periods.items():
        test_start = p["test_start"]
        test_end = p["test_end"]

        # Test panel: dates within [test_start, test_end]
        test_mask = (full_panel["date"] >= test_start) & (full_panel["date"] <= test_end)
        test_panel = full_panel[test_mask].copy()

        # Warmup panel: all data before test_start
        warmup_mask = full_panel["date"] < test_start
        warmup_panel = full_panel[warmup_mask].copy()

        test_dates = sorted(test_panel["date"].unique())
        warmup_dates = sorted(warmup_panel["date"].unique()) if len(warmup_panel) > 0 else []

        print(f"\n--- {p['desc']} ({key}) ---")
        print(f"  Test:   {len(test_dates)} days "
              f"({test_dates[0] if test_dates else 'N/A'} ~ "
              f"{test_dates[-1] if test_dates else 'N/A'})")
        print(f"  Warmup: {len(warmup_dates)} days "
              f"({warmup_dates[0] if warmup_dates else 'N/A'} ~ "
              f"{warmup_dates[-1] if warmup_dates else 'N/A'})")

        test_path = f"data/panel_{key}.parquet"
        warmup_path = f"data/panel_{key}_warmup.parquet"

        test_panel.to_parquet(test_path, index=False)
        print(f"  Saved: {test_path} ({len(test_panel)} rows)")

        if len(warmup_panel) > 0:
            warmup_panel.to_parquet(warmup_path, index=False)
            print(f"  Saved: {warmup_path} ({len(warmup_panel)} rows)")
        else:
            print(f"  [warn] No warmup data for {key}!")

    # Also save full panel
    full_panel.to_parquet("data/panel_full.parquet", index=False)
    print(f"\nFull panel saved: data/panel_full.parquet ({len(full_panel)} rows)")

    print("\n" + "="*60)
    print("DONE! Data ready for multi-scale backtesting.")
    print("="*60)


if __name__ == "__main__":
    main()
