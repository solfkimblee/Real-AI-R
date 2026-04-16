"""生成合成面板数据用于 wf_backtest.py 冒烟测试。

用法:
    python scripts/gen_synthetic_panel.py \\
        --out data/panel_synth.parquet --days 140 --boards 30

生成的 parquet 包含 V8/V10 所需的全部列，可直接:
    python scripts/wf_backtest.py --panel data/panel_synth.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REAL_BOARD_NAMES = [
    "半导体", "芯片", "算力", "光模块", "AI应用",
    "锂电池", "光伏", "新能源汽车", "锂矿", "正极材料",
    "消费电子", "显示器", "面板",
    "医药", "创新药", "CXO", "医疗器械",
    "军工", "航空", "船舶",
    "银行", "证券", "保险", "信托",
    "煤炭", "有色", "化工", "钢铁", "水泥",
    "食品饮料", "乳业", "日化",
    "互联网", "传媒", "游戏",
    "房地产", "白酒",  # redline
]


def gen(days: int, n_boards: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    names = REAL_BOARD_NAMES[:n_boards]
    n = len(names)

    is_growth = np.array([
        any(
            kw in nm for kw in (
                "半导体", "芯片", "算力", "光模块", "AI",
                "锂", "光伏", "新能源", "消费电子",
                "显示器", "面板", "传媒", "游戏", "互联网",
            )
        )
        for nm in names
    ])
    is_cycle = np.array([
        any(
            kw in nm for kw in (
                "煤炭", "有色", "化工", "钢铁", "水泥",
                "银行", "证券", "保险",
            )
        )
        for nm in names
    ])

    # 7 × 20 天 regime 切换（接近用户 WF 协议）
    schedule = ["neutral", "cycle", "tech", "neutral", "cycle", "tech", "neutral"]
    dates = pd.bdate_range("2024-01-02", periods=days)
    persist = rng.uniform(0.0, 0.3, n)
    x = np.zeros(n)

    rows = []
    for di, d in enumerate(dates):
        reg = schedule[min(di // 20, len(schedule) - 1)]
        if reg == "tech":
            drift = np.where(is_growth, 0.3, -0.1)
        elif reg == "cycle":
            drift = np.where(is_cycle, 0.3, np.where(is_growth, -0.1, 0.0))
        else:
            drift = np.zeros(n)
        x = persist * x + rng.normal(drift, 1.5)
        for i, nm in enumerate(names):
            rows.append(
                {
                    "date": d,
                    "name": nm,
                    "code": f"BK{i:04d}",
                    "change_pct": float(x[i]),
                    "turnover_rate": float(rng.uniform(0.8, 4.0)),
                    "rise_count": int(rng.integers(10, 45)),
                    "fall_count": int(rng.integers(5, 30)),
                    "lead_stock_pct": float(x[i]) + float(rng.normal(1.0, 1.5)),
                    "main_net_inflow": float(rng.normal(0, 20)),
                },
            )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/panel_synth.parquet")
    ap.add_argument("--days", type=int, default=140)
    ap.add_argument("--boards", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = gen(args.days, args.boards, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix in (".parquet", ".pq"):
        try:
            df.to_parquet(out, index=False)
        except Exception as e:
            print(f"[warn] parquet failed ({e}), falling back to csv")
            out = out.with_suffix(".csv")
            df.to_csv(out, index=False)
    else:
        df.to_csv(out, index=False)
    print(f"[ok] wrote {out}  rows={len(df)}  dates={df['date'].nunique()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
