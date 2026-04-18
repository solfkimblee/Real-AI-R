"""全策略 365 天专业回测 — V1 ~ V10 统一对比。

用法:
    python scripts/backtest_all_365d.py

输出:
    - 终端打印全策略对比表
    - data/report_365d.md  Markdown 报告
    - data/results_365d.json  原始数据
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# 允许脚本直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _strategy_runners import build_runners  # noqa: E402
from wf_backtest import (  # noqa: E402
    WFResult,
    _compute_tech_cycle,
    _metrics,
    load_panel,
    run_wf,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
REPORT_PATH = DATA_DIR / "report_365d.md"
JSON_PATH = DATA_DIR / "results_365d.json"


# ======================================================================
# 数据准备
# ======================================================================


def prepare_365d_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """合并 train/val/test 面板，截取最近 365 日历天数据。

    Returns:
        (panel_365d, warmup_panel)
    """
    dfs = []
    for name in ("panel_train.parquet", "panel_val.parquet", "panel_test.parquet"):
        p = DATA_DIR / name
        if p.exists():
            dfs.append(pd.read_parquet(p))

    if not dfs:
        raise FileNotFoundError("No panel_*.parquet files found in data/")

    combined = pd.concat(dfs, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"])
    combined = (
        combined.sort_values(["date", "name"])
        .drop_duplicates(subset=["date", "name"])
        .reset_index(drop=True)
    )

    # 365 日历天 cutoff
    max_date = combined["date"].max()
    cutoff = max_date - pd.Timedelta(days=365)
    panel_365 = combined[combined["date"] >= cutoff].reset_index(drop=True)
    warmup = combined[combined["date"] < cutoff].reset_index(drop=True)

    return panel_365, warmup


# ======================================================================
# 回测执行
# ======================================================================


def run_all_strategies(
    panel: pd.DataFrame,
    warmup: pd.DataFrame,
    n_windows: int = 12,
    window_size: int = 20,
    top_n: int = 10,
) -> list[WFResult]:
    """对全部策略跑 WF 回测。"""
    all_strategies = [
        "V1", "V1.1", "V2", "V5", "V6", "V7", "V8",
        "V9", "V9.2", "V9.3", "V10",
    ]
    runners = build_runners(strategies=all_strategies, warmup_panel=warmup)
    print(f"\n已加载 {len(runners)} 个策略: {[r.name for r in runners]}")

    benchmark = pd.DataFrame()  # 用全板块等权作为基准
    results: list[WFResult] = []

    for runner in runners:
        t0 = time.time()
        print(f"\n{'='*60}")
        print(f"  回测: {runner.name}")
        print(f"{'='*60}")

        try:
            res = run_wf(
                runner=runner,
                panel=panel,
                benchmark=benchmark,
                top_n=top_n,
                n_windows=n_windows,
                window_size=window_size,
                verbose=True,
            )
            elapsed = time.time() - t0
            print(
                f"  完成: WF夏普={res.sharpe:.2f}, "
                f"累计超额={res.cum_excess:.2f}%, "
                f"耗时={elapsed:.1f}s"
            )
            results.append(res)
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  失败: {e} (耗时={elapsed:.1f}s)")
            # 记录失败的策略
            failed = WFResult(strategy=runner.name, total_days=0)
            results.append(failed)

    return results


# ======================================================================
# 报告生成
# ======================================================================


def generate_report(
    results: list[WFResult],
    panel: pd.DataFrame,
    warmup: pd.DataFrame,
    n_windows: int,
    window_size: int,
) -> str:
    """生成 Markdown 专业回测报告。"""
    date_min = panel["date"].min().strftime("%Y-%m-%d")
    date_max = panel["date"].max().strftime("%Y-%m-%d")
    n_days = panel["date"].nunique()
    n_boards = panel["name"].nunique()
    warmup_days = warmup["date"].nunique() if len(warmup) > 0 else 0

    lines = [
        "# 泽平宏观全策略 365 天回测报告",
        "",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 一、回测参数",
        "",
        "| 参数 | 值 |",
        "|------|-----|",
        f"| 回测区间 | {date_min} ~ {date_max} |",
        f"| 交易日数 | {n_days} |",
        f"| 板块数 | {n_boards} |",
        f"| WF窗口数 | {n_windows} |",
        f"| 窗口大小 | {window_size} 天 |",
        f"| V9.3 Warmup | {warmup_days} 天 |",
        f"| 持仓数 | Top 10 等权 |",
        "| 基准 | 全板块等权平均 |",
        "| 交易成本 | 未扣除（纯信号对比） |",
        "",
        "## 二、全策略排名总表",
        "",
    ]

    # Sort by Sharpe descending
    sorted_results = sorted(results, key=lambda r: r.sharpe, reverse=True)

    lines.append(
        "| 排名 | 策略 | WF夏普 | 信息比率 | 胜率 | 累计超额 | "
        "年化超额 | 最大回撤 | 波动率 | 日均持仓 |"
    )
    lines.append(
        "|------|------|--------|---------|------|---------|"
        "---------|---------|--------|---------|"
    )

    for rank, r in enumerate(sorted_results, 1):
        if r.total_days == 0:
            lines.append(
                f"| {rank} | **{r.strategy}** | — | — | — | — | — | — | — | 运行失败 |"
            )
            continue
        lines.append(
            f"| {rank} | **{r.strategy}** | "
            f"{r.sharpe:+.2f} | "
            f"{r.info_ratio:+.2f} | "
            f"{r.win_rate:.1%} | "
            f"{r.cum_excess:+.2f}% | "
            f"{r.annual_excess:+.1%} | "
            f"{r.max_drawdown:.2%} | "
            f"{r.volatility:.2%} | "
            f"{r.avg_positions:.1f} |"
        )

    lines.append("")

    # Window-by-window comparison for top strategies
    lines.append("## 三、逐窗口夏普对比（前5策略）")
    lines.append("")

    top5 = [r for r in sorted_results if r.total_days > 0][:5]
    if top5 and top5[0].windows:
        header = "| 窗口 | 日期范围 |"
        sep = "|------|---------|"
        for r in top5:
            header += f" {r.strategy} |"
            sep += "--------|"
        lines.append(header)
        lines.append(sep)

        n_win = min(len(r.windows) for r in top5) if top5 else 0
        for wi in range(n_win):
            w0 = top5[0].windows[wi]
            start = w0.get("start", "")
            end = w0.get("end", "")
            date_range = f"{start}~{end}" if start and end else f"W{wi+1}"
            row = f"| W{wi+1} | {date_range} |"
            for r in top5:
                w = r.windows[wi] if wi < len(r.windows) else {}
                sharpe = w.get("sharpe", 0.0)
                row += f" {sharpe:+.2f} |"
            lines.append(row)

    lines.append("")

    # Positive windows count
    lines.append("## 四、稳定性分析")
    lines.append("")
    lines.append("| 策略 | 正夏普窗口 | 夏普CV | 最差窗口 | 最佳窗口 | 夏普极差 |")
    lines.append("|------|-----------|--------|---------|---------|---------|")

    for r in sorted_results:
        if r.total_days == 0 or not r.windows:
            continue
        w_sharpes = [w.get("sharpe", 0.0) for w in r.windows]
        positive = sum(1 for s in w_sharpes if s > 0)
        total_w = len(w_sharpes)
        mean_s = np.mean(w_sharpes) if w_sharpes else 0
        std_s = np.std(w_sharpes) if w_sharpes else 0
        cv = abs(std_s / mean_s) if abs(mean_s) > 1e-9 else float("inf")
        worst = min(w_sharpes) if w_sharpes else 0
        best = max(w_sharpes) if w_sharpes else 0
        lines.append(
            f"| {r.strategy} | {positive}/{total_w} | "
            f"{cv:.2f} | {worst:+.2f} | {best:+.2f} | {best-worst:.2f} |"
        )

    lines.append("")

    # Key insights
    lines.append("## 五、关键发现")
    lines.append("")

    if sorted_results and sorted_results[0].total_days > 0:
        best = sorted_results[0]
        lines.append(f"1. **最佳策略: {best.strategy}** — WF夏普 {best.sharpe:+.2f}, "
                      f"累计超额 {best.cum_excess:+.2f}%")

    # Find most stable (lowest CV)
    stable_candidates = [
        r for r in sorted_results
        if r.total_days > 0 and r.windows and r.sharpe > 0
    ]
    if stable_candidates:
        def _cv(r):
            ws = [w.get("sharpe", 0.0) for w in r.windows]
            m = np.mean(ws)
            s = np.std(ws)
            return abs(s / m) if abs(m) > 1e-9 else float("inf")

        most_stable = min(stable_candidates, key=_cv)
        lines.append(
            f"2. **最稳定策略: {most_stable.strategy}** — "
            f"夏普CV {_cv(most_stable):.2f}"
        )

    # Strategies with negative Sharpe
    negative = [r for r in results if r.sharpe < 0 and r.total_days > 0]
    if negative:
        neg_names = ", ".join(r.strategy for r in negative)
        lines.append(f"3. **负夏普策略: {neg_names}** — 不建议使用")

    lines.append("")
    lines.append("---")
    lines.append(f"*回测工具: `scripts/backtest_all_365d.py` | "
                  f"数据: THS 90 板块 {date_min}~{date_max}*")

    return "\n".join(lines)


# ======================================================================
# 主函数
# ======================================================================


def main():
    print("=" * 70)
    print("  泽平宏观全策略 365 天专业回测")
    print("=" * 70)

    # 准备数据
    print("\n[1/3] 准备数据...")
    panel, warmup = prepare_365d_panel()
    n_days = panel["date"].nunique()
    print(f"  365天面板: {panel['date'].min().date()} ~ {panel['date'].max().date()}")
    print(f"  交易日: {n_days}, 板块: {panel['name'].nunique()}")
    print(f"  Warmup: {warmup['date'].nunique()} 天")

    # WF参数: 尽量多切窗口
    window_size = 20
    n_windows = n_days // window_size
    if n_windows < 1:
        n_windows = 1
    print(f"  WF窗口: {n_windows} × {window_size}天 = {n_windows * window_size}天")

    # 跑回测
    print("\n[2/3] 跑全策略回测...")
    t0 = time.time()
    results = run_all_strategies(
        panel=panel,
        warmup=warmup,
        n_windows=n_windows,
        window_size=window_size,
    )
    total_time = time.time() - t0
    print(f"\n总耗时: {total_time:.1f}s")

    # 终端打印排名
    print("\n" + "=" * 70)
    print("  全策略排名")
    print("=" * 70)
    sorted_results = sorted(results, key=lambda r: r.sharpe, reverse=True)
    print(f"\n{'排名':>4} {'策略':<8} {'WF夏普':>8} {'IR':>8} "
          f"{'胜率':>6} {'累计超额':>10} {'最大回撤':>8}")
    print("-" * 70)
    for rank, r in enumerate(sorted_results, 1):
        if r.total_days == 0:
            print(f"{rank:>4} {r.strategy:<8} {'失败':>8}")
            continue
        print(
            f"{rank:>4} {r.strategy:<8} {r.sharpe:>+8.2f} {r.info_ratio:>+8.2f} "
            f"{r.win_rate:>6.1%} {r.cum_excess:>+10.2f}% {r.max_drawdown:>8.2%}"
        )

    # 生成报告
    print("\n[3/3] 生成报告...")
    report = generate_report(results, panel, warmup, n_windows, window_size)
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"  Markdown报告: {REPORT_PATH}")

    # 保存 JSON
    json_data = {
        "meta": {
            "date_range": f"{panel['date'].min().date()} ~ {panel['date'].max().date()}",
            "trading_days": n_days,
            "boards": int(panel["name"].nunique()),
            "n_windows": n_windows,
            "window_size": window_size,
            "total_time_sec": round(total_time, 1),
            "generated_at": datetime.now().isoformat(),
        },
        "results": [r.to_dict() for r in sorted_results],
    }
    JSON_PATH.write_text(json.dumps(json_data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  JSON数据: {JSON_PATH}")

    print("\n完成!")


if __name__ == "__main__":
    main()
