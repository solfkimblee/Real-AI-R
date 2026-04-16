"""通用 Walk-Forward 回测 harness。

用法:
    # 单策略
    python scripts/wf_backtest.py --strategy V10 --panel data/panel_val.parquet

    # 多策略对比
    python scripts/wf_backtest.py --strategies V5,V8,V10 --panel data/panel_val.parquet

    # 带 V9.3 warmup
    python scripts/wf_backtest.py --strategies V8,V9.3,V10 \\
        --panel data/panel_val.parquet --warmup-panel data/panel_train.parquet

    # 带 WF 窗口切分（对齐用户既有协议）
    python scripts/wf_backtest.py --strategies V8,V10 --panel data/panel_val.parquet \\
        --windows 7 --window-size 20

    # 列名映射（若数据列名与默认不同）
    python scripts/wf_backtest.py --strategy V10 --panel data/panel.parquet \\
        --column-map "涨跌幅:change_pct,换手:turnover_rate,板块名:name"

输出:
    - 逐窗口 + 全期 Sharpe / IR / 累计超额 / 胜率 / 最大回撤
    - 逐策略并排对比
    - 可选: --report-md path  保存 Markdown 报告
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 允许脚本直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _strategy_runners import build_runners  # noqa: E402


# ======================================================================
# 数据加载
# ======================================================================


def load_panel(
    path: str, column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """加载面板数据 (parquet/csv)，标准化列名。"""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"panel file not found: {path}")
    if p.suffix in (".parquet", ".pq"):
        try:
            df = pd.read_parquet(p)
        except Exception as e:
            raise RuntimeError(
                f"failed to read parquet (need pyarrow/fastparquet): {e}",
            ) from e
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
    else:
        raise ValueError(f"unsupported file format: {p.suffix}")

    if column_map:
        df = df.rename(columns=column_map)

    for col in ("date", "name", "change_pct"):
        if col not in df.columns:
            raise ValueError(
                f"panel missing required column '{col}'. "
                f"Got: {list(df.columns)}. "
                f"Use --column-map to rename.",
            )

    df["date"] = pd.to_datetime(df["date"])
    df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce")
    df["name"] = df["name"].astype(str)
    df = df.dropna(subset=["change_pct"]).sort_values(["date", "name"])
    return df.reset_index(drop=True)


def load_benchmark(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"])
    df["change_pct"] = pd.to_numeric(df["change_pct"], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def parse_column_map(s: str | None) -> dict[str, str]:
    if not s:
        return {}
    out = {}
    for pair in s.split(","):
        if ":" not in pair:
            continue
        k, v = pair.split(":", 1)
        out[k.strip()] = v.strip()
    return out


# ======================================================================
# 工具: tech/cycle history 计算（跨策略通用）
# ======================================================================


TECH_KEYWORDS = (
    "半导体", "芯片", "算力", "光模块", "AI", "人工智能",
    "新能源", "锂", "光伏", "风电", "消费电子",
    "互联网", "传媒", "游戏",
)
CYCLE_KEYWORDS = (
    "煤炭", "有色", "化工", "钢铁", "化学",
    "银行", "证券", "保险", "信托", "石油",
)


def _compute_tech_cycle(
    today_df: pd.DataFrame,
) -> tuple[float, float]:
    names = today_df["name"].astype(str).values
    cp = pd.to_numeric(today_df["change_pct"], errors="coerce").fillna(0.0).values
    tech_mask = np.array(
        [any(kw in n for kw in TECH_KEYWORDS) for n in names],
    )
    cycle_mask = np.array(
        [any(kw in n for kw in CYCLE_KEYWORDS) for n in names],
    )
    tech_ret = float(cp[tech_mask].mean()) if tech_mask.any() else 0.0
    cycle_ret = float(cp[cycle_mask].mean()) if cycle_mask.any() else 0.0
    return tech_ret, cycle_ret


# ======================================================================
# 核心: 单策略 WF 回测
# ======================================================================


@dataclass
class WFResult:
    strategy: str
    total_days: int
    # 全期指标
    sharpe: float = 0.0
    info_ratio: float = 0.0
    annual_return: float = 0.0
    annual_excess: float = 0.0
    volatility: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    cum_excess: float = 0.0
    avg_positions: float = 0.0
    # 日序列
    daily_returns: list[float] = field(default_factory=list)
    daily_excess: list[float] = field(default_factory=list)
    dates: list[str] = field(default_factory=list)
    # 逐窗口指标
    windows: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "total_days": self.total_days,
            "sharpe": self.sharpe,
            "info_ratio": self.info_ratio,
            "annual_return": self.annual_return,
            "annual_excess": self.annual_excess,
            "volatility": self.volatility,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "cum_excess": self.cum_excess,
            "avg_positions": self.avg_positions,
            "windows": self.windows,
        }


def _metrics(
    daily_ret: np.ndarray,
    daily_excess: np.ndarray,
) -> dict[str, float]:
    if len(daily_ret) == 0:
        return {}
    mean_d = float(daily_ret.mean())
    std_d = float(daily_ret.std(ddof=0))
    mean_e = float(daily_excess.mean())
    std_e = float(daily_excess.std(ddof=0))

    sharpe = (mean_d * 252) / (std_d * np.sqrt(252) + 1e-9)
    ir = (mean_e * 252) / (std_e * np.sqrt(252) + 1e-9)
    ann_r = (1 + mean_d / 100.0) ** 252 - 1 if mean_d > -100 else -1.0
    ann_e = (1 + mean_e / 100.0) ** 252 - 1 if mean_e > -100 else -1.0
    vol = std_d * np.sqrt(252) / 100.0

    equity = np.cumprod(1 + daily_ret / 100.0)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0

    return {
        "sharpe": sharpe,
        "info_ratio": ir,
        "annual_return": ann_r,
        "annual_excess": ann_e,
        "volatility": vol,
        "max_drawdown": max_dd,
        "win_rate": float((daily_ret > 0).mean()),
        "cum_excess": float(daily_excess.sum()),
    }


def run_wf(
    runner,
    panel: pd.DataFrame,
    benchmark: pd.DataFrame,
    top_n: int = 10,
    n_windows: int | None = None,
    window_size: int | None = None,
    verbose: bool = False,
) -> WFResult:
    """单策略 WF 回测。

    n_windows + window_size: 若提供则切分为 n 个 window_size 天的块做逐窗口统计。
    否则只输出全期指标。
    """
    unique_dates = sorted(panel["date"].unique())
    if len(unique_dates) < 2:
        raise ValueError("need at least 2 dates")

    # 基准映射
    bench_map: dict[pd.Timestamp, float] = {}
    if len(benchmark) > 0:
        for _, row in benchmark.iterrows():
            bench_map[pd.Timestamp(row["date"])] = float(row["change_pct"])

    tech_hist: list[float] = []
    cycle_hist: list[float] = []
    daily_ret: list[float] = []
    daily_excess: list[float] = []
    positions_count: list[int] = []
    kept_dates: list[str] = []

    for i in range(len(unique_dates) - 1):
        d = unique_dates[i]
        today = panel[panel["date"] == d].reset_index(drop=True)
        tomorrow_d = unique_dates[i + 1]
        tomorrow = panel[panel["date"] == tomorrow_d]
        if today.empty or tomorrow.empty:
            continue

        t_ret, c_ret = _compute_tech_cycle(today)
        tech_hist.append(t_ret)
        cycle_hist.append(c_ret)

        try:
            picks = runner.predict(
                board_df=today,
                top_n=top_n,
                tech_history=list(tech_hist),
                cycle_history=list(cycle_hist),
            )
        except Exception as e:
            if verbose:
                print(f"[{runner.name}] {d.date()} predict failed: {e}")
            picks = []

        realized = dict(
            zip(
                tomorrow["name"].astype(str).values,
                pd.to_numeric(tomorrow["change_pct"], errors="coerce")
                .fillna(0.0).values,
                strict=False,
            ),
        )
        if picks:
            port_ret = float(np.mean([realized.get(b, 0.0) for b in picks]))
        else:
            port_ret = 0.0

        # 基准
        if bench_map and tomorrow_d in bench_map:
            bench_ret = bench_map[tomorrow_d]
        else:
            bench_ret = float(
                pd.to_numeric(tomorrow["change_pct"], errors="coerce")
                .fillna(0.0).mean(),
            )

        excess = port_ret - bench_ret
        excess_per_board = {
            b: realized.get(b, 0.0) - bench_ret for b in picks
        }

        try:
            runner.record_day(realized, excess_per_board, excess)
        except Exception as e:
            if verbose:
                print(f"[{runner.name}] record_day failed: {e}")

        daily_ret.append(port_ret)
        daily_excess.append(excess)
        positions_count.append(len(picks))
        kept_dates.append(str(d.date()))

    arr_ret = np.array(daily_ret)
    arr_excess = np.array(daily_excess)
    full = _metrics(arr_ret, arr_excess)

    # 逐窗口统计
    windows_info: list[dict[str, Any]] = []
    if n_windows and window_size:
        total_needed = n_windows * window_size
        # 从尾部倒推，保证每窗口有完整 window_size 天
        start = max(0, len(arr_ret) - total_needed)
        for w in range(n_windows):
            s = start + w * window_size
            e = s + window_size
            if e > len(arr_ret):
                break
            wr = arr_ret[s:e]
            we = arr_excess[s:e]
            wm = _metrics(wr, we)
            windows_info.append(
                {
                    "window": w + 1,
                    "start": kept_dates[s] if s < len(kept_dates) else "",
                    "end": kept_dates[e - 1] if e - 1 < len(kept_dates) else "",
                    "days": int(e - s),
                    **wm,
                },
            )

    return WFResult(
        strategy=runner.name,
        total_days=len(arr_ret),
        sharpe=full.get("sharpe", 0.0),
        info_ratio=full.get("info_ratio", 0.0),
        annual_return=full.get("annual_return", 0.0),
        annual_excess=full.get("annual_excess", 0.0),
        volatility=full.get("volatility", 0.0),
        max_drawdown=full.get("max_drawdown", 0.0),
        win_rate=full.get("win_rate", 0.0),
        cum_excess=full.get("cum_excess", 0.0),
        avg_positions=float(np.mean(positions_count)) if positions_count else 0.0,
        daily_returns=daily_ret,
        daily_excess=daily_excess,
        dates=kept_dates,
        windows=windows_info,
    )


# ======================================================================
# 报告输出
# ======================================================================


def format_compare_table(results: list[WFResult]) -> str:
    header = (
        f"| {'策略':<6} | {'Sharpe':>7} | {'IR':>7} | {'年化超额':>9} | "
        f"{'累计超额':>9} | {'MaxDD':>8} | {'胜率':>6} | {'持仓':>5} | {'天数':>5} |"
    )
    sep = "|" + "|".join(["-" * (len(s) + 2) for s in header.split("|")[1:-1]]) + "|"
    lines = [header, sep]
    for r in results:
        lines.append(
            f"| {r.strategy:<6} | {r.sharpe:>7.2f} | {r.info_ratio:>7.2f} | "
            f"{r.annual_excess:>8.1%} | {r.cum_excess:>8.2f}% | "
            f"{r.max_drawdown:>7.1%} | {r.win_rate:>5.1%} | "
            f"{r.avg_positions:>5.1f} | {r.total_days:>5} |",
        )
    return "\n".join(lines)


def format_window_table(result: WFResult) -> str:
    if not result.windows:
        return "(no window breakdown)"
    header = (
        f"| W | 起 | 止 | 天数 | Sharpe | IR | 累计超额 |"
    )
    sep = "|---|---|---|------|--------|-----|----------|"
    lines = [f"### {result.strategy} 逐窗口", header, sep]
    for w in result.windows:
        lines.append(
            f"| W{w['window']} | {w['start']} | {w['end']} | {w['days']} | "
            f"{w['sharpe']:+.2f} | {w['info_ratio']:+.2f} | "
            f"{w['cum_excess']:+.2f}% |",
        )
    return "\n".join(lines)


def write_report_md(
    results: list[WFResult],
    panel_path: str,
    out_path: Path,
) -> None:
    n_strats = len(results)
    lines = [
        f"# WF 回测报告",
        "",
        f"- 面板数据: `{panel_path}`",
        f"- 策略数: {n_strats}",
        f"- 交易日数: {results[0].total_days if results else 0}",
        "",
        "## 全期对比",
        "",
        format_compare_table(results),
        "",
    ]
    for r in results:
        if r.windows:
            lines.append("")
            lines.append(format_window_table(r))
    out_path.write_text("\n".join(lines), encoding="utf-8")


# ======================================================================
# CLI
# ======================================================================


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Walk-Forward 回测 harness",
    )
    ap.add_argument(
        "--panel", required=True, help="parquet/csv 面板数据路径",
    )
    ap.add_argument(
        "--benchmark", default=None, help="可选基准 CSV 路径",
    )
    ap.add_argument(
        "--warmup-panel", default=None,
        help="V9.3 warmup 用历史面板 (parquet/csv)",
    )
    ap.add_argument(
        "--strategy", default=None,
        help="单策略: V5|V7|V8|V9.2|V9.3|V10",
    )
    ap.add_argument(
        "--strategies", default=None,
        help="多策略逗号分隔: V5,V8,V10",
    )
    ap.add_argument("--top-n", type=int, default=10)
    ap.add_argument("--windows", type=int, default=None)
    ap.add_argument("--window-size", type=int, default=None)
    ap.add_argument(
        "--column-map", default=None,
        help='列名映射: "涨跌幅:change_pct,换手:turnover_rate"',
    )
    ap.add_argument(
        "--report-md", default=None, help="输出 Markdown 报告路径",
    )
    ap.add_argument("--report-json", default=None)
    ap.add_argument("--verbose", action="store_true")

    args = ap.parse_args()

    col_map = parse_column_map(args.column_map)
    panel = load_panel(args.panel, column_map=col_map)
    benchmark = (
        load_benchmark(args.benchmark)
        if args.benchmark else pd.DataFrame()
    )
    warmup_panel: pd.DataFrame | None = None
    if args.warmup_panel:
        warmup_panel = load_panel(args.warmup_panel, column_map=col_map)

    if args.strategy and args.strategies:
        ap.error("use --strategy or --strategies, not both")
    if args.strategy:
        strategies = [args.strategy]
    elif args.strategies:
        strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    else:
        strategies = ["V5", "V7", "V8", "V9.2", "V9.3", "V10"]

    print(f"[info] panel: {args.panel}")
    print(f"[info] dates: {len(panel['date'].unique())} unique")
    print(f"[info] boards: {panel['name'].nunique()} unique")
    print(f"[info] strategies: {strategies}")
    if warmup_panel is not None:
        print(
            f"[info] warmup: {len(warmup_panel['date'].unique())} days",
        )

    runners = build_runners(
        strategies=strategies, warmup_panel=warmup_panel,
    )
    results: list[WFResult] = []
    for r in runners:
        print(f"\n[run] {r.name} ...")
        res = run_wf(
            r, panel, benchmark,
            top_n=args.top_n,
            n_windows=args.windows,
            window_size=args.window_size,
            verbose=args.verbose,
        )
        print(
            f"[done] {r.name}  Sharpe={res.sharpe:+.2f}  IR={res.info_ratio:+.2f}"
            f"  累计={res.cum_excess:+.2f}%  胜率={res.win_rate:.1%}"
            f"  MaxDD={res.max_drawdown:.1%}",
        )
        results.append(res)

    print("\n" + "=" * 70)
    print("对比表")
    print("=" * 70)
    print(format_compare_table(results))

    # 逐窗口
    if args.windows and args.window_size:
        for r in results:
            print()
            print(format_window_table(r))

    if args.report_md:
        write_report_md(results, args.panel, Path(args.report_md))
        print(f"\n[report] {args.report_md}")
    if args.report_json:
        Path(args.report_json).write_text(
            json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[json] {args.report_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
