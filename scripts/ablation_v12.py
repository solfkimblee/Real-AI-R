"""V12消融实验 — 分别测试每个改进的独立贡献。

变体:
  V11         = baseline (固定训练, 单horizon, 20个纯价量特征)
  V12a(Retrain) = V11 + 滚动再训练（每20天retrain, 无多horizon, 无联动特征）
  V12b(MultiH)  = V11 + 多horizon融合（1d/3d/5d, 无retrain, 无联动特征）
  V12c(Linkage) = V11 + 泽平联动特征（无retrain, 无多horizon）
  V12(LGB+)     = V11 + 全部三项改进

用法:
    python scripts/ablation_v12.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _strategy_runners import build_runners  # noqa: E402

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PANEL_3Y = DATA_DIR / "panel_3y.parquet"
PANEL_TRAIN = DATA_DIR / "panel_train.parquet"

N_WINDOWS = 36
WINDOW_SIZE = 20
TOP_N = 10

STRATEGIES = ["V11", "V12a", "V12b", "V12c", "V12"]


def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "name"]).reset_index(drop=True)


def run_wf_backtest(
    panel: pd.DataFrame,
    warmup: pd.DataFrame,
    strategies: list[str],
    n_windows: int,
    window_size: int,
    top_n: int,
) -> dict[str, dict]:
    """运行WF回测，返回 {strategy_name: {windows, summary}}。"""
    dates = sorted(panel["date"].unique())
    total_days = len(dates)
    needed = n_windows * window_size
    if total_days < needed:
        print(f"[warn] 数据天数({total_days}) < 需求({needed}), "
              f"调整窗口数为 {total_days // window_size}")
        n_windows = total_days // window_size

    start_idx = total_days - n_windows * window_size
    eval_dates = dates[start_idx:]

    # 构建窗口
    windows = []
    for w in range(n_windows):
        w_dates = eval_dates[w * window_size: (w + 1) * window_size]
        windows.append(w_dates)

    print(f"[ablation] {n_windows}窗口 × {window_size}天 = "
          f"{n_windows * window_size}天, 起始 {eval_dates[0].date()}")

    # 构建 runners
    runners = build_runners(strategies=strategies, warmup_panel=warmup)
    print(f"[ablation] 成功构建 {len(runners)} 个策略: "
          f"{[r.name for r in runners]}")

    # 运行回测
    results: dict[str, dict] = {}
    for runner in runners:
        print(f"\n{'='*60}")
        print(f"  策略: {runner.name}")
        print(f"{'='*60}")

        all_excess: list[float] = []
        window_metrics: list[dict] = []

        for wi, w_dates in enumerate(windows):
            w_excess: list[float] = []

            for d in w_dates:
                day_data = panel[panel["date"] == d].copy()
                if day_data.empty:
                    continue

                # 计算 tech/cycle history
                day_idx = list(dates).index(d)
                hist_start = max(0, day_idx - 20)
                hist_dates = dates[hist_start:day_idx]

                tech_hist: list[float] = []
                cycle_hist: list[float] = []
                for hd in hist_dates:
                    hd_data = panel[panel["date"] == hd]
                    if not hd_data.empty:
                        avg = hd_data["change_pct"].mean()
                        tech_hist.append(avg)
                        cycle_hist.append(avg)

                # 预测
                picks = runner.predict(
                    day_data, top_n, tech_hist, cycle_hist,
                )

                # 计算超额
                day_avg = day_data["change_pct"].mean()
                if picks:
                    pick_returns = day_data[
                        day_data["name"].isin(picks)
                    ]["change_pct"]
                    portfolio_ret = pick_returns.mean() if len(pick_returns) > 0 else day_avg
                else:
                    portfolio_ret = day_avg

                daily_excess = portfolio_ret - day_avg
                w_excess.append(daily_excess)
                all_excess.append(daily_excess)

                # 通知策略
                realized = dict(
                    zip(day_data["name"], day_data["change_pct"])
                )
                excess_map = {
                    name: ret - day_avg
                    for name, ret in realized.items()
                }
                runner.record_day(realized, excess_map, daily_excess)

            # 窗口统计
            if w_excess:
                w_arr = np.array(w_excess)
                w_sharpe = (
                    float(np.mean(w_arr) / np.std(w_arr) * np.sqrt(252))
                    if np.std(w_arr) > 1e-9 else 0.0
                )
                w_ir = (
                    float(np.mean(w_arr) / np.std(w_arr) * np.sqrt(252))
                    if np.std(w_arr) > 1e-9 else 0.0
                )
                w_cum = float(np.sum(w_arr)) * 100
                w_wr = float(np.mean(w_arr > 0)) * 100
            else:
                w_sharpe = w_ir = w_cum = w_wr = 0.0

            wm = {
                "window": wi + 1,
                "start": str(w_dates[0].date()),
                "end": str(w_dates[-1].date()),
                "sharpe": round(w_sharpe, 2),
                "ir": round(w_ir, 2),
                "cum_excess_pct": round(w_cum, 2),
                "win_rate_pct": round(w_wr, 1),
                "n_days": len(w_excess),
            }
            window_metrics.append(wm)
            print(f"  W{wi+1:02d} [{wm['start']}~{wm['end']}] "
                  f"Sharpe={wm['sharpe']:+.2f} IR={wm['ir']:+.2f} "
                  f"Cum={wm['cum_excess_pct']:+.2f}% WR={wm['win_rate_pct']:.0f}%")

        # 全期统计
        arr = np.array(all_excess)
        if len(arr) > 0 and np.std(arr) > 1e-9:
            total_sharpe = float(np.mean(arr) / np.std(arr) * np.sqrt(252))
            total_ir = total_sharpe
        else:
            total_sharpe = total_ir = 0.0

        cum_excess = float(np.sum(arr)) * 100
        win_rate = float(np.mean(arr > 0)) * 100

        # MaxDD
        cum_arr = np.cumsum(arr)
        running_max = np.maximum.accumulate(cum_arr)
        drawdown = cum_arr - running_max
        max_dd = float(np.min(drawdown)) * 100

        # 正Sharpe/IR窗口数
        pos_sharpe = sum(1 for w in window_metrics if w["sharpe"] > 0)
        pos_ir = sum(1 for w in window_metrics if w["ir"] > 0)

        # Sharpe CV
        sharpe_vals = [w["sharpe"] for w in window_metrics]
        sharpe_cv = (
            float(np.std(sharpe_vals) / abs(np.mean(sharpe_vals)))
            if abs(np.mean(sharpe_vals)) > 1e-9 else float("inf")
        )

        summary = {
            "wf_sharpe": round(total_sharpe, 2),
            "ir": round(total_ir, 2),
            "cum_excess_pct": round(cum_excess, 2),
            "win_rate_pct": round(win_rate, 1),
            "max_dd_pct": round(max_dd, 2),
            "pos_sharpe_windows": f"{pos_sharpe}/{n_windows}",
            "pos_ir_windows": f"{pos_ir}/{n_windows}",
            "sharpe_cv": round(sharpe_cv, 2),
            "n_days": len(arr),
        }

        print(f"\n  --- {runner.name} 总结 ---")
        for k, v in summary.items():
            print(f"  {k}: {v}")

        results[runner.name] = {
            "windows": window_metrics,
            "summary": summary,
        }

    return results


def generate_report(results: dict[str, dict], n_windows: int) -> str:
    """生成消融实验 Markdown 报告。"""
    lines = [
        "# V12 消融实验报告",
        "",
        "## 实验设计",
        "",
        "| 变体 | 滚动再训练 | 多Horizon | 联动特征 |",
        "|------|:---------:|:--------:|:-------:|",
        "| V11 (baseline) | ✗ | ✗ | ✗ |",
        "| V12a(Retrain) | ✓ | ✗ | ✗ |",
        "| V12b(MultiH) | ✗ | ✓ | ✗ |",
        "| V12c(Linkage) | ✗ | ✗ | ✓ |",
        "| V12(LGB+) | ✓ | ✓ | ✓ |",
        "",
        f"## 总览（{n_windows}×20天WF窗口）",
        "",
        "| 排名 | 策略 | WF夏普 | IR | 累计超额 | 胜率 | MaxDD | 正IR窗口 | 夏普CV |",
        "|------|------|--------|-----|---------|------|-------|---------|--------|",
    ]

    # 按 WF Sharpe 降序排名
    sorted_names = sorted(
        results.keys(),
        key=lambda n: results[n]["summary"]["wf_sharpe"],
        reverse=True,
    )

    for rank, name in enumerate(sorted_names, 1):
        s = results[name]["summary"]
        lines.append(
            f"| {rank} | {name} | {s['wf_sharpe']:+.2f} | {s['ir']:+.2f} | "
            f"{s['cum_excess_pct']:+.2f}% | {s['win_rate_pct']:.1f}% | "
            f"{s['max_dd_pct']:.2f}% | {s['pos_ir_windows']} | {s['sharpe_cv']:.2f} |"
        )

    # 逐窗口对比表
    lines.extend([
        "",
        "## 逐窗口夏普对比",
        "",
    ])

    header = "| 窗口 | 日期 |"
    sep = "|------|------|"
    for name in sorted_names:
        header += f" {name} |"
        sep += "--------|"
    lines.append(header)
    lines.append(sep)

    # 取第一个策略的窗口信息作为模板
    first_name = sorted_names[0]
    for wi in range(len(results[first_name]["windows"])):
        w = results[first_name]["windows"][wi]
        row = f"| W{wi+1} | {w['start']}~{w['end']} |"
        for name in sorted_names:
            ws = results[name]["windows"][wi]["sharpe"]
            row += f" {ws:+.2f} |"
        lines.append(row)

    # 逐窗口IR对比
    lines.extend([
        "",
        "## 逐窗口IR对比",
        "",
    ])

    header = "| 窗口 | 日期 |"
    sep = "|------|------|"
    for name in sorted_names:
        header += f" {name} |"
        sep += "--------|"
    lines.append(header)
    lines.append(sep)

    for wi in range(len(results[first_name]["windows"])):
        w = results[first_name]["windows"][wi]
        row = f"| W{wi+1} | {w['start']}~{w['end']} |"
        for name in sorted_names:
            ws = results[name]["windows"][wi]["ir"]
            row += f" {ws:+.2f} |"
        lines.append(row)

    # 分析
    lines.extend([
        "",
        "## 消融分析",
        "",
        "### 各改进独立贡献（vs V11 baseline）",
        "",
    ])

    v11_sharpe = results.get("V11(LGB)", {}).get("summary", {}).get("wf_sharpe", 0)
    for name in sorted_names:
        if name == "V11(LGB)":
            continue
        s = results[name]["summary"]
        delta = s["wf_sharpe"] - v11_sharpe
        pct = delta / abs(v11_sharpe) * 100 if abs(v11_sharpe) > 1e-9 else 0
        effect = "提升" if delta > 0 else "降低"
        lines.append(f"- **{name}**: 夏普 {s['wf_sharpe']:+.2f} "
                      f"(vs V11 {v11_sharpe:+.2f}, {effect} {abs(pct):.1f}%)")

    return "\n".join(lines) + "\n"


def main() -> None:
    print("[ablation] 加载数据...")
    panel = load_panel(PANEL_3Y)
    warmup = load_panel(PANEL_TRAIN)
    print(f"[ablation] 3年面板: {panel['date'].nunique()}天, "
          f"训练面板: {warmup['date'].nunique()}天")

    results = run_wf_backtest(
        panel=panel,
        warmup=warmup,
        strategies=STRATEGIES,
        n_windows=N_WINDOWS,
        window_size=WINDOW_SIZE,
        top_n=TOP_N,
    )

    # 保存结果
    report = generate_report(results, N_WINDOWS)
    report_path = DATA_DIR / "report_ablation_v12.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"\n[ablation] 报告保存: {report_path}")

    json_path = DATA_DIR / "results_ablation_v12.json"
    json_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[ablation] 详细结果保存: {json_path}")

    # 打印总览
    print("\n" + "=" * 80)
    print("  V12 消融实验总览")
    print("=" * 80)

    sorted_names = sorted(
        results.keys(),
        key=lambda n: results[n]["summary"]["wf_sharpe"],
        reverse=True,
    )

    print(f"\n{'策略':<20} {'夏普':>8} {'IR':>8} {'累计超额':>10} "
          f"{'胜率':>8} {'MaxDD':>8} {'正IR窗口':>10}")
    print("-" * 80)
    for name in sorted_names:
        s = results[name]["summary"]
        print(f"{name:<20} {s['wf_sharpe']:>+8.2f} {s['ir']:>+8.2f} "
              f"{s['cum_excess_pct']:>+9.2f}% {s['win_rate_pct']:>7.1f}% "
              f"{s['max_dd_pct']:>7.2f}% {s['pos_ir_windows']:>10}")


if __name__ == "__main__":
    main()
