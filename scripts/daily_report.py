"""Real-AI-R 每日市场日报生成器

自动获取 A 股实时数据，运行泽平宏观策略 + 周期轮动 + 科技赛道追踪，
输出一份结构化的每日投资参考报告。

当实时 API 不可用时（如沙箱环境），自动切换到模拟数据模式。

用法:
    python scripts/daily_report.py          # 自动检测（实时 → fallback）
    python scripts/daily_report.py --demo   # 强制模拟数据
"""

from __future__ import annotations

import random
import sys
from datetime import datetime
from pathlib import Path

# 确保 src 在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import pandas as pd


# ======================================================================
# 模拟数据生成（当实时 API 不可用时）
# ======================================================================

def _generate_mock_industry_boards() -> pd.DataFrame:
    """生成模拟行业板块数据。"""
    boards = [
        ("半导体", "BK1036"), ("人工智能", "BK1037"), ("机器人", "BK1038"),
        ("新能源汽车", "BK1039"), ("军工", "BK1040"), ("医药生物", "BK1041"),
        ("有色金属", "BK1042"), ("煤炭", "BK1043"), ("黄金", "BK1044"),
        ("农业", "BK1045"), ("食品饮料", "BK1046"), ("光伏", "BK1047"),
        ("通信", "BK1048"), ("消费电子", "BK1049"), ("存储芯片", "BK1050"),
        ("自动驾驶", "BK1051"), ("锂电池", "BK1052"), ("基本金属", "BK1053"),
        ("石油开采", "BK1054"), ("化肥", "BK1055"), ("乳业", "BK1056"),
        ("房地产", "BK1057"), ("白酒", "BK1058"), ("零售", "BK1059"),
        ("软件外包", "BK1060"), ("大数据", "BK1061"), ("云计算", "BK1062"),
        ("创新药", "BK1063"), ("商业航天", "BK1064"), ("充电桩", "BK1065"),
        ("风电", "BK1066"), ("稀土", "BK1067"), ("天然气", "BK1068"),
        ("种子", "BK1069"), ("日化", "BK1070"),
    ]

    np.random.seed(42)
    rows = []
    lead_stocks = [
        "北方华创", "科大讯飞", "汇川技术", "比亚迪", "中航沈飞",
        "恒瑞医药", "紫金矿业", "中国神华", "山东黄金", "隆平高科",
        "海天味业", "隆基绿能", "中兴通讯", "立讯精密", "兆易创新",
        "华为概念", "宁德时代", "江西铜业", "中国石油", "盐湖股份",
        "伊利股份", "万科A", "贵州茅台", "永辉超市", "中国软件",
        "浪潮信息", "金山云", "药明康德", "中国卫星", "特锐德",
        "金风科技", "北方稀土", "广汇能源", "荃银高科", "上海家化",
    ]

    for i, (name, code) in enumerate(boards):
        change = np.random.normal(0.5, 2.0)
        rise = np.random.randint(30, 180)
        fall = np.random.randint(20, 150)
        turnover = np.random.uniform(0.5, 5.0)
        lead_pct = change + np.random.uniform(1.0, 6.0)
        rows.append({
            "rank": i + 1,
            "name": name,
            "code": code,
            "price": round(np.random.uniform(800, 5000), 2),
            "change_amount": round(change * 10, 2),
            "change_pct": round(change, 2),
            "total_mv": round(np.random.uniform(1e11, 5e12), 0),
            "turnover_rate": round(turnover, 2),
            "rise_count": rise,
            "fall_count": fall,
            "lead_stock": lead_stocks[i],
            "lead_stock_pct": round(lead_pct, 2),
        })

    return pd.DataFrame(rows)


def _generate_mock_fund_flow() -> pd.DataFrame:
    """生成模拟资金流向数据。"""
    sectors = [
        "半导体", "人工智能", "机器人", "新能源汽车", "军工",
        "医药生物", "有色金属", "煤炭", "黄金", "农业",
        "食品饮料", "光伏", "通信", "消费电子", "锂电池",
        "房地产", "白酒", "基本金属", "石油开采", "创新药",
    ]
    np.random.seed(123)
    rows = []
    for i, name in enumerate(sectors):
        main_inflow = np.random.normal(0, 5) * 1e8
        rows.append({
            "名称": name,
            "主力净流入-净额": round(main_inflow, 0),
        })
    return pd.DataFrame(rows)


# ======================================================================
# 报告格式化工具
# ======================================================================

def section(title: str) -> str:
    line = "=" * 60
    return f"\n{line}\n  {title}\n{line}"


def sub_section(title: str) -> str:
    return f"\n--- {title} ---"


# ======================================================================
# 日报核心
# ======================================================================

def generate_daily_report(demo_mode: bool = False) -> str:
    """生成每日市场日报。

    Parameters
    ----------
    demo_mode : bool
        True 强制使用模拟数据；False 先尝试实时 API，失败后回退模拟。
    """
    today = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines: list[str] = []
    data_source = "实时数据"

    # ------------------------------------------------------------------
    # 尝试获取实时数据
    # ------------------------------------------------------------------
    industry_df = pd.DataFrame()
    fund_df = pd.DataFrame()

    if not demo_mode:
        try:
            from real_ai_r.sector.monitor import SectorMonitor
            industry_df = SectorMonitor.get_board_list("industry")
            fund_df = SectorMonitor.get_fund_flow("今日", "行业资金流")
        except Exception:
            pass

    if industry_df.empty:
        data_source = "模拟数据（实时 API 不可用）"
        industry_df = _generate_mock_industry_boards()
        fund_df = _generate_mock_fund_flow()

    lines.append(section(f"Real-AI-R A 股量化日报  {today}"))
    lines.append(f"  数据模式: {data_source}")

    # ==================================================================
    # 1. 市场概览
    # ==================================================================
    lines.append(section("一、市场概览"))

    rise_df = industry_df[industry_df["change_pct"] > 0]
    fall_df = industry_df[industry_df["change_pct"] < 0]
    flat_df = industry_df[industry_df["change_pct"] == 0]
    avg_change = industry_df["change_pct"].mean()

    lines.append(
        f"行业板块总数: {len(industry_df)}  |  "
        f"上涨: {len(rise_df)}  下跌: {len(fall_df)}  平盘: {len(flat_df)}"
    )
    lines.append(f"板块平均涨跌幅: {avg_change:+.2f}%")

    if avg_change > 0.5:
        mood = "偏多（板块普涨）"
    elif avg_change < -0.5:
        mood = "偏空（板块普跌）"
    else:
        mood = "震荡（涨跌参半）"
    lines.append(f"市场情绪: {mood}")

    top5 = industry_df.nlargest(5, "change_pct")
    bot5 = industry_df.nsmallest(5, "change_pct")

    lines.append(sub_section("涨幅前 5 行业板块"))
    for i, (_, b) in enumerate(top5.iterrows(), 1):
        lines.append(
            f"  {i}. {b['name']:<8s}  {b['change_pct']:+.2f}%  "
            f"领涨: {b['lead_stock']}({b['lead_stock_pct']:+.2f}%)"
        )

    lines.append(sub_section("跌幅前 5 行业板块"))
    for i, (_, b) in enumerate(bot5.iterrows(), 1):
        lines.append(
            f"  {i}. {b['name']:<8s}  {b['change_pct']:+.2f}%  "
            f"领涨: {b['lead_stock']}({b['lead_stock_pct']:+.2f}%)"
        )

    # ==================================================================
    # 2. 宏观分类概览
    # ==================================================================
    lines.append(section("二、宏观分类概览（泽平框架）"))

    from real_ai_r.macro.classifier import SectorClassifier

    classifier = SectorClassifier()
    classified = classifier.classify_dataframe(industry_df)
    summary = classifier.get_category_summary(classified)

    for cat_key, label in [
        ("tech", "🗡️  科技主线"),
        ("cycle", "🛡️  周期主线"),
        ("redline", "🚫 红线禁区"),
        ("neutral", "⚪ 其他"),
    ]:
        info = summary.get(cat_key, {"count": 0, "avg_change": 0})
        lines.append(
            f"  {label}: {info['count']} 个板块  "
            f"平均涨幅 {info['avg_change']:+.2f}%"
        )

    # 各分类板块明细
    for cat_key, label in [("tech", "科技"), ("cycle", "周期")]:
        boards_list = summary.get(cat_key, {}).get("boards", [])
        if boards_list:
            lines.append(f"  {label}板块: {', '.join(boards_list[:10])}")

    # ==================================================================
    # 3. 泽平宏观策略 Top10
    # ==================================================================
    lines.append(section("三、泽平宏观策略 Top 10 推荐"))

    from real_ai_r.macro.zeping_strategy import ZepingMacroStrategy

    strategy = ZepingMacroStrategy()
    result = strategy.predict(board_df=industry_df, top_n=10)

    lines.append(
        f"评估板块: {result.total_boards}  |  "
        f"过滤红线: {result.filtered_redline}"
    )
    lines.append(f"市场风格: {result.market_style}")

    for line in result.strategy_summary.split("\n"):
        if line.strip():
            lines.append(f"  {line.strip()}")

    lines.append("")
    lines.append(
        f"{'排名':<4} {'板块':<10} {'总分':>6} {'宏观':>6} "
        f"{'量化':>6} {'周期':>6} {'分类':<6} {'推荐理由'}"
    )
    lines.append("-" * 88)

    for i, s in enumerate(result.predictions, 1):
        cat_icon = {
            "tech": "科技", "cycle": "周期", "neutral": "其他",
        }.get(s.macro_category, s.macro_category)
        reasons = " | ".join(s.reasons[:3])
        lines.append(
            f"  {i:<3} {s.board_name:<10s} {s.total_score:>6.1f} "
            f"{s.macro_score:>6.1f} {s.quant_score:>6.1f} "
            f"{s.cycle_score:>6.1f} {cat_icon:<6s} {reasons}"
        )

    # ==================================================================
    # 4. 周期轮动（基于板块数据直接计算）
    # ==================================================================
    lines.append(section("四、大宗商品周期轮动（五段论）"))

    from real_ai_r.macro.classifier import CYCLE_STAGES

    stage_results = []
    for key, stage_info in CYCLE_STAGES.items():
        keywords = stage_info["keywords"]
        matched = classified[
            classified["name"].apply(
                lambda n: any(kw in str(n) for kw in keywords)
            )
        ]
        matched_names = matched["name"].tolist() if not matched.empty else []
        avg_chg = matched["change_pct"].mean() if not matched.empty else 0.0
        avg_turn = (
            matched["turnover_rate"].mean()
            if not matched.empty and "turnover_rate" in matched.columns
            else 0.0
        )

        # 温度计算
        momentum = max(0, min(100, 50 + avg_chg * 10))
        activity = min(avg_turn * 10, 100) if avg_turn else 0
        total_rise = int(matched["rise_count"].sum()) if not matched.empty else 0
        total_fall = int(matched["fall_count"].sum()) if not matched.empty else 0
        total = total_rise + total_fall
        breadth = (total_rise / total * 100) if total > 0 else 50

        temp = 0.35 * momentum + 0.30 * 50 + 0.20 * activity + 0.15 * breadth

        stage_results.append({
            "stage": stage_info["stage"],
            "icon": stage_info["icon"],
            "display": stage_info["display"],
            "status": stage_info["status"],
            "temperature": round(temp, 1),
            "avg_change": round(avg_chg, 2),
            "matched": matched_names,
        })

    hottest = max(stage_results, key=lambda s: s["temperature"])
    lines.append(
        f"当前最热阶段: 阶段{hottest['stage']} {hottest['icon']} "
        f"{hottest['display']} (温度 {hottest['temperature']:.0f}°)"
    )

    lines.append("")
    lines.append(
        f"{'阶段':<8} {'名称':<10} {'温度':>6} {'涨幅%':>8} "
        f"{'框架定位':<14} {'匹配板块'}"
    )
    lines.append("-" * 72)

    for s in sorted(stage_results, key=lambda x: x["stage"]):
        boards_str = ", ".join(s["matched"][:3]) if s["matched"] else "无匹配"
        lines.append(
            f"  阶段{s['stage']}  {s['icon']} {s['display']:<8s} "
            f"{s['temperature']:>5.0f}° {s['avg_change']:>+7.2f}%  "
            f"{s['status']:<14s} {boards_str}"
        )

    # ==================================================================
    # 5. 科技赛道追踪（基于板块数据直接计算）
    # ==================================================================
    lines.append(section("五、科技赛道热度追踪"))

    from real_ai_r.macro.classifier import TECH_TRACKS

    track_results = []
    for key, track_info in TECH_TRACKS.items():
        keywords = track_info["keywords"]
        matched = classified[
            classified["name"].apply(
                lambda n: any(kw in str(n) for kw in keywords)
            )
        ]
        if matched.empty:
            track_results.append({
                "icon": track_info["icon"],
                "display": track_info["display"],
                "heat": 0,
                "avg_change": 0,
                "rise": 0,
                "fall": 0,
                "lead_stock": "-",
                "lead_pct": 0,
            })
            continue

        avg_chg = matched["change_pct"].mean()
        avg_turn = matched["turnover_rate"].mean() if "turnover_rate" in matched.columns else 0
        total_rise = int(matched["rise_count"].sum()) if "rise_count" in matched.columns else 0
        total_fall = int(matched["fall_count"].sum()) if "fall_count" in matched.columns else 0
        total = total_rise + total_fall
        breadth = (total_rise / total * 100) if total > 0 else 50

        momentum = max(0, min(100, 50 + avg_chg * 10))
        activity = min(avg_turn * 10, 100)
        heat = round(0.40 * momentum + 0.30 * activity + 0.30 * breadth, 1)

        # 领涨
        if "lead_stock_pct" in matched.columns and not matched.empty:
            top_idx = matched["lead_stock_pct"].idxmax()
            lead_stock = str(matched.loc[top_idx, "lead_stock"])
            lead_pct = float(matched.loc[top_idx, "lead_stock_pct"])
        else:
            lead_stock, lead_pct = "-", 0.0

        track_results.append({
            "icon": track_info["icon"],
            "display": track_info["display"],
            "heat": heat,
            "avg_change": round(avg_chg, 2),
            "rise": total_rise,
            "fall": total_fall,
            "lead_stock": lead_stock,
            "lead_pct": round(lead_pct, 2),
        })

    track_results.sort(key=lambda t: t["heat"], reverse=True)

    lines.append("")
    lines.append(
        f"{'赛道':<16} {'热度':>6} {'涨幅%':>8} {'涨/跌':>8} "
        f"{'领涨股':<10} {'领涨%':>8}"
    )
    lines.append("-" * 70)

    for t in track_results:
        lines.append(
            f"  {t['icon']} {t['display']:<12s} {t['heat']:>5.0f}  "
            f"{t['avg_change']:>+7.2f}% {t['rise']:>4}/{t['fall']:<4} "
            f"{t['lead_stock']:<10s} {t['lead_pct']:>+7.2f}%"
        )

    # ==================================================================
    # 6. 资金流向
    # ==================================================================
    lines.append(section("六、行业资金流向 Top 10"))

    inflow_col = None
    name_col = None
    for col in fund_df.columns:
        if "主力净流入" in col:
            inflow_col = col
            break
    for col in ["名称", "name"]:
        if col in fund_df.columns:
            name_col = col
            break

    if inflow_col and name_col:
        fund_df[inflow_col] = pd.to_numeric(fund_df[inflow_col], errors="coerce")
        top_inflow = fund_df.nlargest(10, inflow_col)

        lines.append(sub_section("主力净流入前 10"))
        for i, (_, row) in enumerate(top_inflow.iterrows(), 1):
            val = row[inflow_col]
            val_yi = val / 1e8 if abs(val) > 1e6 else val
            lines.append(f"  {i:>2}. {row[name_col]:<12s}  净流入: {val_yi:>+.2f} 亿")

        bot_inflow = fund_df.nsmallest(5, inflow_col)
        lines.append(sub_section("主力净流出前 5"))
        for i, (_, row) in enumerate(bot_inflow.iterrows(), 1):
            val = row[inflow_col]
            val_yi = val / 1e8 if abs(val) > 1e6 else val
            lines.append(f"  {i:>2}. {row[name_col]:<12s}  净流出: {val_yi:>+.2f} 亿")

    # ==================================================================
    # 7. 操作建议
    # ==================================================================
    lines.append(section("七、今日操作参考"))

    suggestions = []

    # 基于市场情绪
    if avg_change > 1.0:
        suggestions.append("市场强势普涨，可适度追涨科技主线龙头，注意控制仓位")
    elif avg_change > 0:
        suggestions.append("市场小幅上涨，建议关注泽平策略 Top5 推荐板块的回调机会")
    elif avg_change > -1.0:
        suggestions.append("市场震荡偏弱，建议防守为主，关注周期洼地与超跌反弹机会")
    else:
        suggestions.append("市场大幅下跌，建议控制仓位，等待企稳信号再介入")

    # 基于周期位置
    if hottest["temperature"] >= 60:
        suggestions.append(
            f"周期轮动: {hottest['icon']}{hottest['display']}"
            f"处于高热区(温度{hottest['temperature']:.0f}°)，关注该阶段板块机会"
        )
    else:
        suggestions.append("周期轮动: 各阶段温度温和，无明显过热信号")

    # 基于科技赛道
    hot_tracks = [t for t in track_results if t["heat"] >= 55]
    if hot_tracks:
        names = "、".join(f"{t['icon']}{t['display']}" for t in hot_tracks[:3])
        suggestions.append(f"科技赛道: {names} 热度较高，可重点关注")

    # 基于泽平策略
    if result.predictions:
        top3_names = "、".join(s.board_name for s in result.predictions[:3])
        suggestions.append(f"泽平策略 Top3: {top3_names}")

    # 红线提醒
    redline_boards = summary.get("redline", {}).get("boards", [])
    if redline_boards:
        suggestions.append(
            f"红线提醒: 避开 {', '.join(redline_boards[:5])} 等 "
            f"{len(redline_boards)} 个红线板块"
        )

    suggestions.append(
        "以上为量化模型参考，不构成投资建议，请结合基本面与风险偏好决策"
    )

    for i, s in enumerate(suggestions, 1):
        lines.append(f"  {i}. {s}")

    # ==================================================================
    # 尾部
    # ==================================================================
    lines.append("")
    lines.append("=" * 60)
    lines.append(f"  报告生成时间: {today}")
    lines.append(f"  数据来源: {data_source}")
    lines.append("  策略引擎: Real-AI-R v0.1.0")
    lines.append("  免责声明: 本报告仅供学习参考，不构成任何投资建议")
    lines.append("=" * 60)

    return "\n".join(lines)


# ======================================================================
# 入口
# ======================================================================

if __name__ == "__main__":
    demo = "--demo" in sys.argv
    print("正在生成日报，请稍候...\n")
    report = generate_daily_report(demo_mode=demo)
    print(report)

    # 保存到文件
    out_dir = Path(__file__).resolve().parent.parent / "reports"
    out_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d")
    out_path = out_dir / f"daily_report_{date_str}.txt"
    out_path.write_text(report, encoding="utf-8")
    print(f"\n日报已保存至: {out_path}")
