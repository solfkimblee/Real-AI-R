"""宏观分析 Streamlit 页面

包含六个功能标签页：
1. 板块分类总览 — 科技/周期/红线三大类标签分布
2. 周期轮动仪表盘 — 五段论温度计
3. 科技赛道追踪 — 六大赛道热度对比
4. 避雷指南 — 红线禁区板块明细
5. 攻防组合 — 科技矛 + 周期盾推荐
6. 泽平宏观策略 — 融合宏观方法论 + 规则引擎预测
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from real_ai_r.macro.classifier import SectorClassifier
from real_ai_r.macro.cycle_tracker import CycleTracker
from real_ai_r.macro.portfolio import AttackDefensePortfolio
from real_ai_r.macro.red_filter import RedLineFilter
from real_ai_r.macro.tech_tracker import TechTracker
from real_ai_r.macro.zeping_strategy import ZepingMacroStrategy, ZepingWeights
from real_ai_r.sector.monitor import SectorMonitor

# ======================================================================
# 1. 板块分类总览
# ======================================================================

def render_sector_classification() -> None:
    """渲染板块分类总览页。"""
    st.markdown("## 🏷️ 板块宏观分类总览")
    st.info(
        "基于**泽平宏观框架**，将所有板块划分为"
        "**🗡️ 科技主线** / **🛡️ 周期主线** / **🚫 红线禁区** / **⚪ 其他**"
    )

    col_type, col_btn = st.columns([3, 1])
    with col_type:
        board_type = st.radio(
            "板块类型",
            ["行业板块", "概念板块"],
            horizontal=True,
            key="cls_board_type",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔄 刷新", key="cls_refresh")

    bt = "industry" if board_type == "行业板块" else "concept"

    with st.spinner("正在分类板块..."):
        try:
            board_df = SectorMonitor.get_board_list(bt)
            classifier = SectorClassifier()
            classified = classifier.classify_dataframe(board_df)
            summary = classifier.get_category_summary(classified)
        except Exception as e:
            st.error(f"数据获取失败: {e}")
            return

    # ---- 分类统计卡片 ----
    st.markdown("### 📊 分类统计")
    c1, c2, c3, c4 = st.columns(4)
    cat_cols = [
        (c1, "tech", "🗡️ 科技主线"),
        (c2, "cycle", "🛡️ 周期主线"),
        (c3, "redline", "🚫 红线禁区"),
        (c4, "neutral", "⚪ 其他"),
    ]
    for col, cat_key, label in cat_cols:
        info = summary.get(cat_key, {"count": 0, "avg_change": 0})
        with col:
            st.metric(
                label,
                f"{info['count']} 个板块",
                delta=f"{info['avg_change']:.2f}%",
            )

    # ---- 分类饼图 ----
    st.markdown("### 🥧 分类分布")
    pie_data = pd.DataFrame([
        {"分类": "🗡️ 科技主线", "数量": summary.get("tech", {}).get("count", 0)},
        {"分类": "🛡️ 周期主线", "数量": summary.get("cycle", {}).get("count", 0)},
        {"分类": "🚫 红线禁区", "数量": summary.get("redline", {}).get("count", 0)},
        {"分类": "⚪ 其他", "数量": summary.get("neutral", {}).get("count", 0)},
    ])
    fig_pie = px.pie(
        pie_data, names="分类", values="数量",
        color="分类",
        color_discrete_map={
            "🗡️ 科技主线": "#FF6B6B",
            "🛡️ 周期主线": "#4ECDC4",
            "🚫 红线禁区": "#95A5A6",
            "⚪ 其他": "#DDD",
        },
    )
    fig_pie.update_layout(height=350, margin=dict(t=20, b=20))
    st.plotly_chart(fig_pie, use_container_width=True)

    # ---- 各分类涨幅对比 ----
    st.markdown("### 📈 各分类平均涨幅对比")
    bar_data = pd.DataFrame([
        {
            "分类": info["display_name"],
            "平均涨幅%": info["avg_change"],
            "板块数": info["count"],
        }
        for info in summary.values()
    ])
    fig_bar = px.bar(
        bar_data, x="分类", y="平均涨幅%",
        color="平均涨幅%",
        color_continuous_scale=["green", "yellow", "red"],
        text="平均涨幅%",
    )
    fig_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_bar.update_layout(height=350, margin=dict(t=20, b=30))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ---- 分类明细表 ----
    st.markdown("### 📋 完整分类列表")
    display = classified[
        ["name", "change_pct", "turnover_rate", "macro_icon", "macro_display", "macro_desc"]
    ].copy()
    display.columns = ["板块", "涨跌幅%", "换手率%", "标签", "分类", "说明"]
    st.dataframe(display, use_container_width=True, hide_index=True, height=400)


# ======================================================================
# 2. 周期轮动仪表盘
# ======================================================================

def render_cycle_dashboard() -> None:
    """渲染周期轮动仪表盘。"""
    st.markdown("## 🔄 大宗商品周期轮动仪表盘")
    st.info(
        "**五段论**：贵金属 → 基本金属 → 传统能源 → 农业后周期 → 必选消费\n\n"
        "温度越高 = 该阶段越活跃，资金越集中"
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        st.button("🔄 刷新数据", key="cycle_refresh")
    with col_info:
        st.caption("数据来源：AKShare 实时行情 + 资金流向")

    with st.spinner("正在追踪周期轮动..."):
        try:
            tracker = CycleTracker()
            stages = tracker.track()
        except Exception as e:
            st.error(f"周期追踪失败: {e}")
            return

    if not stages:
        st.warning("未获取到周期数据")
        return

    # ---- 温度计面板 ----
    st.markdown("### 🌡️ 各阶段温度")
    cols = st.columns(5)
    for i, stage in enumerate(stages):
        with cols[i]:
            # 温度颜色
            if stage.temperature >= 70:
                color = "🔴"
            elif stage.temperature >= 50:
                color = "🟡"
            else:
                color = "🟢"
            st.metric(
                f"{stage.icon} {stage.display}",
                f"{stage.temperature:.0f}°",
                delta=f"{stage.avg_change_pct:+.2f}%",
            )
            st.caption(f"{color} {stage.framework_status}")

    # ---- 温度柱状图 ----
    st.markdown("### 📊 周期温度对比")
    temp_data = pd.DataFrame([
        {
            "阶段": f"{s.icon} {s.display}",
            "温度": s.temperature,
            "涨幅%": s.avg_change_pct,
            "定位": s.framework_status,
        }
        for s in stages
    ])
    fig = px.bar(
        temp_data, x="阶段", y="温度",
        color="温度",
        color_continuous_scale=["#2E86AB", "#F6D55C", "#ED553B"],
        text="温度",
        hover_data=["涨幅%", "定位"],
    )
    fig.update_traces(texttemplate="%{text:.0f}°", textposition="outside")
    fig.update_layout(
        height=400, margin=dict(t=20, b=30),
        yaxis=dict(range=[0, 110]),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ---- 五段论流程图 ----
    st.markdown("### 🗺️ 五段论演进沙盘")
    for stage in stages:
        arrow = "→" if stage.stage < 5 else ""
        if stage.temperature >= 70:
            status_badge = "🔥 **高热**"
        elif stage.temperature >= 50:
            status_badge = "⚡ 温热"
        else:
            status_badge = "❄️ 低温"

        st.markdown(
            f"**阶段{stage.stage}** {stage.icon} **{stage.display}** "
            f"| 温度 `{stage.temperature:.0f}°` | {status_badge} "
            f"| 框架定位：{stage.framework_status} "
            f"| 匹配板块：{', '.join(stage.matched_boards[:5]) or '无'} "
            f"{arrow}"
        )

    # ---- 详细数据表 ----
    st.markdown("### 📋 详细数据")
    detail_data = pd.DataFrame([
        {
            "阶段": f"阶段{s.stage}",
            "名称": f"{s.icon} {s.display}",
            "温度": s.temperature,
            "涨幅%": s.avg_change_pct,
            "换手率%": s.avg_turnover,
            "资金得分": s.fund_flow_score,
            "动量得分": s.momentum_score,
            "框架定位": s.framework_status,
            "匹配板块数": len(s.matched_boards),
        }
        for s in stages
    ])
    st.dataframe(detail_data, use_container_width=True, hide_index=True)


# ======================================================================
# 3. 科技赛道追踪
# ======================================================================

def render_tech_tracker() -> None:
    """渲染科技六赛道追踪页。"""
    st.markdown("## 🚀 科技六赛道实时追踪")
    st.info(
        "追踪康波第六波六大科技赛道：\n"
        "🔧 芯片/算力 | 🧠 大模型/Agent | 🤖 机器人/自动驾驶 | "
        "🏥 AI医疗 | 🚀 商业航天 | 🚗 新能源车"
    )

    st.button("🔄 刷新数据", key="tech_refresh")

    with st.spinner("正在追踪科技赛道..."):
        try:
            tracker = TechTracker()
            tracks = tracker.track_all()
            comparison_df = tracker.get_track_comparison()
        except Exception as e:
            st.error(f"科技赛道追踪失败: {e}")
            return

    if not tracks:
        st.warning("未获取到科技赛道数据")
        return

    # ---- 赛道热度卡片 ----
    st.markdown("### 🔥 赛道热度排行")
    cols = st.columns(3)
    for i, track in enumerate(tracks[:6]):
        with cols[i % 3]:
            if track.heat_score >= 70:
                badge = "🔥"
            elif track.heat_score >= 50:
                badge = "⚡"
            else:
                badge = "❄️"
            st.metric(
                f"{track.icon} {track.display}",
                f"{track.heat_score:.0f} 分",
                delta=f"{track.avg_change_pct:+.2f}%",
            )
            st.caption(
                f"{badge} 上涨{track.total_rise}/下跌{track.total_fall} | "
                f"领涨：{track.top_lead_stock} ({track.top_lead_pct:+.1f}%)"
            )

    # ---- 热度雷达图 ----
    st.markdown("### 🎯 赛道热度雷达图")
    fig_radar = go.Figure()
    labels = [f"{t.icon} {t.display}" for t in tracks[:6]]
    values = [t.heat_score for t in tracks[:6]]
    # 闭合雷达图
    labels_closed = labels + [labels[0]]
    values_closed = values + [values[0]]
    fig_radar.add_trace(go.Scatterpolar(
        r=values_closed,
        theta=labels_closed,
        fill="toself",
        name="热度",
        line=dict(color="#FF6B6B"),
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        height=450, margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ---- 赛道涨幅对比 ----
    st.markdown("### 📊 赛道涨幅对比")
    change_data = pd.DataFrame([
        {
            "赛道": f"{t.icon} {t.display}",
            "平均涨幅%": t.avg_change_pct,
            "热度": t.heat_score,
        }
        for t in tracks[:6]
    ])
    fig_bar = px.bar(
        change_data, x="赛道", y="平均涨幅%",
        color="热度",
        color_continuous_scale="YlOrRd",
        text="平均涨幅%",
    )
    fig_bar.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
    fig_bar.update_layout(height=400, margin=dict(t=20, b=30))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ---- 详细对比表 ----
    st.markdown("### 📋 赛道详细对比")
    st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    # ---- 各赛道匹配板块 ----
    with st.expander("📂 各赛道匹配板块明细", expanded=False):
        for track in tracks[:6]:
            if track.matched_boards:
                st.markdown(
                    f"**{track.icon} {track.display}** ({len(track.matched_boards)} 个板块)："
                    f" {', '.join(track.matched_boards[:15])}"
                )


# ======================================================================
# 4. 避雷指南
# ======================================================================

def render_redline_guide() -> None:
    """渲染避雷指南页。"""
    st.markdown("## 🚫 避雷指南 — 红线禁区")
    st.warning(
        "以下板块因**基本面趋势长期向下**，应避免投资：\n\n"
        "🏚️ 房地产及泛地产链 | 🍶 传统白酒 | 🏪 传统零售/低端餐饮 | 💻 旧软件外包"
    )

    # ---- 红线禁区说明 ----
    st.markdown("### ⚠️ 四大红线禁区")
    red_filter = RedLineFilter()
    zones = red_filter.get_zone_descriptions()

    for zone in zones:
        st.markdown(
            f"#### {zone['icon']} {zone['name']}\n"
            f"> {zone['description']}\n\n"
            f"关键词：`{'`、`'.join(zone['keywords'][:8])}`"
        )

    # ---- 实时红线板块扫描 ----
    st.markdown("---")
    st.markdown("### 🔍 实时红线板块扫描")

    col_type, col_btn = st.columns([3, 1])
    with col_type:
        board_type = st.radio(
            "板块类型",
            ["行业板块", "概念板块"],
            horizontal=True,
            key="red_board_type",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔍 扫描", key="red_scan")

    bt = "industry" if board_type == "行业板块" else "concept"

    with st.spinner("正在扫描红线板块..."):
        try:
            board_df = SectorMonitor.get_board_list(bt)
            red_boards = red_filter.filter_boards(board_df, keep_redline=True)
            safe_boards = red_filter.filter_boards(board_df, keep_redline=False)
        except Exception as e:
            st.error(f"扫描失败: {e}")
            return

    col1, col2 = st.columns(2)
    with col1:
        st.metric("🚫 红线板块", f"{len(red_boards)} 个")
    with col2:
        st.metric("✅ 安全板块", f"{len(safe_boards)} 个")

    if not red_boards.empty:
        st.markdown("### 🚫 红线板块列表")
        redline_details = red_filter.get_redline_summary(red_boards)
        red_display = pd.DataFrame(redline_details)
        if not red_display.empty:
            red_display.columns = ["板块", "红线原因"] + (
                ["涨跌幅%"] if "change_pct" in red_display.columns else []
            ) + (
                ["换手率%"] if "turnover_rate" in red_display.columns else []
            )
        st.dataframe(red_display, use_container_width=True, hide_index=True)

        # 红线板块涨跌幅分布
        if "change_pct" in red_boards.columns:
            st.markdown("### 📉 红线板块今日涨跌幅")
            fig = px.bar(
                red_boards.nlargest(20, "change_pct"),
                x="change_pct", y="name", orientation="h",
                color="change_pct",
                color_continuous_scale=["green", "yellow", "red"],
                labels={"change_pct": "涨跌幅 (%)", "name": "板块"},
            )
            fig.update_layout(
                height=max(300, len(red_boards.head(20)) * 30),
                yaxis=dict(autorange="reversed"),
                margin=dict(t=20, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("未发现红线板块 — 当前市场板块均不在禁区内")


# ======================================================================
# 5. 攻防组合
# ======================================================================

def render_portfolio() -> None:
    """渲染攻防组合推荐页。"""
    st.markdown("## ⚔️ 攻防组合推荐")
    st.info(
        "**核心阵型**：以科技真龙头为 🗡️ **矛**（博取超额收益），"
        "以周期洼地为 🛡️ **盾**（防御通胀与黑天鹅）\n\n"
        "自动排除红线禁区标的，实时计算最优攻防配比"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        attack_ratio = st.slider(
            "进攻比例", 0.3, 0.8, 0.6, 0.05,
            key="pf_attack_ratio",
            help="进攻仓位占比，剩余为防御仓位",
        )
    with col2:
        n_attack = st.slider("进攻个股数", 3, 10, 5, key="pf_n_attack")
    with col3:
        n_defense = st.slider("防御个股数", 2, 8, 4, key="pf_n_defense")

    build_btn = st.button("⚔️ 构建组合", type="primary", key="pf_build")

    if build_btn:
        with st.spinner("⚔️ 正在构建攻防组合...（需要获取多个板块数据，请稍候）"):
            try:
                builder = AttackDefensePortfolio(
                    attack_ratio=attack_ratio,
                    attack_slots=n_attack,
                    defense_slots=n_defense,
                )
                result = builder.build()
                portfolio_df = builder.to_dataframe(result)
            except Exception as e:
                st.error(f"组合构建失败: {e}")
                return

        if portfolio_df.empty:
            st.warning("未能构建有效组合，可能是数据获取失败")
            return

        st.success(
            f"组合构建完成！🗡️ 进攻 {result.summary['attack_count']} 只 "
            f"+ 🛡️ 防御 {result.summary['defense_count']} 只"
        )

        # ---- 攻防比例图 ----
        st.markdown("### 📊 攻防配比")
        col_pie, col_info = st.columns([1, 1])
        with col_pie:
            ratio_data = pd.DataFrame([
                {"角色": "🗡️ 科技矛（进攻）", "比例": result.attack_ratio * 100},
                {"角色": "🛡️ 周期盾（防御）", "比例": result.defense_ratio * 100},
            ])
            fig_ratio = px.pie(
                ratio_data, names="角色", values="比例",
                color="角色",
                color_discrete_map={
                    "🗡️ 科技矛（进攻）": "#FF6B6B",
                    "🛡️ 周期盾（防御）": "#4ECDC4",
                },
            )
            fig_ratio.update_layout(height=300, margin=dict(t=20, b=20))
            st.plotly_chart(fig_ratio, use_container_width=True)

        with col_info:
            st.markdown("**进攻赛道：**")
            for track in result.summary.get("attack_tracks", []):
                st.markdown(f"- {track}")
            st.markdown("**防御阶段：**")
            for track in result.summary.get("defense_tracks", []):
                st.markdown(f"- {track}")

        # ---- 进攻仓位 ----
        if result.attack_slots:
            st.markdown("### 🗡️ 进攻仓位（科技矛）")
            attack_df = pd.DataFrame([
                {
                    "赛道": s.track,
                    "板块": s.board_name,
                    "代码": s.stock_code,
                    "名称": s.stock_name,
                    "最新价": s.price,
                    "涨跌幅%": s.change_pct,
                    "评分": round(s.score, 1),
                    "理由": s.reason,
                }
                for s in result.attack_slots
            ])
            st.dataframe(attack_df, use_container_width=True, hide_index=True)

            # 进攻仓位涨幅图
            if len(attack_df) > 0:
                fig_atk = px.bar(
                    attack_df, x="涨跌幅%", y="名称", orientation="h",
                    color="评分", color_continuous_scale="YlOrRd",
                    labels={"涨跌幅%": "涨跌幅 (%)", "名称": "股票"},
                )
                fig_atk.update_layout(
                    height=max(250, len(attack_df) * 40),
                    yaxis=dict(autorange="reversed"),
                    margin=dict(t=20, b=30),
                )
                st.plotly_chart(fig_atk, use_container_width=True)

        # ---- 防御仓位 ----
        if result.defense_slots:
            st.markdown("### 🛡️ 防御仓位（周期盾）")
            defense_df = pd.DataFrame([
                {
                    "阶段": s.track,
                    "板块": s.board_name,
                    "代码": s.stock_code,
                    "名称": s.stock_name,
                    "最新价": s.price,
                    "涨跌幅%": s.change_pct,
                    "评分": round(s.score, 1),
                    "理由": s.reason,
                }
                for s in result.defense_slots
            ])
            st.dataframe(defense_df, use_container_width=True, hide_index=True)

            if len(defense_df) > 0:
                fig_def = px.bar(
                    defense_df, x="涨跌幅%", y="名称", orientation="h",
                    color="评分", color_continuous_scale="Blues",
                    labels={"涨跌幅%": "涨跌幅 (%)", "名称": "股票"},
                )
                fig_def.update_layout(
                    height=max(250, len(defense_df) * 40),
                    yaxis=dict(autorange="reversed"),
                    margin=dict(t=20, b=30),
                )
                st.plotly_chart(fig_def, use_container_width=True)

        # ---- 完整组合表 ----
        st.markdown("### 📋 完整组合明细")
        st.dataframe(portfolio_df, use_container_width=True, hide_index=True)


# ======================================================================
# 6. 泽平宏观策略
# ======================================================================

def render_zeping_strategy() -> None:
    """渲染泽平宏观策略预测页。"""
    st.markdown("## 🧭 泽平宏观策略")
    st.info(
        "**深度融合泽平投资方法论 + 规则引擎量化信号**\n\n"
        "核心公式：`总分 = 宏观维度(40%) + 量化维度(40%) + 周期维度(20%)`\n\n"
        "📌 宏观定方向 → 周期定节奏 → 产业定赛道 → 壁垒和订单定龙头 → 催化剂定爆发"
    )

    # ---- 参数配置 ----
    with st.expander("⚙️ 策略参数配置", expanded=False):
        st.markdown("#### 三维度权重")
        col_w1, col_w2, col_w3 = st.columns(3)
        with col_w1:
            w_macro = st.slider(
                "宏观维度权重", 0.1, 0.6, 0.4, 0.05, key="zp_w_macro",
                help="科技/周期分类 + 赛道热度 + 红线过滤",
            )
        with col_w2:
            w_quant = st.slider(
                "量化维度权重", 0.1, 0.6, 0.4, 0.05, key="zp_w_quant",
                help="动量 + 换手率 + 上涨广度 + 领涨强度",
            )
        with col_w3:
            w_cycle = st.slider(
                "周期维度权重", 0.0, 0.4, 0.2, 0.05, key="zp_w_cycle",
                help="五段论阶段匹配度（顺周期加分、逆周期减分）",
            )

        # 归一化权重
        w_total = w_macro + w_quant + w_cycle
        if abs(w_total - 1.0) > 0.01:
            st.warning(f"权重之和为 {w_total:.2f}，已自动归一化为 1.0")
            w_macro, w_quant, w_cycle = (
                w_macro / w_total,
                w_quant / w_total,
                w_cycle / w_total,
            )

        st.markdown("#### 其他参数")
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            top_n = st.slider("推荐板块数", 5, 30, 10, key="zp_top_n")
        with col_p2:
            board_type = st.radio(
                "板块类型", ["行业板块", "概念板块"],
                horizontal=True, key="zp_board_type",
            )

    # ---- 运行按钮 ----
    run_btn = st.button("🧭 运行泽平宏观策略", type="primary", key="zp_run")

    if not run_btn:
        # 方法论说明
        st.markdown("---")
        st.markdown("### 📖 泽平投资方法论")
        col_m1, col_m2 = st.columns(2)

        with col_m1:
            st.markdown("""
            **宏观维度 (40%)**
            - 🗡️ **科技主线** → 基础分70 + 赛道热度加成(0-30)
            - 🛡️ **周期主线** → 基础分55
            - 🚫 **红线禁区** → 直接排除
            - ⚪ **其他** → 基础分40

            **"卖铲人"偏好**
            - 🔧 芯片/算力链 → 额外加10分
            - 🚗 新能源车 → 额外加10分
            - 原因：基础设施层更早兑现业绩
            """)

        with col_m2:
            st.markdown("""
            **量化维度 (40%)**
            - 📈 1日动量 × 5（趋势延续，A股最有效因子）
            - 🔄 换手率 × 2（活跃度信号）
            - 📊 上涨广度（板块内多数个股涨=健康趋势）
            - 🏆 领涨强度（龙头效应加成）

            **周期维度 (20%)**
            - 🔥 热阶段匹配 → +20分
            - ⚡ 相邻阶段 → +10分
            - ❄️ 远离阶段 → -5分
            - 🌾 重点布局区(农业后周期) → +15分
            """)

        return

    # ---- 运行策略 ----
    bt = "industry" if board_type == "行业板块" else "concept"
    weights = ZepingWeights(macro=w_macro, quant=w_quant, cycle=w_cycle)
    strategy = ZepingMacroStrategy(weights=weights)

    with st.spinner("🧭 正在运行泽平宏观策略..."):
        try:
            board_df = SectorMonitor.get_board_list(bt)
            result = strategy.predict(board_df=board_df, top_n=top_n)
        except Exception as e:
            st.error(f"策略运行失败: {e}")
            return

    if not result.predictions:
        st.warning("未生成有效预测，可能是数据获取失败")
        return

    # ---- 策略摘要 ----
    st.success(
        f"策略运行完成！评估 {result.total_boards} 个板块，"
        f"过滤 {result.filtered_redline} 个红线板块"
    )

    st.markdown("### 🎯 策略摘要")
    summary_lines = result.strategy_summary.split("\n")
    for line in summary_lines:
        if line.strip():
            st.markdown(f"- {line.strip()}")

    # ---- 核心指标卡片 ----
    st.markdown("### 📊 市场风格判断")
    col_s1, col_s2, col_s3, col_s4 = st.columns(4)
    top_preds = result.predictions
    tech_count = sum(1 for s in top_preds if s.macro_category == "tech")
    cycle_count = sum(1 for s in top_preds if s.macro_category == "cycle")
    neutral_count = sum(1 for s in top_preds if s.macro_category == "neutral")

    with col_s1:
        st.metric("🗡️ 科技主线", f"{tech_count} 个")
    with col_s2:
        st.metric("🛡️ 周期主线", f"{cycle_count} 个")
    with col_s3:
        st.metric("⚪ 其他", f"{neutral_count} 个")
    with col_s4:
        avg_score = sum(s.total_score for s in top_preds) / len(top_preds)
        st.metric("平均得分", f"{avg_score:.1f}")

    # ---- 预测排名表 ----
    st.markdown("### 🏆 泽平宏观策略 Top 推荐")
    score_df = strategy.to_dataframe(top_preds)
    st.dataframe(score_df, use_container_width=True, hide_index=True)

    # ---- 三维度评分柱状图 ----
    st.markdown("### 📊 三维度评分拆解")
    chart_data = pd.DataFrame([
        {
            "板块": s.board_name,
            "宏观分": s.macro_score * weights.macro,
            "量化分": s.quant_score * weights.quant,
            "周期分": s.cycle_score * weights.cycle,
        }
        for s in top_preds
    ])
    fig_stack = go.Figure()
    colors = {"宏观分": "#FF6B6B", "量化分": "#4ECDC4", "周期分": "#45B7D1"}
    for dim, color in colors.items():
        fig_stack.add_trace(go.Bar(
            name=dim,
            x=chart_data["板块"],
            y=chart_data[dim],
            marker_color=color,
        ))
    fig_stack.update_layout(
        barmode="stack",
        height=450,
        margin=dict(t=30, b=60),
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_stack, use_container_width=True)

    # ---- 分类分布饼图 ----
    st.markdown("### 🥧 推荐板块分类分布")
    pie_data = pd.DataFrame([
        {"分类": "🗡️ 科技主线", "数量": tech_count},
        {"分类": "🛡️ 周期主线", "数量": cycle_count},
        {"分类": "⚪ 其他", "数量": neutral_count},
    ])
    pie_data = pie_data[pie_data["数量"] > 0]
    if not pie_data.empty:
        fig_pie = px.pie(
            pie_data, names="分类", values="数量",
            color="分类",
            color_discrete_map={
                "🗡️ 科技主线": "#FF6B6B",
                "🛡️ 周期主线": "#4ECDC4",
                "⚪ 其他": "#DDD",
            },
        )
        fig_pie.update_layout(height=350, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    # ---- 推荐理由详情 ----
    with st.expander("📝 各板块推荐理由", expanded=False):
        for s in top_preds:
            reasons_str = " | ".join(s.reasons)
            icon = (
                "🗡️" if s.macro_category == "tech"
                else "🛡️" if s.macro_category == "cycle"
                else "⚪"
            )
            st.markdown(
                f"**{icon} {s.board_name}** (总分 {s.total_score:.1f}) — {reasons_str}"
            )
