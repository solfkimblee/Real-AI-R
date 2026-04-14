"""催化剂追踪 & 投资决策检查清单 Streamlit 页面"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from real_ai_r.catalyst import CatalystTracker
from real_ai_r.checklist import (
    InvestmentChecklist,
    create_zepin_checklist,
    evaluate_with_macro_context,
)
from real_ai_r.macro.classifier import SectorClassifier


def render_catalyst_tracker() -> None:
    """渲染催化剂追踪标签页。"""
    st.markdown("## 📅 催化剂追踪")
    st.info(
        "**催化剂定爆发** — 追踪财报日历、宏观政策事件和产业催化剂，"
        "帮助判断最佳买卖时机"
    )

    sub_tab1, sub_tab2, sub_tab3 = st.tabs([
        "📊 财报日历", "🏛️ 宏观政策事件", "🔬 产业催化剂",
    ])

    # ---- 子标签1：财报日历 ----
    with sub_tab1:
        _render_earnings_calendar()

    # ---- 子标签2：宏观政策事件 ----
    with sub_tab2:
        _render_macro_events()

    # ---- 子标签3：产业催化剂 ----
    with sub_tab3:
        _render_industry_catalysts()


def _render_earnings_calendar() -> None:
    """渲染财报日历子页面。"""
    st.markdown("### 📊 财报披露日历")

    # 显示当前财报日历
    calendar = CatalystTracker.get_report_calendar()
    if calendar:
        cal_df = pd.DataFrame(calendar)
        cal_df = cal_df.rename(columns={
            "report_name": "报告类型",
            "period": "报告期",
            "disclosure_window": "披露窗口",
            "status": "当前状态",
        })
        st.dataframe(cal_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 业绩预告/快报查询
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        data_type = st.radio(
            "数据类型",
            ["业绩预告", "业绩快报"],
            horizontal=True,
            key="catalyst_data_type",
        )
    with col2:
        current_period = CatalystTracker.get_current_report_period()
        period = st.text_input(
            "报告期 (YYYYMMDD)",
            value=current_period,
            key="catalyst_period",
            help="如 20250331 表示2025年一季报",
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        fetch_btn = st.button("📥 获取数据", type="primary", key="catalyst_fetch")

    if fetch_btn:
        with st.spinner("正在获取财报数据..."):
            if data_type == "业绩预告":
                df = CatalystTracker.get_earnings_forecast(period)
                if df.empty:
                    st.warning(f"未找到 {period} 期间的业绩预告数据")
                    return

                st.success(f"获取到 {len(df)} 条业绩预告")

                # 预告类型分布
                if "forecast_type" in df.columns:
                    type_counts = df["forecast_type"].value_counts()
                    fig = px.pie(
                        values=type_counts.values,
                        names=type_counts.index,
                        title="业绩预告类型分布",
                        color_discrete_sequence=px.colors.qualitative.Set2,
                    )
                    fig.update_layout(height=350, margin=dict(t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                # 数据表格
                display_cols = ["code", "name", "indicator", "forecast_type",
                                "change_pct", "announce_date"]
                available = [c for c in display_cols if c in df.columns]
                display = df[available].copy()
                display = display.rename(columns={
                    "code": "代码", "name": "名称",
                    "indicator": "预测指标",
                    "forecast_type": "预告类型",
                    "change_pct": "变动幅度%",
                    "announce_date": "公告日期",
                })
                st.dataframe(
                    display, use_container_width=True,
                    hide_index=True, height=400,
                )

            else:  # 业绩快报
                df = CatalystTracker.get_earnings_express(period)
                if df.empty:
                    st.warning(f"未找到 {period} 期间的业绩快报数据")
                    return

                st.success(f"获取到 {len(df)} 条业绩快报")

                # 营收同比分布
                if "revenue_yoy" in df.columns:
                    fig = px.histogram(
                        df, x="revenue_yoy", nbins=30,
                        title="营收同比增长率分布",
                        labels={"revenue_yoy": "营收同比增长率 (%)"},
                        color_discrete_sequence=["#636EFA"],
                    )
                    fig.add_vline(x=0, line_dash="dash", line_color="red")
                    fig.update_layout(height=350, margin=dict(t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True)

                # 数据表格
                display_cols = [
                    "code", "name", "eps", "revenue_yoy",
                    "profit_yoy", "roe", "industry", "announce_date",
                ]
                available = [c for c in display_cols if c in df.columns]
                display = df[available].copy()
                display = display.rename(columns={
                    "code": "代码", "name": "名称", "eps": "每股收益",
                    "revenue_yoy": "营收同比%", "profit_yoy": "净利润同比%",
                    "roe": "ROE%", "industry": "行业",
                    "announce_date": "公告日期",
                })
                st.dataframe(
                    display, use_container_width=True,
                    hide_index=True, height=400,
                )


def _render_macro_events() -> None:
    """渲染宏观政策事件子页面。"""
    st.markdown("### 🏛️ 宏观政策事件日历")
    st.caption("基于已知的宏观经济数据发布节奏，追踪影响市场的关键政策事件")

    events = CatalystTracker.get_macro_events()
    if events:
        events_df = pd.DataFrame(events)
        events_df = events_df.rename(columns={
            "event": "事件",
            "frequency": "频率",
            "impact": "影响",
            "affected_sectors": "影响板块",
            "category": "类别",
        })

        # 用卡片形式展示
        for _, row in events_df.iterrows():
            with st.expander(f"📌 {row['事件']} — {row['频率']}", expanded=False):
                st.markdown(f"**影响：** {row['影响']}")
                st.markdown(f"**影响板块：** {row['影响板块']}")

        st.markdown("---")
        st.markdown("### 📋 完整事件列表")
        st.dataframe(
            events_df[["事件", "频率", "影响", "影响板块"]],
            use_container_width=True, hide_index=True,
        )


def _render_industry_catalysts() -> None:
    """渲染产业催化剂子页面。"""
    st.markdown("### 🔬 产业催化剂追踪")
    st.caption("基于泽平方法论，追踪各赛道关键催化剂事件")

    catalysts = CatalystTracker.get_industry_catalysts()

    # 按类别分组显示
    col_tech, col_cycle = st.columns(2)

    with col_tech:
        st.markdown("#### 🚀 科技主线催化剂")
        for item in catalysts:
            if item["category"] == "科技主线":
                with st.expander(f"**{item['track']}**", expanded=False):
                    for c in item["catalysts"]:
                        st.markdown(f"- {c}")

    with col_cycle:
        st.markdown("#### 🔄 周期主线催化剂")
        for item in catalysts:
            if item["category"] == "周期主线":
                with st.expander(f"**{item['track']}**", expanded=False):
                    for c in item["catalysts"]:
                        st.markdown(f"- {c}")


def render_investment_checklist() -> None:
    """渲染投资决策检查清单标签页。"""
    st.markdown("## 📝 投资决策检查清单")
    st.info(
        "**泽平方法论五问** — 每次做投资判断前，用结构化清单辅助决策，"
        "避免冲动交易"
    )

    # 输入标的信息
    col1, col2 = st.columns([2, 1])
    with col1:
        target_name = st.text_input(
            "标的名称（板块/个股）",
            placeholder="如：半导体、宁德时代、人工智能",
            key="checklist_target",
        )
    with col2:
        auto_eval = st.checkbox(
            "自动宏观评估",
            value=True,
            key="checklist_auto",
            help="开启后自动根据宏观分类填充部分评分",
        )

    if not target_name:
        # 显示方法论说明
        st.markdown("---")
        st.markdown("""
        ### 📖 泽平投资方法论核心框架

        **一句话概括：** 宏观定方向，周期定节奏，产业定赛道，壁垒和订单定龙头，催化剂定爆发。

        **五问决策框架：**
        1. 🎯 **这是不是未来1-3年的主线？** — 只做景气度最清晰的主线
        2. 📈 **这是趋势机会，还是周期轮动？** — 趋势做主仓，周期做轮动
        3. 📊 **有没有真实订单和业绩兑现？** — 不炒纯概念，看商业闭环
        4. 🔧 **是不是产业链最硬的环节？** — 优先选"卖铲人"
        5. ⏰ **追高还是等催化剂前布局？** — 提前布局 > 追涨杀跌

        **红线禁区：** 房地产链 | 传统白酒 | 传统零售 | 被AI替代的旧行业

        👈 在上方输入标的名称，开始决策评估
        """)
        return

    # 自动宏观评估
    classifier = SectorClassifier()
    label = classifier.classify(target_name)
    is_tech = label.category == "tech"
    is_cycle = label.category == "cycle"
    is_redline = label.category == "redline"

    from real_ai_r.macro.classifier import CYCLE_STAGES
    cycle_stage = 0
    if is_cycle and label.sub_category:
        for key, info in CYCLE_STAGES.items():
            if key == label.sub_category:
                cycle_stage = info["stage"]
                break

    # 显示宏观分类结果
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        if is_tech:
            st.success(f"🚀 科技主线 — {label.sub_category or '通用科技'}")
        elif is_cycle:
            st.info(f"🔄 周期主线 — 阶段{cycle_stage}")
        elif is_redline:
            st.error(f"🚫 红线禁区 — {label.sub_category or '避雷'}")
        else:
            st.warning("⚪ 非核心主线")
    with col_b:
        st.metric("分类", label.category or "other")
    with col_c:
        if label.sub_category:
            st.metric("子分类", label.sub_category)

    st.markdown("---")

    # 红线检查
    if is_redline:
        st.error(
            "⛔ **红线警告** — 该标的属于宏观红线禁区，"
            "基本面长期向下，建议坚决回避！"
        )
        return

    # 五问决策清单
    st.markdown("### 📝 五问决策清单")

    if auto_eval:
        checklist = evaluate_with_macro_context(
            target_name=target_name,
            is_tech=is_tech,
            is_cycle=is_cycle,
            is_redline=is_redline,
            cycle_stage=cycle_stage,
        )
    else:
        checklist = InvestmentChecklist(
            target_name=target_name,
            items=create_zepin_checklist(),
        )

    # 逐项显示并允许用户修改评分
    for i, item in enumerate(checklist.items):
        with st.expander(
            f"{'✅' if item.score >= 60 else '⚠️' if item.score >= 40 else '❌'} "
            f"问题{i + 1}: {item.question}",
            expanded=(i < 2),
        ):
            st.caption(item.description)
            if item.answer:
                st.markdown(f"**自动评估：** {item.answer}")

            # 用户可以手动调整评分
            new_score = st.slider(
                "评分 (0-100)",
                0, 100, item.score,
                key=f"checklist_score_{i}",
                help="0=完全不符合, 50=一般, 100=完全符合",
            )
            checklist.items[i].score = new_score

            # 用户备注
            note = st.text_input(
                "备注",
                value=item.answer,
                key=f"checklist_note_{i}",
                placeholder="填写你的分析依据...",
            )
            checklist.items[i].answer = note

    # 计算总分
    st.markdown("---")
    overall = checklist.compute_overall_score()
    recommendation = checklist.generate_recommendation()

    # 总分展示
    col_score, col_rec = st.columns([1, 2])
    with col_score:
        # 使用仪表盘样式
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=overall,
            title={"text": "综合评分"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkblue"},
                "steps": [
                    {"range": [0, 35], "color": "#ff4b4b"},
                    {"range": [35, 50], "color": "#ffa726"},
                    {"range": [50, 65], "color": "#ffee58"},
                    {"range": [65, 80], "color": "#66bb6a"},
                    {"range": [80, 100], "color": "#2e7d32"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": overall,
                },
            },
        ))
        fig.update_layout(height=250, margin=dict(t=50, b=0, l=30, r=30))
        st.plotly_chart(fig, use_container_width=True)

    with col_rec:
        st.markdown("### 💡 投资建议")
        st.markdown(f"**{recommendation}**")

        if checklist.risk_notes:
            st.markdown("#### ⚠️ 风险提示")
            for note in checklist.risk_notes:
                st.warning(note)

        # 各维度评分雷达图
        labels = [item.question[:8] + "..." for item in checklist.items]
        scores = [item.score for item in checklist.items]

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=scores + [scores[0]],
            theta=labels + [labels[0]],
            fill="toself",
            name=target_name,
            line_color="#636EFA",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            height=300, margin=dict(t=30, b=30),
            showlegend=False,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # 详细评分表
    st.markdown("### 📋 详细评分")
    detail_data = []
    for i, item in enumerate(checklist.items):
        detail_data.append({
            "序号": i + 1,
            "问题": item.question,
            "评分": item.score,
            "权重": f"{item.weight:.0%}",
            "加权分": f"{item.score * item.weight:.1f}",
            "备注": item.answer or "-",
        })
    st.dataframe(
        pd.DataFrame(detail_data),
        use_container_width=True, hide_index=True,
    )
