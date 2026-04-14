"""板块监控 Streamlit 页面"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from real_ai_r.sector.monitor import SectorMonitor
from real_ai_r.sector.predictor import HotSectorPredictor
from real_ai_r.sector.recommender import StockRecommender


def render_sector_monitor() -> None:
    """渲染板块监控标签页。"""
    st.markdown("## 📡 板块实时监控")

    col_type, col_refresh = st.columns([3, 1])
    with col_type:
        board_type = st.radio(
            "板块类型",
            ["行业板块", "概念板块"],
            horizontal=True,
            key="monitor_board_type",
        )
    with col_refresh:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔄 刷新数据", key="refresh_monitor")

    bt = "industry" if board_type == "行业板块" else "concept"

    with st.spinner("📡 正在获取板块数据..."):
        try:
            board_df = SectorMonitor.get_board_list(bt)
            stats = SectorMonitor.get_board_stats(bt)
        except Exception as e:
            st.error(f"数据获取失败: {e}")
            return

    # ---- 涨跌统计卡片 ----
    st.markdown("### 📊 涨跌统计")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("上涨板块", stats["rise_count"],
                   delta=f"{stats['rise_count']}/{stats['total_count']}")
    with c2:
        st.metric("下跌板块", stats["fall_count"],
                   delta=f"-{stats['fall_count']}", delta_color="inverse")
    with c3:
        st.metric("平盘板块", stats["flat_count"])
    with c4:
        st.metric("平均涨幅", f"{stats['avg_change']:.2f}%")

    # ---- 涨幅 Top5 / Bottom5 ----
    col_top, col_bot = st.columns(2)
    with col_top:
        st.markdown("#### 🔥 涨幅前5")
        top5 = pd.DataFrame(stats["top5"])
        if not top5.empty:
            top5.columns = ["板块", "涨幅%", "领涨股", "领涨股涨幅%"]
            st.dataframe(top5, use_container_width=True, hide_index=True)
    with col_bot:
        st.markdown("#### 💚 跌幅前5")
        bot5 = pd.DataFrame(stats["bottom5"])
        if not bot5.empty:
            bot5.columns = ["板块", "涨幅%", "领涨股", "领涨股涨幅%"]
            st.dataframe(bot5, use_container_width=True, hide_index=True)

    # ---- 板块涨跌幅分布图 ----
    st.markdown("### 📈 板块涨跌幅分布")
    fig = px.histogram(
        board_df, x="change_pct", nbins=40,
        labels={"change_pct": "涨跌幅 (%)"},
        title="板块涨跌幅分布",
        color_discrete_sequence=["#636EFA"],
    )
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(height=350, margin=dict(t=40, b=30))
    st.plotly_chart(fig, use_container_width=True)

    # ---- 板块涨幅排行（柱状图 Top20）----
    st.markdown("### 🏆 板块涨幅排行 (Top 20)")
    top20 = board_df.nlargest(20, "change_pct")
    fig2 = px.bar(
        top20, x="change_pct", y="name", orientation="h",
        color="change_pct",
        color_continuous_scale=["green", "yellow", "red"],
        labels={"change_pct": "涨跌幅 (%)", "name": "板块"},
    )
    fig2.update_layout(
        height=500, yaxis=dict(autorange="reversed"),
        margin=dict(t=20, b=30),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # ---- 完整板块列表 ----
    st.markdown("### 📋 全部板块行情")
    display_cols = ["rank", "name", "price", "change_pct",
                    "turnover_rate", "rise_count", "fall_count",
                    "lead_stock", "lead_stock_pct"]
    available = [c for c in display_cols if c in board_df.columns]
    col_labels = {
        "rank": "排名", "name": "板块", "price": "最新价",
        "change_pct": "涨跌幅%", "turnover_rate": "换手率%",
        "rise_count": "上涨家数", "fall_count": "下跌家数",
        "lead_stock": "领涨股", "lead_stock_pct": "领涨股涨幅%",
    }
    display = board_df[available].rename(columns=col_labels)
    st.dataframe(display, use_container_width=True, hide_index=True, height=400)


def render_sector_prediction() -> None:
    """渲染明日热门板块预测标签页。"""
    st.markdown("## 🔮 明日热门板块预测")
    st.info("基于**资金流向 + 涨幅动量 + 活跃度 + 上涨广度 + 领涨强度**五因子综合评分模型")

    col1, col2, col3 = st.columns(3)
    with col1:
        board_type = st.radio(
            "板块类型",
            ["行业板块", "概念板块"],
            horizontal=True,
            key="predict_board_type",
        )
    with col2:
        top_n = st.slider("推荐数量", 5, 30, 10, key="predict_top_n")
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🔮 开始预测", type="primary", key="predict_btn")

    # 因子权重调整（可选）
    with st.expander("⚙️ 调整因子权重", expanded=False):
        st.markdown("调整各因子在综合评分中的权重（总和应为 1.0）")
        w_fund = st.slider("资金流向权重", 0.0, 1.0, 0.30, 0.05, key="w_fund")
        w_mom = st.slider("涨幅动量权重", 0.0, 1.0, 0.25, 0.05, key="w_mom")
        w_act = st.slider("活跃度权重", 0.0, 1.0, 0.20, 0.05, key="w_act")
        w_brd = st.slider("上涨广度权重", 0.0, 1.0, 0.15, 0.05, key="w_brd")
        w_lead = st.slider("领涨强度权重", 0.0, 1.0, 0.10, 0.05, key="w_lead")
        total_w = w_fund + w_mom + w_act + w_brd + w_lead
        if abs(total_w - 1.0) > 0.01:
            st.warning(f"⚠️ 当前权重总和 = {total_w:.2f}，建议调整为 1.0")

    if predict_btn:
        bt = "industry" if board_type == "行业板块" else "concept"
        weights = {
            "fund_flow": w_fund,
            "momentum": w_mom,
            "activity": w_act,
            "breadth": w_brd,
            "lead_strength": w_lead,
        }

        with st.spinner("🔮 正在计算评分模型..."):
            try:
                predictor = HotSectorPredictor(weights=weights, board_type=bt)
                result = predictor.predict(top_n=top_n)
            except Exception as e:
                st.error(f"预测失败: {e}")
                return

        if result.empty:
            st.warning("未获取到足够数据进行预测")
            return

        st.success(f"预测完成！推荐 {len(result)} 个潜力板块")

        # ---- 综合评分排行 ----
        st.markdown("### 🏅 综合评分排行")
        fig = px.bar(
            result, x="total_score", y="name", orientation="h",
            color="total_score",
            color_continuous_scale="YlOrRd",
            labels={"total_score": "综合评分", "name": "板块"},
        )
        fig.update_layout(
            height=max(300, top_n * 35),
            yaxis=dict(autorange="reversed"),
            margin=dict(t=20, b=30),
        )
        st.plotly_chart(fig, use_container_width=True)

        # ---- 雷达图（因子分解）Top 5 ----
        st.markdown("### 🎯 Top 5 因子分解")
        top5 = result.head(5)
        factor_cols = [
            "score_fund_flow", "score_momentum",
            "score_activity", "score_breadth", "score_lead_strength",
        ]
        factor_labels = ["资金流向", "涨幅动量", "活跃度", "上涨广度", "领涨强度"]
        available_factors = [c for c in factor_cols if c in top5.columns]

        if available_factors:
            fig_radar = go.Figure()
            for _, row in top5.iterrows():
                values = [row.get(c, 0) for c in available_factors]
                values.append(values[0])  # 闭合
                labels = [factor_labels[factor_cols.index(c)]
                          for c in available_factors]
                labels.append(labels[0])
                fig_radar.add_trace(go.Scatterpolar(
                    r=values, theta=labels,
                    fill="toself", name=row["name"],
                ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                height=450, margin=dict(t=30, b=30),
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        # ---- 详细数据表 ----
        st.markdown("### 📋 详细评分数据")
        display = result.copy()
        col_labels = {
            "name": "板块", "code": "代码", "total_score": "综合评分",
            "change_pct": "涨跌幅%", "turnover_rate": "换手率%",
            "rise_count": "上涨家数", "fall_count": "下跌家数",
            "lead_stock": "领涨股", "lead_stock_pct": "领涨股涨幅%",
            "net_inflow": "主力净流入",
            "score_fund_flow": "资金流向分", "score_momentum": "动量分",
            "score_activity": "活跃度分", "score_breadth": "广度分",
            "score_lead_strength": "领涨分",
        }
        avail_labels = {k: v for k, v in col_labels.items() if k in display.columns}
        display = display.rename(columns=avail_labels)

        # 格式化净流入
        if "主力净流入" in display.columns:
            display["主力净流入"] = display["主力净流入"].apply(
                lambda x: f"{x / 1e8:.2f}亿" if abs(x) >= 1e8
                else f"{x / 1e4:.0f}万"
            )

        st.dataframe(display, use_container_width=True, hide_index=True)


def render_stock_recommendation() -> None:
    """渲染热门板块股票推荐标签页。"""
    st.markdown("## 💎 热门板块股票推荐")

    # 先获取板块列表让用户选择
    col1, col2 = st.columns(2)
    with col1:
        board_type = st.radio(
            "板块类型",
            ["行业板块", "概念板块"],
            horizontal=True,
            key="rec_board_type",
        )
    with col2:
        bt = "industry" if board_type == "行业板块" else "concept"
        # 预加载板块列表
        try:
            board_list = SectorMonitor.get_board_list(bt)
            board_names = board_list["name"].tolist()
        except Exception:
            board_names = []
            st.warning("板块列表加载失败，请手动输入板块名称")

    if board_names:
        # 按涨幅排序，热门板块优先
        selected_board = st.selectbox(
            "选择板块",
            board_names,
            key="rec_board_select",
        )
    else:
        selected_board = st.text_input("输入板块名称", key="rec_board_input")

    col_n, col_sort, col_btn = st.columns([1, 1, 1])
    with col_n:
        top_n = st.slider("推荐数量", 5, 30, 10, key="rec_top_n")
    with col_sort:
        sort_by = st.selectbox(
            "排序方式",
            ["综合评分", "涨幅", "换手率", "成交额"],
            key="rec_sort",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        rec_btn = st.button("💎 获取推荐", type="primary", key="rec_btn")

    sort_map = {
        "综合评分": "composite",
        "涨幅": "change_pct",
        "换手率": "turnover_rate",
        "成交额": "amount",
    }

    if rec_btn and selected_board:
        with st.spinner(f"💎 正在分析 [{selected_board}] 板块个股..."):
            try:
                result = StockRecommender.recommend(
                    board_name=selected_board,
                    board_type=bt,
                    top_n=top_n,
                    sort_by=sort_map[sort_by],
                )
            except Exception as e:
                st.error(f"获取失败: {e}")
                return

        if result.empty:
            st.warning(f"板块 [{selected_board}] 无有效个股数据")
            return

        st.success(f"从 [{selected_board}] 板块推荐 {len(result)} 只个股")

        # ---- 涨跌幅排行 ----
        st.markdown("### 📊 推荐个股涨跌幅")
        if "change_pct" in result.columns and "name" in result.columns:
            fig = px.bar(
                result, x="change_pct", y="name", orientation="h",
                color="change_pct",
                color_continuous_scale=["green", "yellow", "red"],
                labels={"change_pct": "涨跌幅 (%)", "name": "股票"},
            )
            fig.update_layout(
                height=max(300, len(result) * 32),
                yaxis=dict(autorange="reversed"),
                margin=dict(t=20, b=30),
            )
            st.plotly_chart(fig, use_container_width=True)

        # ---- 综合评分散点图 ----
        if "composite_score" in result.columns:
            st.markdown("### 🎯 综合评分")
            cols_for_scatter = ["name", "change_pct"]
            if "turnover_rate" in result.columns:
                cols_for_scatter.append("turnover_rate")
            if "composite_score" in result.columns:
                fig2 = px.scatter(
                    result,
                    x="change_pct",
                    y="turnover_rate" if "turnover_rate" in result.columns else "change_pct",
                    size="composite_score",
                    color="composite_score",
                    hover_name="name",
                    color_continuous_scale="YlOrRd",
                    labels={
                        "change_pct": "涨跌幅 (%)",
                        "turnover_rate": "换手率 (%)",
                        "composite_score": "综合评分",
                    },
                    title="涨幅 vs 换手率（气泡大小 = 综合评分）",
                )
                fig2.update_layout(height=400, margin=dict(t=40, b=30))
                st.plotly_chart(fig2, use_container_width=True)

        # ---- 详细表格 ----
        st.markdown("### 📋 推荐个股明细")
        display = result.copy()
        col_labels = {
            "code": "代码", "name": "名称", "price": "最新价",
            "change_pct": "涨跌幅%", "turnover_rate": "换手率%",
            "amount": "成交额", "pe_dynamic": "市盈率(动态)",
            "composite_score": "综合评分",
        }
        avail = {k: v for k, v in col_labels.items() if k in display.columns}
        display = display.rename(columns=avail)

        # 格式化成交额
        if "成交额" in display.columns:
            display["成交额"] = display["成交额"].apply(
                lambda x: f"{x / 1e8:.2f}亿" if pd.notna(x) and abs(x) >= 1e8
                else (f"{x / 1e4:.0f}万" if pd.notna(x) else "-")
            )

        st.dataframe(display, use_container_width=True, hide_index=True)

        # ---- 全部成分股 ----
        with st.expander("📂 查看全部成分股", expanded=False):
            try:
                all_stocks = StockRecommender.get_board_stocks(
                    selected_board, bt,
                )
                all_display = all_stocks[
                    ["code", "name", "price", "change_pct",
                     "turnover_rate", "amount"]
                ].copy() if not all_stocks.empty else pd.DataFrame()
                all_display = all_display.rename(columns={
                    "code": "代码", "name": "名称", "price": "最新价",
                    "change_pct": "涨跌幅%", "turnover_rate": "换手率%",
                    "amount": "成交额",
                })
                st.dataframe(all_display, use_container_width=True,
                             hide_index=True, height=400)
            except Exception as e:
                st.error(f"加载失败: {e}")
