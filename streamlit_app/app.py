"""Real-AI-R Streamlit Dashboard

A股量化交易系统 — 交互式回测平台 + 板块分析
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# 确保 src 在 path 中
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from pages_sector import (
    render_sector_monitor,
    render_sector_prediction,
    render_stock_recommendation,
)

from real_ai_r.analysis.charts import (
    plot_candlestick_with_signals,
    plot_drawdown,
    plot_monthly_returns,
    plot_portfolio_value,
)
from real_ai_r.analysis.performance import calculate_metrics, generate_report_df
from real_ai_r.data.fetcher import DataFetcher
from real_ai_r.engine.backtest import BacktestEngine
from real_ai_r.strategies.bollinger_strategy import BollingerStrategy
from real_ai_r.strategies.ma_cross import MACrossStrategy
from real_ai_r.strategies.macd_strategy import MACDStrategy

# ------------------------------------------------------------------
# 页面配置
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Real-AI-R | A股量化回测",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Real-AI-R — A股量化交易回测平台")

# ------------------------------------------------------------------
# 顶部导航标签页
# ------------------------------------------------------------------
tab_backtest, tab_monitor, tab_predict, tab_recommend = st.tabs([
    "🚀 策略回测",
    "📡 板块监控",
    "🔮 热门板块预测",
    "💎 个股推荐",
])

# ==================================================================
# Tab 1: 策略回测（原有功能）
# ==================================================================
with tab_backtest:
    # 侧边栏 — 参数配置
    with st.sidebar:
        st.header("⚙️ 回测参数")

        # 股票选择
        st.subheader("📊 标的选择")
        symbol = st.text_input(
            "股票代码", value="000001",
            help="如: 000001 (平安银行), 600519 (贵州茅台)",
        )
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("开始日期", value=pd.to_datetime("2023-01-01"))
        with col2:
            end_date = st.date_input("结束日期", value=pd.to_datetime("2024-12-31"))

        adjust = st.selectbox("复权方式", ["qfq", "hfq", ""], format_func=lambda x: {
            "qfq": "前复权", "hfq": "后复权", "": "不复权"
        }.get(x, x))

        # 策略选择
        st.subheader("🎯 策略选择")
        strategy_name = st.selectbox(
            "选择策略",
            ["双均线交叉", "MACD策略", "布林带策略"],
        )

        # 策略参数
        if strategy_name == "双均线交叉":
            short_window = st.slider("短期均线", 3, 30, 5)
            long_window = st.slider("长期均线", 10, 120, 20)
        elif strategy_name == "MACD策略":
            macd_fast = st.slider("MACD快线", 5, 20, 12)
            macd_slow = st.slider("MACD慢线", 15, 40, 26)
            macd_signal = st.slider("信号线", 5, 15, 9)
        else:  # 布林带
            bb_window = st.slider("布林带周期", 10, 50, 20)
            bb_std = st.slider("标准差倍数", 1.0, 3.0, 2.0, 0.1)

        # 资金与费用
        st.subheader("💰 资金设置")
        initial_capital = st.number_input(
            "初始资金 (元)", value=100000, step=10000, min_value=10000,
        )
        commission_rate = st.number_input(
            "佣金费率", value=0.0003, step=0.0001, format="%.4f",
        )
        stamp_tax = st.number_input(
            "印花税率", value=0.001, step=0.0001, format="%.4f",
        )
        slippage = st.number_input(
            "滑点比例", value=0.001, step=0.0005, format="%.4f",
        )

        # 运行按钮
        st.markdown("---")
        run_btn = st.button("🚀 运行回测", type="primary", use_container_width=True)

    # 主区域 — 回测执行与结果展示
    if run_btn:
        # 1. 获取数据
        with st.spinner("📥 正在获取行情数据..."):
            try:
                fetcher = DataFetcher()
                data = fetcher.get_stock_daily(
                    symbol=symbol,
                    start_date=start_date.strftime("%Y%m%d"),
                    end_date=end_date.strftime("%Y%m%d"),
                    adjust=adjust,
                )
                st.success(f"获取到 {len(data)} 条日线数据")
            except Exception as e:
                st.error(f"数据获取失败: {e}")
                st.stop()

        # 2. 创建策略
        if strategy_name == "双均线交叉":
            strategy = MACrossStrategy(
                short_window=short_window, long_window=long_window,
            )
        elif strategy_name == "MACD策略":
            strategy = MACDStrategy(
                fast=macd_fast, slow=macd_slow, signal_period=macd_signal,
            )
        else:
            strategy = BollingerStrategy(window=bb_window, num_std=bb_std)

        # 3. 运行回测
        with st.spinner("⚡ 正在运行回测..."):
            engine = BacktestEngine(
                data=data,
                strategy=strategy,
                initial_capital=float(initial_capital),
                commission_rate=commission_rate,
                stamp_tax_rate=stamp_tax,
                slippage=slippage,
            )
            result = engine.run()
            metrics = calculate_metrics(result)

        # 4. 显示结果
        st.markdown("## 📊 回测结果")

        # 核心指标卡片
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            total_ret = metrics.get("总收益率", 0)
            st.metric("总收益率", f"{total_ret:.2%}", delta=f"{total_ret:.2%}")
        with col2:
            annual_ret = metrics.get("年化收益率", 0)
            st.metric("年化收益率", f"{annual_ret:.2%}")
        with col3:
            sharpe = metrics.get("夏普比率", 0)
            st.metric("夏普比率", f"{sharpe:.2f}")
        with col4:
            max_dd = metrics.get("最大回撤", 0)
            st.metric("最大回撤", f"{max_dd:.2%}")
        with col5:
            win_rate = metrics.get("日胜率", 0)
            st.metric("日胜率", f"{win_rate:.2%}")

        st.markdown("---")

        # 净值曲线
        st.plotly_chart(plot_portfolio_value(result), use_container_width=True)

        # 两列布局：回撤 + 月度收益
        col_left, col_right = st.columns(2)
        with col_left:
            st.plotly_chart(plot_drawdown(result), use_container_width=True)
        with col_right:
            st.plotly_chart(plot_monthly_returns(result), use_container_width=True)

        # K线图 + 信号
        st.plotly_chart(
            plot_candlestick_with_signals(data, result), use_container_width=True,
        )

        # 详细指标表格
        st.markdown("### 📋 详细绩效指标")
        report_df = generate_report_df(metrics)
        st.dataframe(report_df, use_container_width=True, hide_index=True)

        # 交易记录
        st.markdown("### 📝 交易记录")
        if result.trades:
            trades_data = [
                {
                    "日期": (
                        str(t.date.date()) if hasattr(t.date, "date")
                        else str(t.date)
                    ),
                    "方向": "买入" if t.direction == "BUY" else "卖出",
                    "价格": f"¥{t.price:.2f}",
                    "数量": t.shares,
                    "金额": f"¥{t.amount:,.2f}",
                    "手续费": f"¥{t.commission:.2f}",
                }
                for t in result.trades
            ]
            st.dataframe(
                pd.DataFrame(trades_data),
                use_container_width=True, hide_index=True,
            )
        else:
            st.info("该回测期间无交易记录")

    else:
        # 首页欢迎
        st.markdown("""
        ### 欢迎使用 Real-AI-R 量化回测平台！

        **使用方法：**
        1. 在左侧配置股票代码、时间范围
        2. 选择交易策略及参数
        3. 设置初始资金和费用
        4. 点击 **🚀 运行回测** 开始

        **内置策略：**
        - **双均线交叉** — 短期均线上穿长期均线买入，下穿卖出
        - **MACD策略** — 基于MACD金叉/死叉信号
        - **布林带策略** — 价格触及下轨买入，触及上轨卖出

        **数据来源：** AKShare（完全免费，无需注册）
        """)

# ==================================================================
# Tab 2: 板块监控
# ==================================================================
with tab_monitor:
    render_sector_monitor()

# ==================================================================
# Tab 3: 热门板块预测
# ==================================================================
with tab_predict:
    render_sector_prediction()

# ==================================================================
# Tab 4: 个股推荐
# ==================================================================
with tab_recommend:
    render_stock_recommendation()
