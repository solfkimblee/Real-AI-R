"""ML 增强板块预测 Streamlit 页面

包含三个升级功能：
1. ML 融合热门板块预测 — 宏观+量化小模型预测，含个股推荐
2. 日热门板块推荐 — 自动选择热门板块并推荐个股
3. 板块联动分析 — 宏观分类 × 量化因子联动可视化
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
import plotly.express as px

from real_ai_r.macro.classifier import SectorClassifier
from real_ai_r.ml.backtest import ModelBacktester
from real_ai_r.ml.data_collector import BoardHistoryCollector, SnapshotCollector
from real_ai_r.ml.features import FeatureEngineer
from real_ai_r.ml.model import HotBoardModel
from real_ai_r.ml.registry import ModelRegistry
from real_ai_r.sector.recommender import StockRecommender

# ======================================================================
# 辅助函数
# ======================================================================

def _backtest_with_fixed_model(
    model: HotBoardModel,
    feature_df: pd.DataFrame,
    top_n: int,
):
    """使用固定模型对历史数据进行回测（不重新训练）。"""
    from real_ai_r.ml.backtest import BacktestDay, BacktestReport

    if feature_df.empty or "date" not in feature_df.columns:
        return BacktestReport()

    dates = sorted(feature_df["date"].unique())
    if len(dates) < 10:
        return BacktestReport()

    daily_results: list[BacktestDay] = []
    equity = [1.0]

    for i in range(len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]

        today_data = feature_df[feature_df["date"] == current_date]
        if today_data.empty:
            continue

        predictions = model.predict(today_data)
        predicted_hot = [p.board_name for p in predictions[:top_n]]

        next_data = feature_df[feature_df["date"] == next_date]
        if next_data.empty:
            continue

        actual_hot_df = next_data.nlargest(top_n, "momentum_1d")
        actual_hot = actual_hot_df["board_name"].tolist()

        hit_set = set(predicted_hot) & set(actual_hot)
        hit_count = len(hit_set)
        precision = hit_count / len(predicted_hot) if predicted_hot else 0.0

        predicted_returns = next_data[
            next_data["board_name"].isin(predicted_hot)
        ]["momentum_1d"]
        predicted_avg = predicted_returns.mean() if not predicted_returns.empty else 0.0
        market_avg = next_data["momentum_1d"].mean()
        excess = predicted_avg - market_avg

        daily_results.append(BacktestDay(
            date=str(current_date)[:10],
            predicted_hot=predicted_hot,
            actual_hot=actual_hot,
            hit_count=hit_count,
            precision=precision,
            predicted_avg_return=round(predicted_avg, 4),
            market_avg_return=round(market_avg, 4),
            excess_return=round(excess, 4),
        ))

        daily_return = predicted_avg / 100
        equity.append(equity[-1] * (1 + daily_return))

    if not daily_results:
        return BacktestReport()

    import numpy as np
    precisions = [d.precision for d in daily_results]
    excess_returns = [d.excess_return for d in daily_results]
    win_days = sum(1 for e in excess_returns if e > 0)

    peak = equity[0]
    max_dd = 0.0
    for val in equity:
        if val > peak:
            peak = val
        dd = (peak - val) / peak
        if dd > max_dd:
            max_dd = dd

    daily_rets = np.diff(equity) / np.array(equity[:-1])
    sharpe = 0.0
    if len(daily_rets) > 1 and np.std(daily_rets) > 0:
        sharpe = float(np.mean(daily_rets) / np.std(daily_rets) * np.sqrt(252))

    return BacktestReport(
        total_days=len(daily_results),
        avg_precision=float(np.mean(precisions)),
        avg_hit_rate=float(np.mean(precisions)),
        avg_excess_return=float(np.mean(excess_returns)),
        cumulative_return=float((equity[-1] / equity[0] - 1) * 100),
        max_drawdown=float(max_dd * 100),
        sharpe_ratio=round(sharpe, 2),
        win_rate=win_days / len(daily_results) if daily_results else 0.0,
        model_metrics=model.metrics,
        daily_results=daily_results,
        equity_curve=equity,
    )


def _train_and_predict(board_type: str, top_n: int, train_days: int, max_boards: int):
    """训练模型并预测热门板块。"""
    # 1. 采集历史数据
    collector = BoardHistoryCollector(board_type=board_type)
    history = collector.collect_all_boards(days=train_days, max_boards=max_boards)

    if history.empty:
        return None, None, None

    # 2. 特征工程
    engineer = FeatureEngineer()
    feature_df = engineer.build_features_from_history(history)

    if feature_df.empty:
        return None, None, None

    # 3. 训练模型
    model = HotBoardModel()
    metrics = model.train(feature_df)

    # 4. 采集今日截面
    snapshot = SnapshotCollector.collect_today_snapshot(board_type)
    if snapshot.empty:
        return model, metrics, None

    # 5. 构建截面特征
    # 获取每个板块最近的历史数据用于多日特征
    board_histories = {}
    for board_name in snapshot["name"].unique():
        board_hist = history[history["board_name"] == board_name]
        if not board_hist.empty:
            board_histories[board_name] = board_hist

    snapshot_features = engineer.build_features_from_snapshot(
        snapshot, board_histories=board_histories,
    )

    # 6. 预测
    predictions = model.predict(snapshot_features)

    return model, metrics, predictions[:top_n]


# ======================================================================
# 1. ML 融合热门板块预测
# ======================================================================

def render_ml_prediction() -> None:
    """渲染 ML 融合板块预测页。"""
    st.markdown("## 🔮 ML 融合热门板块预测")
    st.info(
        "**升级版预测模型**：结合宏观分类（科技/周期/红线）+ 量化因子（动量/波动/资金流）"
        "训练 LightGBM 小模型，预测明日热门板块并推荐板块内个股\n\n"
        "📊 模型特征：20个量化因子（含周线/月线动量） + 4个宏观标签 + 4个市场环境因子 = 28维特征"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        board_type = st.radio(
            "板块类型",
            ["行业板块", "概念板块"],
            horizontal=True,
            key="ml_board_type",
        )
    with col2:
        top_n = st.slider("预测数量", 5, 20, 10, key="ml_top_n")
    with col3:
        train_days = st.slider("训练天数", 30, 120, 60, key="ml_train_days")

    with st.expander("⚙️ 高级设置", expanded=False):
        max_boards = st.slider(
            "最大采集板块数（0=全部，减少可加速）",
            0, 100, 50, key="ml_max_boards",
        )
        st.caption("⚠️ 采集全部板块历史数据耗时较长（约3-8分钟），建议先用50个板块快速验证")

    predict_btn = st.button("🧠 训练模型 & 预测", type="primary", key="ml_predict_btn")

    if predict_btn:
        bt = "industry" if board_type == "行业板块" else "concept"

        progress = st.progress(0, text="📥 采集板块历史数据...")

        with st.spinner("🧠 正在训练模型...（首次较慢，需采集历史数据）"):
            try:
                # 1. 采集
                progress.progress(10, text="📥 采集板块历史数据...")
                collector = BoardHistoryCollector(board_type=bt)
                history = collector.collect_all_boards(
                    days=train_days, max_boards=max_boards or 0,
                )

                if history.empty:
                    st.error("历史数据采集失败")
                    return

                board_count = history["board_name"].nunique()
                progress.progress(40, text=f"✅ 已采集 {board_count} 个板块历史数据")

                # 2. 特征工程
                progress.progress(50, text="🔧 特征工程计算中...")
                engineer = FeatureEngineer()
                feature_df = engineer.build_features_from_history(history)

                if feature_df.empty:
                    st.error("特征工程失败：数据不足")
                    return

                progress.progress(60, text=f"✅ 构建 {len(feature_df)} 个特征样本")

                # 3. 训练
                progress.progress(70, text="🧠 LightGBM 模型训练中...")
                model = HotBoardModel()
                metrics = model.train(feature_df)

                progress.progress(80, text="✅ 模型训练完成")

                # 4. 采集今日截面
                progress.progress(85, text="📡 采集今日截面数据...")
                snapshot = SnapshotCollector.collect_today_snapshot(bt)

                # 5. 构建截面特征
                board_histories = {}
                for bname in snapshot["name"].unique():
                    bh = history[history["board_name"] == bname]
                    if not bh.empty:
                        board_histories[bname] = bh

                snapshot_features = engineer.build_features_from_snapshot(
                    snapshot, board_histories=board_histories,
                )

                # 6. 预测
                progress.progress(90, text="🔮 生成预测结果...")
                predictions = model.predict(snapshot_features)
                top_predictions = predictions[:top_n]

                progress.progress(95, text="💾 保存模型版本...")
                registry = ModelRegistry()
                saved_version = registry.save_model(
                    model=model,
                    board_type=bt,
                    train_days=train_days,
                    max_boards=max_boards or 0,
                    sample_count=len(feature_df),
                )

                progress.progress(100, text="✅ 预测完成！")

            except Exception as e:
                st.error(f"预测失败: {e}")
                return

        # ---- 模型版本信息 ----
        if saved_version:
            st.success(
                f"💾 模型已保存为版本 **{saved_version.version_id}** "
                f"（可在「📦 模型管理」标签页查看所有版本）"
            )

        # ---- 模型评估指标 ----
        st.markdown("### 📊 模型评估")
        if metrics:
            mc1, mc2, mc3, mc4 = st.columns(4)
            with mc1:
                st.metric("AUC", f"{metrics.auc:.4f}")
            with mc2:
                st.metric("F1 Score", f"{metrics.f1:.4f}")
            with mc3:
                st.metric("精确率", f"{metrics.precision:.2%}")
            with mc4:
                st.metric("召回率", f"{metrics.recall:.2%}")

            # 特征重要性
            fi = model.get_feature_importance(top_n=12)
            if fi:
                st.markdown("#### 🎯 特征重要性 Top 12")
                fi_df = pd.DataFrame(fi, columns=["特征", "重要性"])
                # 翻译特征名
                name_map = {
                    "momentum_1d": "1日动量", "momentum_3d": "3日动量",
                    "momentum_5d": "5日动量", "momentum_10d": "10日动量",
                    "volatility_5d": "5日波动率", "volatility_10d": "10日波动率",
                    "volume_ratio_5d": "量比(5日)", "turnover_rate": "换手率",
                    "amplitude": "振幅", "ma5_bias": "MA5偏离",
                    "ma10_bias": "MA10偏离", "ma20_bias": "MA20偏离",
                    "rsi_14": "RSI(14)", "price_position": "价格位置",
                    "is_tech": "科技主线", "is_cycle": "周期主线",
                    "is_redline": "红线禁区", "cycle_stage": "周期阶段",
                    "market_momentum": "大盘动量", "market_breadth": "市场广度",
                    "net_inflow_rank": "资金排名", "rise_ratio": "上涨占比",
                }
                fi_df["特征中文"] = fi_df["特征"].map(name_map).fillna(fi_df["特征"])
                fig_fi = px.bar(
                    fi_df, x="重要性", y="特征中文", orientation="h",
                    color="重要性", color_continuous_scale="YlOrRd",
                )
                fig_fi.update_layout(
                    height=400, yaxis=dict(autorange="reversed"),
                    margin=dict(t=20, b=30),
                )
                st.plotly_chart(fig_fi, use_container_width=True)

        # ---- 预测结果 ----
        if top_predictions:
            st.markdown("### 🏅 明日热门板块预测（ML模型）")
            st.success(f"模型预测 Top {len(top_predictions)} 热门板块")

            # 预测结果表
            pred_rows = []
            for p in top_predictions:
                pred_rows.append({
                    "板块": p.board_name,
                    "预测评分": round(p.predicted_score, 1),
                    "热门概率": f"{p.hot_probability:.1%}",
                    "宏观分类": p.macro_category,
                    "当日涨幅%": round(p.momentum_1d, 2),
                    "换手率%": round(p.turnover_rate, 2),
                    "关键因子": " | ".join(p.key_factors[:3]),
                })
            pred_df = pd.DataFrame(pred_rows)

            # 评分柱状图
            fig_pred = px.bar(
                pred_df, x="预测评分", y="板块", orientation="h",
                color="宏观分类",
                color_discrete_map={
                    "科技主线": "#FF6B6B", "周期主线": "#4ECDC4",
                    "红线禁区": "#95A5A6", "其他": "#DDDDDD",
                },
                text="预测评分",
            )
            fig_pred.update_traces(texttemplate="%{text:.0f}", textposition="outside")
            fig_pred.update_layout(
                height=max(300, len(pred_df) * 35),
                yaxis=dict(autorange="reversed"),
                margin=dict(t=20, b=30),
            )
            st.plotly_chart(fig_pred, use_container_width=True)

            # 详细数据表
            st.dataframe(pred_df, use_container_width=True, hide_index=True)

            # ---- 板块内个股推荐 ----
            st.markdown("### 💎 热门板块个股推荐")
            st.info("自动获取预测排名前3板块的成分股，综合评分推荐")

            for p in top_predictions[:3]:
                with st.expander(
                    f"📊 {p.board_name} (评分 {p.predicted_score:.0f} | {p.macro_category})",
                    expanded=(p == top_predictions[0]),
                ):
                    try:
                        stocks = StockRecommender.recommend(
                            board_name=p.board_name,
                            board_type=bt,
                            top_n=8,
                            sort_by="composite",
                        )
                        if stocks.empty:
                            st.warning("无有效个股数据")
                        else:
                            display = stocks.copy()
                            col_labels = {
                                "code": "代码", "name": "名称", "price": "最新价",
                                "change_pct": "涨跌幅%", "turnover_rate": "换手率%",
                                "amount": "成交额", "composite_score": "综合评分",
                            }
                            avail = {k: v for k, v in col_labels.items() if k in display.columns}
                            display = display.rename(columns=avail)
                            if "成交额" in display.columns:
                                display["成交额"] = display["成交额"].apply(
                                    lambda x: f"{x / 1e8:.2f}亿"
                                    if pd.notna(x) and abs(x) >= 1e8
                                    else (f"{x / 1e4:.0f}万" if pd.notna(x) else "-")
                                )
                            st.dataframe(display, use_container_width=True, hide_index=True)
                    except Exception as e:
                        st.warning(f"个股数据获取失败: {e}")


# ======================================================================
# 2. 日热门板块推荐
# ======================================================================

def render_daily_hot_boards() -> None:
    """渲染日热门板块推荐页（升级版个股推荐）。"""
    st.markdown("## 💎 今日热门板块 & 个股推荐")
    st.info(
        "**智能板块选择**：自动结合宏观分类、实时行情和资金流向，"
        "筛选今日最值得关注的热门板块，并推荐板块内优质个股\n\n"
        "🗡️ 科技主线优先 | 🛡️ 周期机会关注 | 🚫 红线板块自动排除"
    )

    col1, col2 = st.columns(2)
    with col1:
        board_type = st.radio(
            "板块类型",
            ["行业板块", "概念板块"],
            horizontal=True,
            key="hot_board_type",
        )
    with col2:
        n_boards = st.slider("推荐板块数", 3, 10, 5, key="hot_n_boards")

    filter_mode = st.radio(
        "筛选模式",
        ["综合（宏观+量化）", "仅科技主线", "仅周期主线", "全部（含红线）"],
        horizontal=True,
        key="hot_filter_mode",
    )

    hot_btn = st.button("🔥 获取今日热门", type="primary", key="hot_btn")

    if hot_btn:
        bt = "industry" if board_type == "行业板块" else "concept"

        with st.spinner("🔥 正在分析今日热门板块..."):
            try:
                # 获取截面数据
                snapshot = SnapshotCollector.collect_today_snapshot(bt)
                if snapshot.empty:
                    st.error("板块数据获取失败")
                    return

                # 宏观分类
                classifier = SectorClassifier()
                classified = classifier.classify_dataframe(snapshot)

                # 过滤
                if filter_mode == "仅科技主线":
                    classified = classified[classified["macro_category"] == "tech"]
                elif filter_mode == "仅周期主线":
                    classified = classified[classified["macro_category"] == "cycle"]
                elif filter_mode == "综合（宏观+量化）":
                    # 排除红线
                    classified = classified[classified["macro_category"] != "redline"]

                if classified.empty:
                    st.warning("当前筛选条件下无可用板块")
                    return

                # 综合评分：涨幅 + 换手率 + 宏观加成
                scored = classified.copy()
                # 基础分
                change_rank = scored["change_pct"].rank(pct=True) * 40
                turnover_rank = scored["turnover_rate"].rank(pct=True) * 25

                # 资金流向分
                inflow_rank = pd.Series(0.0, index=scored.index)
                if "net_inflow" in scored.columns:
                    inflow_rank = scored["net_inflow"].rank(pct=True) * 20

                # 宏观加成
                macro_bonus = scored["macro_category"].map({
                    "tech": 15, "cycle": 10, "neutral": 0, "redline": -20,
                }).fillna(0)

                scored["hot_score"] = change_rank + turnover_rank + inflow_rank + macro_bonus
                scored = scored.nlargest(n_boards, "hot_score")

            except Exception as e:
                st.error(f"分析失败: {e}")
                return

        st.success(f"今日推荐 {len(scored)} 个热门板块")

        # ---- 热门板块排行 ----
        st.markdown("### 🏆 今日热门板块")
        hot_display = scored[["name", "change_pct", "turnover_rate",
                              "macro_icon", "macro_display", "hot_score"]].copy()
        hot_display.columns = ["板块", "涨跌幅%", "换手率%", "标签", "宏观分类", "热度评分"]
        hot_display["热度评分"] = hot_display["热度评分"].round(1)

        fig_hot = px.bar(
            hot_display, x="热度评分", y="板块", orientation="h",
            color="宏观分类",
            color_discrete_map={
                "🗡️ 科技主线": "#FF6B6B", "🛡️ 周期主线": "#4ECDC4",
                "⚪ 其他": "#DDDDDD",
            },
            text="热度评分",
        )
        fig_hot.update_traces(texttemplate="%{text:.0f}", textposition="outside")
        fig_hot.update_layout(
            height=max(300, len(hot_display) * 45),
            yaxis=dict(autorange="reversed"),
            margin=dict(t=20, b=30),
        )
        st.plotly_chart(fig_hot, use_container_width=True)
        st.dataframe(hot_display, use_container_width=True, hide_index=True)

        # ---- 各板块个股推荐 ----
        st.markdown("### 💎 板块内个股推荐")

        for _, row in scored.iterrows():
            board_name = row["name"]
            macro_tag = f"{row['macro_icon']} {row['macro_display']}"
            change = row["change_pct"]

            with st.expander(
                f"📊 {board_name} | {macro_tag} | 涨幅 {change:+.2f}%",
                expanded=(_ == scored.index[0]),
            ):
                try:
                    stocks = StockRecommender.recommend(
                        board_name=board_name,
                        board_type=bt,
                        top_n=10,
                        sort_by="composite",
                    )
                    if stocks.empty:
                        st.warning("无有效个股数据")
                        continue

                    # 显示表格
                    display = stocks.copy()
                    col_labels = {
                        "code": "代码", "name": "名称", "price": "最新价",
                        "change_pct": "涨跌幅%", "turnover_rate": "换手率%",
                        "amount": "成交额", "pe_dynamic": "市盈率",
                        "composite_score": "综合评分",
                    }
                    avail = {k: v for k, v in col_labels.items() if k in display.columns}
                    display = display.rename(columns=avail)
                    if "成交额" in display.columns:
                        display["成交额"] = display["成交额"].apply(
                            lambda x: f"{x / 1e8:.2f}亿"
                            if pd.notna(x) and abs(x) >= 1e8
                            else (f"{x / 1e4:.0f}万" if pd.notna(x) else "-")
                        )
                    st.dataframe(display, use_container_width=True, hide_index=True)

                    # 个股涨跌幅柱状图
                    if "change_pct" in stocks.columns and len(stocks) > 1:
                        fig_stock = px.bar(
                            stocks.head(10), x="change_pct", y="name", orientation="h",
                            color="change_pct",
                            color_continuous_scale=["green", "yellow", "red"],
                            labels={"change_pct": "涨跌幅 (%)", "name": "个股"},
                        )
                        fig_stock.update_layout(
                            height=max(250, len(stocks.head(10)) * 30),
                            yaxis=dict(autorange="reversed"),
                            margin=dict(t=20, b=30),
                        )
                        st.plotly_chart(fig_stock, use_container_width=True)

                except Exception as e:
                    st.warning(f"数据获取失败: {e}")


# ======================================================================
# 3. 板块联动分析
# ======================================================================

def render_board_linkage() -> None:
    """渲染板块联动分析页。"""
    st.markdown("## 🔗 板块联动分析")
    st.info(
        "**宏观 × 量化联动**：展示板块分类与市场表现的多维联动关系\n\n"
        "发现科技/周期/红线三大类别的实时强弱对比，辅助判断大盘风格切换"
    )

    col_type, col_btn = st.columns([3, 1])
    with col_type:
        board_type = st.radio(
            "板块类型",
            ["行业板块", "概念板块"],
            horizontal=True,
            key="link_board_type",
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        st.button("🔄 刷新", key="link_refresh")

    bt = "industry" if board_type == "行业板块" else "concept"

    with st.spinner("正在分析板块联动..."):
        try:
            snapshot = SnapshotCollector.collect_today_snapshot(bt)
            if snapshot.empty:
                st.error("数据获取失败")
                return

            classifier = SectorClassifier()
            classified = classifier.classify_dataframe(snapshot)
            summary = classifier.get_category_summary(classified)
        except Exception as e:
            st.error(f"分析失败: {e}")
            return

    # ---- 三大类别强弱对比 ----
    st.markdown("### ⚖️ 宏观主线强弱对比")
    cat_data = []
    for cat in ["tech", "cycle", "redline", "neutral"]:
        info = summary.get(cat, {})
        subset = classified[classified["macro_category"] == cat]
        cat_data.append({
            "分类": info.get("display_name", cat),
            "板块数": info.get("count", 0),
            "平均涨幅%": round(info.get("avg_change", 0), 2),
            "平均换手率%": round(subset["turnover_rate"].mean(), 2) if not subset.empty else 0,
            "上涨占比%": round(
                (subset["change_pct"] > 0).mean() * 100, 1
            ) if not subset.empty else 0,
        })

    cat_df = pd.DataFrame(cat_data)
    st.dataframe(cat_df, use_container_width=True, hide_index=True)

    # 对比柱状图
    col_bar1, col_bar2 = st.columns(2)
    with col_bar1:
        fig_change = px.bar(
            cat_df, x="分类", y="平均涨幅%",
            color="平均涨幅%",
            color_continuous_scale=["green", "yellow", "red"],
            text="平均涨幅%",
            title="各分类平均涨幅",
        )
        fig_change.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig_change.update_layout(height=350, margin=dict(t=40, b=30))
        st.plotly_chart(fig_change, use_container_width=True)

    with col_bar2:
        fig_breadth = px.bar(
            cat_df, x="分类", y="上涨占比%",
            color="上涨占比%",
            color_continuous_scale=["#FF6B6B", "#FFC107", "#4CAF50"],
            text="上涨占比%",
            title="各分类上涨板块占比",
        )
        fig_breadth.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_breadth.update_layout(height=350, margin=dict(t=40, b=30))
        st.plotly_chart(fig_breadth, use_container_width=True)

    # ---- 散点图：涨幅 vs 换手率（分类着色）----
    st.markdown("### 🎯 板块分布图（涨幅 × 换手率）")
    scatter_df = classified[["name", "change_pct", "turnover_rate", "macro_display"]].copy()
    scatter_df.columns = ["板块", "涨跌幅%", "换手率%", "宏观分类"]

    fig_scatter = px.scatter(
        scatter_df, x="涨跌幅%", y="换手率%",
        color="宏观分类",
        color_discrete_map={
            "🗡️ 科技主线": "#FF6B6B", "🛡️ 周期主线": "#4ECDC4",
            "🚫 红线禁区": "#95A5A6", "其他": "#DDDDDD",
        },
        hover_name="板块",
        title="板块涨跌幅 vs 换手率分布",
    )
    fig_scatter.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
    fig_scatter.add_hline(
        y=classified["turnover_rate"].median(),
        line_dash="dash", line_color="gray", opacity=0.3,
    )
    fig_scatter.update_layout(height=500, margin=dict(t=40, b=30))
    st.plotly_chart(fig_scatter, use_container_width=True)

    # ---- 资金流向热力图 ----
    if "net_inflow" in classified.columns:
        st.markdown("### 💰 资金流向联动")

        # 按分类汇总资金流向
        fund_by_cat = classified.groupby("macro_display").agg(
            总净流入=("net_inflow", "sum"),
            平均净流入=("net_inflow", "mean"),
            板块数=("name", "count"),
        ).reset_index()
        fund_by_cat.columns = ["宏观分类", "总净流入", "平均净流入", "板块数"]
        fund_by_cat["总净流入(亿)"] = (fund_by_cat["总净流入"] / 1e8).round(2)
        fund_by_cat["平均净流入(万)"] = (fund_by_cat["平均净流入"] / 1e4).round(0)

        fig_fund = px.bar(
            fund_by_cat, x="宏观分类", y="总净流入(亿)",
            color="总净流入(亿)",
            color_continuous_scale=["red", "white", "green"],
            text="总净流入(亿)",
            title="各分类资金净流入",
        )
        fig_fund.update_traces(texttemplate="%{text:.2f}亿", textposition="outside")
        fig_fund.update_layout(height=400, margin=dict(t=40, b=30))
        st.plotly_chart(fig_fund, use_container_width=True)

    # ---- 风格轮动信号 ----
    st.markdown("### 🔄 风格轮动信号")
    tech_avg = summary.get("tech", {}).get("avg_change", 0)
    cycle_avg = summary.get("cycle", {}).get("avg_change", 0)
    diff = tech_avg - cycle_avg

    if diff > 1.5:
        signal = "🗡️ **强科技风格** — 科技主线大幅领先周期，进攻风格占优"
        signal_color = "#FF6B6B"
    elif diff > 0.5:
        signal = "🗡️ 偏科技风格 — 科技主线小幅领先，但差距不大"
        signal_color = "#FFB347"
    elif diff > -0.5:
        signal = "⚖️ **均衡风格** — 科技与周期势均力敌，关注切换信号"
        signal_color = "#FFC107"
    elif diff > -1.5:
        signal = "🛡️ 偏周期风格 — 周期防御略强，注意降低科技仓位"
        signal_color = "#87CEEB"
    else:
        signal = "🛡️ **强周期风格** — 周期主线大幅领先，防御为主"
        signal_color = "#4ECDC4"

    st.markdown(
        f"<div style='padding:15px; border-radius:10px; border:2px solid {signal_color}; "
        f"background-color:{signal_color}22'>"
        f"<h4 style='margin:0'>今日风格: {signal}</h4>"
        f"<p style='margin:5px 0 0 0'>科技主线平均涨幅: <b>{tech_avg:.2f}%</b> | "
        f"周期主线平均涨幅: <b>{cycle_avg:.2f}%</b> | "
        f"差值: <b>{diff:+.2f}%</b></p></div>",
        unsafe_allow_html=True,
    )


# ======================================================================
# 4. 模型回测报告
# ======================================================================

def render_model_backtest() -> None:
    """渲染模型回测页面。"""
    st.markdown("## 📈 模型回测验证")
    st.info(
        "**滚动窗口回测**：模拟真实使用场景，每天用过去 N 天数据训练，预测次日热门板块\n\n"
        "评估模型的命中率、超额收益和稳定性。支持选择已有模型版本进行回测。"
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        board_type = st.radio(
            "板块类型",
            ["行业板块", "概念板块"],
            horizontal=True,
            key="bt_board_type",
        )
    with col2:
        train_window = st.slider("训练窗口(天)", 20, 60, 30, key="bt_train_window")
    with col3:
        top_n = st.slider("每日预测板块数", 5, 20, 10, key="bt_top_n")

    # 模型版本选择
    bt_type = "industry" if board_type == "行业板块" else "concept"
    registry = ModelRegistry()
    versions = registry.list_versions(board_type=bt_type)

    model_options = ["🔄 重新训练（滚动窗口）"]
    version_map: dict[str, str] = {}
    for v in versions:
        label = (
            f"📦 {v.version_id} | AUC={v.auc:.4f} | "
            f"训练{v.train_days}天 | {v.sample_count}样本"
        )
        model_options.append(label)
        version_map[label] = v.version_id

    selected_model = st.selectbox(
        "选择模型",
        model_options,
        key="bt_model_select",
        help="选择已保存的模型版本进行回测，或重新训练",
    )

    use_saved_model = selected_model != "🔄 重新训练（滚动窗口）"

    with st.expander("⚙️ 高级设置", expanded=False):
        total_days = st.slider("回测总天数", 40, 120, 60, key="bt_total_days")
        max_boards = st.slider(
            "采集板块数（0=全部）", 0, 100, 30, key="bt_max_boards",
        )
        if not use_saved_model:
            retrain_every = st.slider("重训间隔(天)", 1, 10, 5, key="bt_retrain")
        else:
            retrain_every = 5
        st.caption("⚠️ 回测需要大量数据采集，可能需要 3-10 分钟")

    backtest_btn = st.button("📊 开始回测", type="primary", key="bt_run_btn")

    if backtest_btn:
        bt = bt_type

        with st.spinner("📊 正在执行模型回测..."):
            try:
                # 采集数据
                st.text("📥 采集板块历史数据...")
                collector = BoardHistoryCollector(board_type=bt)
                history = collector.collect_all_boards(
                    days=total_days, max_boards=max_boards or 0,
                )

                if history.empty:
                    st.error("历史数据采集失败")
                    return

                # 特征工程
                st.text("🔧 特征工程...")
                engineer = FeatureEngineer()
                feature_df = engineer.build_features_from_history(history)

                if feature_df.empty:
                    st.error("特征工程失败")
                    return

                # 执行回测
                st.text("📊 回测中...")
                if use_saved_model:
                    # 使用已保存的模型进行固定模型回测
                    version_id = version_map[selected_model]
                    loaded_model = registry.load_model(version_id)
                    if loaded_model is None:
                        st.error(f"模型加载失败: {version_id}")
                        return
                    st.text(f"📦 使用已保存模型 {version_id} 进行回测...")
                    report = _backtest_with_fixed_model(
                        loaded_model, feature_df, top_n,
                    )
                else:
                    backtester = ModelBacktester(
                        train_window=train_window,
                        top_n=top_n,
                        retrain_every=retrain_every,
                    )
                    report = backtester.run(feature_df)

            except Exception as e:
                st.error(f"回测失败: {e}")
                return

        if report.total_days == 0:
            st.warning("回测数据不足，请增加历史天数或板块数")
            return

        # ---- 回测摘要 ----
        st.markdown("### 📊 回测结果摘要")
        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.metric("回测天数", report.total_days)
        with rc2:
            st.metric("平均命中率", f"{report.avg_precision:.1%}")
        with rc3:
            st.metric("累计收益率", f"{report.cumulative_return:.2f}%")
        with rc4:
            st.metric("夏普比率", f"{report.sharpe_ratio:.2f}")

        rc5, rc6, rc7, rc8 = st.columns(4)
        with rc5:
            st.metric("平均超额收益", f"{report.avg_excess_return:.2f}%")
        with rc6:
            st.metric("最大回撤", f"{report.max_drawdown:.2f}%")
        with rc7:
            st.metric("胜率", f"{report.win_rate:.1%}")
        with rc8:
            model_auc = report.model_metrics.auc if report.model_metrics else 0
            st.metric("模型AUC", f"{model_auc:.4f}")

        # ---- 净值曲线 ----
        if report.equity_curve:
            st.markdown("### 📈 策略净值曲线")
            eq_df = pd.DataFrame({
                "交易日": range(len(report.equity_curve)),
                "净值": report.equity_curve,
            })
            fig_eq = px.line(
                eq_df, x="交易日", y="净值",
                title="模型预测策略净值曲线",
            )
            fig_eq.add_hline(y=1.0, line_dash="dash", line_color="gray")
            fig_eq.update_layout(height=400, margin=dict(t=40, b=30))
            st.plotly_chart(fig_eq, use_container_width=True)

        # ---- 每日命中率 ----
        if report.daily_results:
            st.markdown("### 🎯 每日命中率")
            daily_df = pd.DataFrame([
                {
                    "日期": d.date,
                    "命中数": d.hit_count,
                    "精确率": round(d.precision * 100, 1),
                    "预测收益%": d.predicted_avg_return,
                    "市场收益%": d.market_avg_return,
                    "超额收益%": d.excess_return,
                }
                for d in report.daily_results
            ])

            fig_hit = px.bar(
                daily_df, x="日期", y="精确率",
                color="超额收益%",
                color_continuous_scale=["red", "white", "green"],
                title="每日命中率与超额收益",
            )
            fig_hit.update_layout(height=400, margin=dict(t=40, b=30))
            st.plotly_chart(fig_hit, use_container_width=True)

            # 详细数据表
            st.markdown("### 📋 每日回测明细")
            st.dataframe(daily_df, use_container_width=True, hide_index=True, height=300)


# ======================================================================
# 5. 模型版本管理
# ======================================================================

def render_model_management() -> None:
    """渲染模型版本管理页面。"""
    st.markdown("## 📦 模型版本管理")
    st.info(
        "**模型注册表**：管理所有已训练的 LightGBM 模型版本\n\n"
        "查看版本详情、对比模型性能、加载已有模型进行预测、删除旧版本"
    )

    registry = ModelRegistry()

    # ---- 版本概览 ----
    all_versions = registry.list_versions()

    if not all_versions:
        st.warning(
            "暂无已保存的模型版本。请先到「🧠 ML板块预测」标签页训练并保存模型。"
        )
        return

    st.markdown(f"### 📋 已保存 {len(all_versions)} 个模型版本")

    # 筛选
    col_filter, col_sort = st.columns(2)
    with col_filter:
        filter_type = st.radio(
            "筛选板块类型",
            ["全部", "行业板块", "概念板块"],
            horizontal=True,
            key="mgmt_filter_type",
        )
    with col_sort:
        sort_by = st.radio(
            "排序方式",
            ["创建时间", "AUC", "F1"],
            horizontal=True,
            key="mgmt_sort",
        )

    filtered = all_versions
    if filter_type == "行业板块":
        filtered = [v for v in filtered if v.board_type == "industry"]
    elif filter_type == "概念板块":
        filtered = [v for v in filtered if v.board_type == "concept"]

    if sort_by == "AUC":
        filtered = sorted(filtered, key=lambda v: v.auc, reverse=True)
    elif sort_by == "F1":
        filtered = sorted(filtered, key=lambda v: v.f1, reverse=True)

    # 版本列表表格
    rows = []
    for v in filtered:
        bt_display = "行业" if v.board_type == "industry" else "概念"
        rows.append({
            "版本ID": v.version_id,
            "创建时间": v.created_at[:19].replace("T", " "),
            "板块类型": bt_display,
            "训练天数": v.train_days,
            "样本数": v.sample_count,
            "特征数": v.feature_count,
            "AUC": round(v.auc, 4),
            "F1": round(v.f1, 4),
            "精确率": round(v.precision, 4),
            "召回率": round(v.recall, 4),
            "备注": v.note,
        })

    if rows:
        versions_df = pd.DataFrame(rows)
        st.dataframe(versions_df, use_container_width=True, hide_index=True)

    # ---- 版本对比 ----
    if len(filtered) >= 2:
        st.markdown("### 📊 版本性能对比")
        compare_options = [v.version_id for v in filtered]
        selected_ids = st.multiselect(
            "选择要对比的版本（至少2个）",
            compare_options,
            default=compare_options[:min(3, len(compare_options))],
            key="mgmt_compare",
        )

        if len(selected_ids) >= 2:
            compare_data = registry.compare_versions(selected_ids)
            compare_df = pd.DataFrame(compare_data)
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

            # AUC 对比柱状图
            fig_compare = px.bar(
                compare_df, x="版本", y=["AUC", "F1", "精确率", "召回率"],
                barmode="group",
                title="模型版本性能对比",
                color_discrete_sequence=["#FF6B6B", "#4ECDC4", "#FFB347", "#87CEEB"],
            )
            fig_compare.update_layout(height=400, margin=dict(t=40, b=30))
            st.plotly_chart(fig_compare, use_container_width=True)

    # ---- 版本详情与操作 ----
    st.markdown("### 🔍 版本详情与操作")
    version_labels = {
        v.version_id: (
            f"{v.version_id} | "
            f"{'行业' if v.board_type == 'industry' else '概念'} | "
            f"AUC={v.auc:.4f}"
        )
        for v in filtered
    }
    selected_version = st.selectbox(
        "选择版本查看详情",
        list(version_labels.keys()),
        format_func=lambda x: version_labels.get(x, x),
        key="mgmt_detail_select",
    )

    if selected_version:
        v = registry.get_version(selected_version)
        if v:
            col_detail1, col_detail2 = st.columns(2)
            with col_detail1:
                st.markdown("**基本信息**")
                st.write(f"- 版本ID: `{v.version_id}`")
                st.write(f"- 创建时间: {v.created_at[:19].replace('T', ' ')}")
                bt_display = "行业板块" if v.board_type == "industry" else "概念板块"
                st.write(f"- 板块类型: {bt_display}")
                st.write(f"- 训练天数: {v.train_days}")
                st.write(f"- 采集板块数: {v.max_boards}")
                st.write(f"- 样本数: {v.sample_count}")
                st.write(f"- 特征维度: {v.feature_count}")
            with col_detail2:
                st.markdown("**性能指标**")
                st.write(f"- AUC: **{v.auc:.4f}**")
                st.write(f"- F1: **{v.f1:.4f}**")
                st.write(f"- 精确率: **{v.precision:.4f}**")
                st.write(f"- 召回率: **{v.recall:.4f}**")
                st.write(f"- 准确率: **{v.accuracy:.4f}**")

            # 模型参数
            if v.params:
                with st.expander("⚙️ 模型参数", expanded=False):
                    params_df = pd.DataFrame(
                        [{"参数": k, "值": str(val)} for k, val in v.params.items()],
                    )
                    st.dataframe(params_df, use_container_width=True, hide_index=True)

            # 特征列表
            if v.feature_columns:
                with st.expander(f"📋 特征列表 ({v.feature_count}个)", expanded=False):
                    name_map = {
                        "momentum_1d": "1日动量", "momentum_3d": "3日动量",
                        "momentum_5d": "5日动量", "momentum_10d": "10日动量",
                        "volatility_5d": "5日波动率", "volatility_10d": "10日波动率",
                        "volume_ratio_5d": "量比(5日)", "turnover_rate": "换手率",
                        "amplitude": "振幅", "ma5_bias": "MA5偏离",
                        "ma10_bias": "MA10偏离", "ma20_bias": "MA20偏离",
                        "rsi_14": "RSI(14)", "price_position": "价格位置",
                        "is_tech": "科技主线", "is_cycle": "周期主线",
                        "is_redline": "红线禁区", "cycle_stage": "周期阶段",
                        "market_momentum": "大盘动量", "market_breadth": "市场广度",
                        "net_inflow_rank": "资金排名", "rise_ratio": "上涨占比",
                    }
                    feat_rows = [
                        {"特征": f, "中文名": name_map.get(f, f)}
                        for f in v.feature_columns
                    ]
                    st.dataframe(
                        pd.DataFrame(feat_rows),
                        use_container_width=True, hide_index=True,
                    )

            # 删除操作
            st.markdown("---")
            col_del, col_confirm = st.columns([3, 1])
            with col_del:
                st.caption("⚠️ 删除操作不可恢复")
            with col_confirm:
                if st.button(
                    f"🗑️ 删除 {selected_version}",
                    type="secondary",
                    key=f"del_{selected_version}",
                ):
                    registry.delete_version(selected_version)
                    st.success(f"已删除版本 {selected_version}")
                    st.rerun()
