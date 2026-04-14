"""特征工程模块

将宏观状态和量化因子融合为统一特征向量，供 ML 模型使用。

特征分三组：
1. 量化技术特征：动量、波动率、换手率、成交额、振幅等
2. 宏观标签特征：板块分类 (one-hot)、周期阶段、科技赛道热度
3. 市场环境特征：大盘涨跌、板块轮动强度、资金流向

所有特征均标准化到模型友好的数值范围。
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from real_ai_r.macro.classifier import SectorClassifier

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """板块特征工程器。

    将板块历史行情 + 宏观标签融合为训练/预测特征矩阵。
    """

    # 量化特征列表
    QUANT_FEATURES = [
        "momentum_1d",      # 1日涨跌幅
        "momentum_3d",      # 3日累计涨跌幅
        "momentum_5d",      # 5日累计涨跌幅
        "momentum_10d",     # 10日累计涨跌幅
        "volatility_5d",    # 5日波动率
        "volatility_10d",   # 10日波动率
        "volume_ratio_5d",  # 成交量/5日均量
        "turnover_rate",    # 换手率
        "amplitude",        # 振幅
        "ma5_bias",         # 收盘价偏离MA5的百分比
        "ma10_bias",        # 收盘价偏离MA10的百分比
        "ma20_bias",        # 收盘价偏离MA20的百分比
        "rsi_14",           # 14日RSI
        "price_position",   # 价格在20日高低点中的位置 (0-1)
    ]

    # 宏观特征列表
    MACRO_FEATURES = [
        "is_tech",           # 科技主线
        "is_cycle",          # 周期主线
        "is_redline",        # 红线禁区
        "cycle_stage",       # 周期阶段 (0=非周期, 1-5)
    ]

    # 市场环境特征
    MARKET_FEATURES = [
        "market_momentum",   # 大盘当日涨跌幅
        "market_breadth",    # 上涨板块占比
        "net_inflow_rank",   # 资金净流入排名百分位
        "rise_ratio",        # 板块内上涨家数占比
    ]

    ALL_FEATURES = QUANT_FEATURES + MACRO_FEATURES + MARKET_FEATURES

    def __init__(self) -> None:
        self.classifier = SectorClassifier()

    def build_features_from_history(
        self,
        history_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """从历史数据构建特征矩阵（用于训练）。

        Parameters
        ----------
        history_df : pd.DataFrame
            板块历史数据，需含列：date, board_name, close, volume, change_pct,
            turnover_rate, amplitude, high, low

        Returns
        -------
        pd.DataFrame
            含所有特征列和 target 列的 DataFrame。
        """
        if history_df.empty:
            return pd.DataFrame()

        all_features = []
        for board_name, board_df in history_df.groupby("board_name"):
            board_df = board_df.sort_values("date").reset_index(drop=True)
            features = self._compute_board_features(board_name, board_df)
            if not features.empty:
                all_features.append(features)

        if not all_features:
            return pd.DataFrame()

        result = pd.concat(all_features, ignore_index=True)

        # 添加市场环境特征（截面级）
        result = self._add_market_features(result)

        # 添加 target：次日涨跌幅
        result = self._add_target(result)

        return result

    def build_features_from_snapshot(
        self,
        snapshot_df: pd.DataFrame,
        board_histories: dict[str, pd.DataFrame] | None = None,
    ) -> pd.DataFrame:
        """从当日截面数据构建预测特征（用于推理）。

        Parameters
        ----------
        snapshot_df : pd.DataFrame
            当日板块截面数据，需含列：name, change_pct, turnover_rate, rise_count,
            fall_count, net_inflow
        board_histories : dict
            可选的板块历史数据字典 {board_name: history_df}，用于计算多日特征。

        Returns
        -------
        pd.DataFrame
            含所有特征列的 DataFrame。
        """
        if snapshot_df.empty:
            return pd.DataFrame()

        rows = []
        for _, row in snapshot_df.iterrows():
            board_name = row.get("name", "")
            feat = {"board_name": board_name}

            # 1日动量
            feat["momentum_1d"] = float(row.get("change_pct", 0) or 0)
            feat["turnover_rate"] = float(row.get("turnover_rate", 0) or 0)
            feat["amplitude"] = float(row.get("amplitude", 0) if "amplitude" in row.index else 0)

            # 多日特征从历史数据计算
            if board_histories and board_name in board_histories:
                hist = board_histories[board_name]
                if not hist.empty and "close" in hist.columns:
                    closes = hist["close"].values
                    volumes = hist["volume"].values if "volume" in hist.columns else None
                    feat.update(self._multi_day_features(closes, volumes))

            # 填充缺失的多日特征
            for f in self.QUANT_FEATURES:
                if f not in feat:
                    feat[f] = 0.0

            # 宏观标签
            label = self.classifier.classify(board_name)
            feat["is_tech"] = 1 if label.category == "tech" else 0
            feat["is_cycle"] = 1 if label.category == "cycle" else 0
            feat["is_redline"] = 1 if label.category == "redline" else 0
            feat["cycle_stage"] = self._get_cycle_stage(label)

            # 市场特征
            total = float(row.get("rise_count", 0) or 0) + float(row.get("fall_count", 0) or 0)
            feat["rise_ratio"] = (
                float(row.get("rise_count", 0) or 0) / total if total > 0 else 0.5
            )
            feat["net_inflow_rank"] = 0.5  # 截面中稍后统一计算

            rows.append(feat)

        result = pd.DataFrame(rows)

        # 统一计算截面级特征
        if not result.empty:
            avg_change = result["momentum_1d"].mean()
            result["market_momentum"] = avg_change
            total_boards = len(result)
            up_boards = (result["momentum_1d"] > 0).sum()
            result["market_breadth"] = up_boards / total_boards if total_boards > 0 else 0.5

            if "net_inflow" in snapshot_df.columns:
                result["net_inflow_rank"] = snapshot_df["net_inflow"].rank(pct=True).values

        return result

    def _compute_board_features(
        self,
        board_name: str,
        board_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """计算单个板块的时序特征。"""
        if len(board_df) < 20:
            return pd.DataFrame()

        records = []
        closes = board_df["close"].values
        volumes = board_df["volume"].values if "volume" in board_df.columns else None
        changes = board_df["change_pct"].values if "change_pct" in board_df.columns else None
        turnovers = (
            board_df["turnover_rate"].values
            if "turnover_rate" in board_df.columns else None
        )
        amplitudes = (
            board_df["amplitude"].values
            if "amplitude" in board_df.columns else None
        )
        highs = board_df["high"].values if "high" in board_df.columns else None
        lows = board_df["low"].values if "low" in board_df.columns else None

        # 宏观标签（静态）
        label = self.classifier.classify(board_name)
        is_tech = 1 if label.category == "tech" else 0
        is_cycle = 1 if label.category == "cycle" else 0
        is_redline = 1 if label.category == "redline" else 0
        cycle_stage = self._get_cycle_stage(label)

        for i in range(20, len(board_df)):
            feat: dict = {
                "date": board_df.iloc[i]["date"],
                "board_name": board_name,
            }

            # 动量
            feat["momentum_1d"] = (
                float(changes[i]) if changes is not None else 0.0
            )
            feat["momentum_3d"] = (
                float(np.sum(changes[i - 2:i + 1]))
                if changes is not None else 0.0
            )
            feat["momentum_5d"] = (
                float(np.sum(changes[i - 4:i + 1]))
                if changes is not None else 0.0
            )
            feat["momentum_10d"] = (
                float(np.sum(changes[i - 9:i + 1]))
                if changes is not None else 0.0
            )

            # 波动率
            if changes is not None:
                feat["volatility_5d"] = float(np.std(changes[i - 4:i + 1]))
                feat["volatility_10d"] = float(np.std(changes[i - 9:i + 1]))
            else:
                feat["volatility_5d"] = 0.0
                feat["volatility_10d"] = 0.0

            # 量比
            if volumes is not None:
                avg_vol_5 = np.mean(volumes[i - 5:i])
                feat["volume_ratio_5d"] = (
                    float(volumes[i] / avg_vol_5) if avg_vol_5 > 0 else 1.0
                )
            else:
                feat["volume_ratio_5d"] = 1.0

            # 换手率、振幅
            feat["turnover_rate"] = float(turnovers[i]) if turnovers is not None else 0.0
            feat["amplitude"] = float(amplitudes[i]) if amplitudes is not None else 0.0

            # MA偏离
            ma5 = np.mean(closes[i - 4:i + 1])
            ma10 = np.mean(closes[i - 9:i + 1])
            ma20 = np.mean(closes[i - 19:i + 1])
            c = closes[i]
            feat["ma5_bias"] = float((c - ma5) / ma5 * 100) if ma5 > 0 else 0.0
            feat["ma10_bias"] = float((c - ma10) / ma10 * 100) if ma10 > 0 else 0.0
            feat["ma20_bias"] = float((c - ma20) / ma20 * 100) if ma20 > 0 else 0.0

            # RSI-14
            if changes is not None and i >= 14:
                recent = changes[i - 13:i + 1]
                gains = np.where(recent > 0, recent, 0)
                losses = np.where(recent < 0, -recent, 0)
                avg_gain = np.mean(gains)
                avg_loss = np.mean(losses)
                if avg_loss > 0:
                    rs = avg_gain / avg_loss
                    feat["rsi_14"] = float(100 - 100 / (1 + rs))
                else:
                    feat["rsi_14"] = 100.0
            else:
                feat["rsi_14"] = 50.0

            # 价格位置
            if highs is not None and lows is not None:
                high_20 = np.max(highs[i - 19:i + 1])
                low_20 = np.min(lows[i - 19:i + 1])
                rng = high_20 - low_20
                feat["price_position"] = float((c - low_20) / rng) if rng > 0 else 0.5
            else:
                feat["price_position"] = 0.5

            # 宏观标签
            feat["is_tech"] = is_tech
            feat["is_cycle"] = is_cycle
            feat["is_redline"] = is_redline
            feat["cycle_stage"] = cycle_stage

            records.append(feat)

        return pd.DataFrame(records)

    def _add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加截面级市场环境特征。"""
        if df.empty:
            return df

        for date, group in df.groupby("date"):
            avg_momentum = group["momentum_1d"].mean()
            total = len(group)
            up_count = (group["momentum_1d"] > 0).sum()

            df.loc[group.index, "market_momentum"] = avg_momentum
            df.loc[group.index, "market_breadth"] = up_count / total if total > 0 else 0.5

        # 资金净流入排名暂用 0.5（历史数据中无资金流向数据）
        if "net_inflow_rank" not in df.columns:
            df["net_inflow_rank"] = 0.5

        # 板块内上涨家数占比暂用 0.5（历史数据中无此数据）
        if "rise_ratio" not in df.columns:
            df["rise_ratio"] = 0.5

        return df

    def _add_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加预测目标：次日涨跌幅及是否热门。"""
        if df.empty:
            return df

        df = df.sort_values(["board_name", "date"]).reset_index(drop=True)

        # 次日涨跌幅
        df["next_day_change"] = df.groupby("board_name")["momentum_1d"].shift(-1)

        # 去除最后一天（无次日数据）
        df = df.dropna(subset=["next_day_change"]).reset_index(drop=True)

        # 是否热门：在当日所有板块中，次日涨幅排名前 20%
        df["is_hot_next_day"] = 0
        for date, group in df.groupby("date"):
            threshold = group["next_day_change"].quantile(0.80)
            hot_idx = group[group["next_day_change"] >= threshold].index
            df.loc[hot_idx, "is_hot_next_day"] = 1

        return df

    @staticmethod
    def _multi_day_features(closes: np.ndarray, volumes: np.ndarray | None) -> dict:
        """从历史收盘价序列计算多日特征。"""
        feat: dict = {}
        n = len(closes)

        # Use arithmetic sum of daily returns to match training path
        daily_changes = np.diff(closes) / closes[:-1] * 100 if n >= 2 else np.array([])
        if n >= 3:
            feat["momentum_3d"] = float(np.sum(daily_changes[-3:]))
        if n >= 5:
            feat["momentum_5d"] = float(np.sum(daily_changes[-5:]))
        if n >= 10:
            feat["momentum_10d"] = float(np.sum(daily_changes[-10:]))

        if n >= 5:
            returns = np.diff(closes[-5:]) / closes[-5:-1] * 100
            feat["volatility_5d"] = float(np.std(returns))
        if n >= 10:
            returns = np.diff(closes[-10:]) / closes[-10:-1] * 100
            feat["volatility_10d"] = float(np.std(returns))

        if volumes is not None and n >= 5:
            avg_vol = np.mean(volumes[-6:-1])
            feat["volume_ratio_5d"] = float(volumes[-1] / avg_vol) if avg_vol > 0 else 1.0

        if n >= 5:
            ma5 = np.mean(closes[-5:])
            feat["ma5_bias"] = float((closes[-1] - ma5) / ma5 * 100) if ma5 > 0 else 0.0
        if n >= 10:
            ma10 = np.mean(closes[-10:])
            feat["ma10_bias"] = float((closes[-1] - ma10) / ma10 * 100) if ma10 > 0 else 0.0
        if n >= 20:
            ma20 = np.mean(closes[-20:])
            feat["ma20_bias"] = float((closes[-1] - ma20) / ma20 * 100) if ma20 > 0 else 0.0

            high_20 = np.max(closes[-20:])
            low_20 = np.min(closes[-20:])
            rng = high_20 - low_20
            feat["price_position"] = float((closes[-1] - low_20) / rng) if rng > 0 else 0.5

        if n >= 14:
            daily_returns = np.diff(closes[-14:]) / closes[-14:-1] * 100
            gains = np.where(daily_returns > 0, daily_returns, 0)
            losses = np.where(daily_returns < 0, -daily_returns, 0)
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                feat["rsi_14"] = float(100 - 100 / (1 + rs))
            else:
                feat["rsi_14"] = 100.0

        return feat

    @staticmethod
    def _get_cycle_stage(label) -> int:
        """从 MacroLabel 获取周期阶段编号。"""
        from real_ai_r.macro.classifier import CYCLE_STAGES
        if label.category != "cycle":
            return 0
        for key, info in CYCLE_STAGES.items():
            if key == label.sub_category:
                return info["stage"]
        return 0
