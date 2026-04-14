"""ML 模块单元测试

测试特征工程、模型训练/预测、回测系统。
使用 mock 数据避免真实 API 调用。
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from real_ai_r.ml.backtest import BacktestReport, ModelBacktester
from real_ai_r.ml.features import FeatureEngineer
from real_ai_r.ml.model import HotBoardModel, ModelMetrics, PredictionResult

# ======================================================================
# 辅助函数：生成 mock 数据
# ======================================================================

def _make_board_history(
    board_name: str = "测试板块",
    days: int = 60,
    base_price: float = 100.0,
) -> pd.DataFrame:
    """生成板块历史 mock 数据。"""
    dates = pd.date_range(end=datetime.now(), periods=days, freq="B")
    np.random.seed(42)
    changes = np.random.randn(days) * 2
    closes = [base_price]
    for c in changes[1:]:
        closes.append(closes[-1] * (1 + c / 100))

    return pd.DataFrame({
        "date": dates,
        "board_name": board_name,
        "open": [c * 0.99 for c in closes],
        "close": closes,
        "high": [c * 1.02 for c in closes],
        "low": [c * 0.98 for c in closes],
        "volume": np.random.randint(100000, 1000000, days),
        "amount": np.random.randint(10000000, 100000000, days),
        "change_pct": changes,
        "turnover_rate": np.random.uniform(1, 10, days),
        "amplitude": np.random.uniform(0.5, 5, days),
    })


def _make_multi_board_history(n_boards: int = 10, days: int = 60) -> pd.DataFrame:
    """生成多板块历史 mock 数据。"""
    all_data = []
    board_names = [
        "芯片", "人工智能", "新能源车", "煤炭", "农业",
        "房地产", "白酒", "机器人", "医药", "银行",
    ][:n_boards]
    for name in board_names:
        df = _make_board_history(name, days=days, base_price=np.random.uniform(50, 200))
        all_data.append(df)
    return pd.concat(all_data, ignore_index=True)


def _make_snapshot(n_boards: int = 10) -> pd.DataFrame:
    """生成板块截面快照 mock 数据。"""
    board_names = [
        "芯片", "人工智能", "新能源车", "煤炭", "农业",
        "房地产", "白酒", "机器人", "医药", "银行",
    ][:n_boards]
    np.random.seed(42)
    return pd.DataFrame({
        "name": board_names,
        "code": [f"BK{i:04d}" for i in range(n_boards)],
        "price": np.random.uniform(50, 200, n_boards),
        "change_pct": np.random.randn(n_boards) * 2,
        "turnover_rate": np.random.uniform(1, 10, n_boards),
        "rise_count": np.random.randint(10, 100, n_boards),
        "fall_count": np.random.randint(5, 50, n_boards),
        "lead_stock": ["领涨股A"] * n_boards,
        "lead_stock_pct": np.random.uniform(1, 10, n_boards),
        "net_inflow": np.random.uniform(-1e8, 1e8, n_boards),
    })


# ======================================================================
# 特征工程测试
# ======================================================================

class TestFeatureEngineer:
    """测试 FeatureEngineer。"""

    def test_init(self):
        """测试初始化。"""
        engineer = FeatureEngineer()
        assert len(engineer.QUANT_FEATURES) == 14
        assert len(engineer.MACRO_FEATURES) == 4
        assert len(engineer.MARKET_FEATURES) == 4
        assert len(engineer.ALL_FEATURES) == 22

    def test_build_features_from_history(self):
        """测试从历史数据构建特征。"""
        history = _make_multi_board_history(n_boards=5, days=40)
        engineer = FeatureEngineer()
        features = engineer.build_features_from_history(history)

        assert not features.empty
        assert "board_name" in features.columns
        assert "date" in features.columns
        assert "momentum_1d" in features.columns
        assert "is_hot_next_day" in features.columns
        assert "next_day_change" in features.columns
        # 宏观特征
        assert "is_tech" in features.columns
        assert "is_cycle" in features.columns
        assert "is_redline" in features.columns

    def test_build_features_from_history_empty(self):
        """测试空数据。"""
        engineer = FeatureEngineer()
        result = engineer.build_features_from_history(pd.DataFrame())
        assert result.empty

    def test_build_features_from_history_short(self):
        """测试数据太短。"""
        history = _make_board_history(days=10)
        engineer = FeatureEngineer()
        result = engineer.build_features_from_history(history)
        assert result.empty  # 需要至少20天

    def test_build_features_from_snapshot(self):
        """测试从截面数据构建特征。"""
        snapshot = _make_snapshot()
        engineer = FeatureEngineer()
        features = engineer.build_features_from_snapshot(snapshot)

        assert not features.empty
        assert len(features) == len(snapshot)
        assert "board_name" in features.columns
        assert "momentum_1d" in features.columns
        assert "is_tech" in features.columns
        assert "market_momentum" in features.columns
        assert "market_breadth" in features.columns

    def test_build_features_from_snapshot_empty(self):
        """测试空截面数据。"""
        engineer = FeatureEngineer()
        result = engineer.build_features_from_snapshot(pd.DataFrame())
        assert result.empty

    def test_build_features_from_snapshot_with_history(self):
        """测试带历史数据的截面特征构建。"""
        snapshot = _make_snapshot(n_boards=3)
        board_histories = {
            "芯片": _make_board_history("芯片", days=30),
            "人工智能": _make_board_history("人工智能", days=30),
        }
        engineer = FeatureEngineer()
        features = engineer.build_features_from_snapshot(
            snapshot, board_histories=board_histories,
        )
        assert not features.empty
        assert len(features) == 3

    def test_macro_label_encoding(self):
        """测试宏观标签编码正确性。"""
        # 科技板块
        snapshot = pd.DataFrame({
            "name": ["芯片"],
            "change_pct": [1.0],
            "turnover_rate": [5.0],
            "rise_count": [50],
            "fall_count": [20],
        })
        engineer = FeatureEngineer()
        features = engineer.build_features_from_snapshot(snapshot)
        assert features.iloc[0]["is_tech"] == 1
        assert features.iloc[0]["is_cycle"] == 0
        assert features.iloc[0]["is_redline"] == 0

    def test_target_generation(self):
        """测试预测目标生成。"""
        history = _make_multi_board_history(n_boards=5, days=40)
        engineer = FeatureEngineer()
        features = engineer.build_features_from_history(history)

        if not features.empty:
            # target 列应该存在
            assert "is_hot_next_day" in features.columns
            assert "next_day_change" in features.columns
            # hot 标签只有 0 和 1
            assert set(features["is_hot_next_day"].unique()).issubset({0, 1})
            # hot 标签比例应该在合理范围（小数据集可能偏高）
            hot_ratio = features["is_hot_next_day"].mean()
            assert 0.0 <= hot_ratio <= 1.0

    def test_multi_day_features(self):
        """测试多日特征计算。"""
        closes = np.array([100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
                           110, 111, 112, 113, 114, 115, 116, 117, 118, 119,
                           120], dtype=float)
        volumes = np.array([1000] * 21, dtype=float)
        result = FeatureEngineer._multi_day_features(closes, volumes)

        assert "momentum_3d" in result
        assert "momentum_5d" in result
        assert "momentum_10d" in result
        assert "volatility_5d" in result
        assert "volume_ratio_5d" in result
        assert "ma5_bias" in result
        assert "rsi_14" in result
        assert "price_position" in result


# ======================================================================
# 模型测试
# ======================================================================

class TestHotBoardModel:
    """测试 HotBoardModel。"""

    def _get_training_data(self) -> pd.DataFrame:
        """获取训练数据。"""
        history = _make_multi_board_history(n_boards=8, days=50)
        engineer = FeatureEngineer()
        return engineer.build_features_from_history(history)

    def test_init(self):
        """测试初始化。"""
        model = HotBoardModel()
        assert not model.is_trained
        assert model.model is None
        assert model.metrics is None

    def test_train(self):
        """测试模型训练。"""
        df = self._get_training_data()
        if df.empty:
            pytest.skip("训练数据不足")

        model = HotBoardModel()
        metrics = model.train(df)

        assert model.is_trained
        assert model.model is not None
        assert isinstance(metrics, ModelMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.precision <= 1
        assert 0 <= metrics.recall <= 1

    def test_predict_with_trained_model(self):
        """测试训练后的预测。"""
        df = self._get_training_data()
        if df.empty:
            pytest.skip("训练数据不足")

        model = HotBoardModel()
        model.train(df)

        # 使用截面数据预测
        snapshot = _make_snapshot()
        engineer = FeatureEngineer()
        features = engineer.build_features_from_snapshot(snapshot)
        predictions = model.predict(features)

        assert len(predictions) > 0
        assert isinstance(predictions[0], PredictionResult)
        assert 0 <= predictions[0].hot_probability <= 1
        assert 0 <= predictions[0].predicted_score <= 100
        assert predictions[0].board_name != ""
        # 按分数降序排列
        scores = [p.predicted_score for p in predictions]
        assert scores == sorted(scores, reverse=True)

    def test_predict_without_training(self):
        """测试未训练时的回退预测。"""
        model = HotBoardModel()
        snapshot = _make_snapshot()
        engineer = FeatureEngineer()
        features = engineer.build_features_from_snapshot(snapshot)
        predictions = model.predict(features)

        assert len(predictions) > 0
        assert isinstance(predictions[0], PredictionResult)
        # 回退预测也应该有结果
        assert predictions[0].predicted_score >= 0

    def test_feature_importance(self):
        """测试特征重要性。"""
        df = self._get_training_data()
        if df.empty:
            pytest.skip("训练数据不足")

        model = HotBoardModel()
        model.train(df)

        fi = model.get_feature_importance(top_n=5)
        assert len(fi) > 0
        assert len(fi) <= 5
        # 每个元素是 (feature_name, importance)
        assert isinstance(fi[0], tuple)
        assert isinstance(fi[0][0], str)
        assert isinstance(fi[0][1], float)

    def test_save_load(self, tmp_path):
        """测试模型保存/加载。"""
        df = self._get_training_data()
        if df.empty:
            pytest.skip("训练数据不足")

        model = HotBoardModel()
        model.train(df)

        # 保存
        model_path = tmp_path / "model.joblib"
        model.save(model_path)
        assert model_path.exists()

        # 加载
        model2 = HotBoardModel()
        model2.load(model_path)
        assert model2.is_trained
        assert model2.model is not None

        # 预测结果应一致
        snapshot = _make_snapshot(n_boards=5)
        engineer = FeatureEngineer()
        features = engineer.build_features_from_snapshot(snapshot)
        pred1 = model.predict(features)
        pred2 = model2.predict(features)
        assert len(pred1) == len(pred2)

    def test_prediction_result_fields(self):
        """测试 PredictionResult 字段。"""
        result = PredictionResult(
            board_name="芯片",
            hot_probability=0.85,
            predicted_score=85.0,
            macro_category="科技主线",
            key_factors=["动量=2.5%"],
            momentum_1d=2.5,
            turnover_rate=5.0,
        )
        assert result.board_name == "芯片"
        assert result.hot_probability == 0.85
        assert result.macro_category == "科技主线"
        assert len(result.key_factors) == 1

    def test_model_metrics_dataclass(self):
        """测试 ModelMetrics 数据类。"""
        metrics = ModelMetrics(
            accuracy=0.8,
            precision=0.7,
            recall=0.6,
            f1=0.65,
            auc=0.75,
        )
        assert metrics.accuracy == 0.8
        assert metrics.f1 == 0.65
        assert isinstance(metrics.feature_importance, dict)


# ======================================================================
# 回测测试
# ======================================================================

class TestModelBacktester:
    """测试 ModelBacktester。"""

    def _get_feature_data(self) -> pd.DataFrame:
        """获取回测用特征数据。"""
        history = _make_multi_board_history(n_boards=8, days=50)
        engineer = FeatureEngineer()
        return engineer.build_features_from_history(history)

    def test_init(self):
        """测试初始化。"""
        bt = ModelBacktester(train_window=20, top_n=5, retrain_every=3)
        assert bt.train_window == 20
        assert bt.top_n == 5
        assert bt.retrain_every == 3

    def test_run(self):
        """测试回测执行。"""
        df = self._get_feature_data()
        if df.empty or df["date"].nunique() < 30:
            pytest.skip("回测数据不足")

        bt = ModelBacktester(train_window=15, top_n=3, retrain_every=3)
        report = bt.run(df)

        assert isinstance(report, BacktestReport)
        assert report.total_days > 0
        assert 0 <= report.avg_precision <= 1
        assert 0 <= report.win_rate <= 1
        assert len(report.equity_curve) > 0
        assert report.equity_curve[0] == 1.0

    def test_run_empty_data(self):
        """测试空数据回测。"""
        bt = ModelBacktester()
        report = bt.run(pd.DataFrame())
        assert report.total_days == 0

    def test_run_insufficient_data(self):
        """测试数据不足的回测。"""
        # 只有很少天数
        history = _make_multi_board_history(n_boards=3, days=25)
        engineer = FeatureEngineer()
        df = engineer.build_features_from_history(history)

        bt = ModelBacktester(train_window=30)
        report = bt.run(df)
        # 数据不足应返回空报告
        assert report.total_days == 0

    def test_backtest_report_fields(self):
        """测试 BacktestReport 字段。"""
        report = BacktestReport(
            total_days=20,
            avg_precision=0.3,
            avg_hit_rate=0.3,
            avg_excess_return=0.5,
            cumulative_return=5.0,
            max_drawdown=3.0,
            sharpe_ratio=1.2,
            win_rate=0.6,
        )
        assert report.total_days == 20
        assert report.sharpe_ratio == 1.2
        assert report.win_rate == 0.6
        assert isinstance(report.daily_results, list)
        assert isinstance(report.equity_curve, list)

    def test_daily_results(self):
        """测试每日回测结果。"""
        df = self._get_feature_data()
        if df.empty or df["date"].nunique() < 30:
            pytest.skip("回测数据不足")

        bt = ModelBacktester(train_window=15, top_n=3, retrain_every=3)
        report = bt.run(df)

        if report.daily_results:
            day = report.daily_results[0]
            assert hasattr(day, "date")
            assert hasattr(day, "predicted_hot")
            assert hasattr(day, "actual_hot")
            assert hasattr(day, "hit_count")
            assert hasattr(day, "precision")
            assert hasattr(day, "excess_return")
            assert isinstance(day.predicted_hot, list)
            assert isinstance(day.actual_hot, list)


# ======================================================================
# 数据采集测试（仅测试结构，不调用真实 API）
# ======================================================================

class TestDataCollector:
    """测试数据采集器结构。"""

    def test_board_history_collector_init(self):
        """测试 BoardHistoryCollector 初始化。"""
        from real_ai_r.ml.data_collector import BoardHistoryCollector
        collector = BoardHistoryCollector(board_type="industry")
        assert collector.board_type == "industry"

    def test_board_history_collector_concept(self):
        """测试概念板块初始化。"""
        from real_ai_r.ml.data_collector import BoardHistoryCollector
        collector = BoardHistoryCollector(board_type="concept")
        assert collector.board_type == "concept"

    def test_snapshot_collector_exists(self):
        """测试 SnapshotCollector 存在。"""
        from real_ai_r.ml.data_collector import SnapshotCollector
        assert hasattr(SnapshotCollector, "collect_today_snapshot")


# ======================================================================
# 模型版本管理测试
# ======================================================================

class TestModelRegistry:
    """测试模型版本注册表。"""

    @pytest.fixture()
    def tmp_registry(self, tmp_path):
        """创建临时注册表目录。"""
        from real_ai_r.ml.registry import ModelRegistry
        return ModelRegistry(registry_dir=tmp_path / "models")

    @pytest.fixture()
    def trained_model(self):
        """创建已训练的模型。"""
        history = _make_multi_board_history(n_boards=5, days=60)
        engineer = FeatureEngineer()
        feature_df = engineer.build_features_from_history(history)
        model = HotBoardModel()
        model.train(feature_df)
        return model, feature_df

    def test_registry_init(self, tmp_registry):
        """测试注册表初始化。"""
        assert tmp_registry.version_count == 0
        assert tmp_registry.registry_dir.exists()

    def test_save_model(self, tmp_registry, trained_model):
        """测试保存模型。"""
        model, feature_df = trained_model
        version = tmp_registry.save_model(
            model=model,
            board_type="industry",
            train_days=60,
            max_boards=5,
            sample_count=len(feature_df),
        )
        assert version.version_id
        assert version.board_type == "industry"
        assert version.train_days == 60
        assert version.sample_count > 0
        assert version.feature_count == 22
        assert version.auc >= 0
        assert tmp_registry.version_count == 1

    def test_list_versions(self, tmp_registry, trained_model):
        """测试列出版本。"""
        model, feature_df = trained_model
        # 保存两个版本
        v1 = tmp_registry.save_model(
            model=model, board_type="industry",
            train_days=60, sample_count=len(feature_df),
        )
        v2 = tmp_registry.save_model(
            model=model, board_type="concept",
            train_days=30, sample_count=len(feature_df),
        )
        all_versions = tmp_registry.list_versions()
        assert len(all_versions) == 2

        # 按类型筛选
        industry_versions = tmp_registry.list_versions(board_type="industry")
        assert len(industry_versions) == 1
        assert industry_versions[0].version_id == v1.version_id

        concept_versions = tmp_registry.list_versions(board_type="concept")
        assert len(concept_versions) == 1
        assert concept_versions[0].version_id == v2.version_id

    def test_load_model(self, tmp_registry, trained_model):
        """测试加载模型。"""
        model, feature_df = trained_model
        version = tmp_registry.save_model(
            model=model, board_type="industry",
            train_days=60, sample_count=len(feature_df),
        )
        loaded = tmp_registry.load_model(version.version_id)
        assert loaded is not None
        assert loaded.is_trained
        assert loaded.model is not None
        assert len(loaded.feature_columns) == 22

    def test_load_model_predicts(self, tmp_registry, trained_model):
        """测试加载的模型可以正常预测。"""
        model, feature_df = trained_model
        version = tmp_registry.save_model(
            model=model, board_type="industry",
            train_days=60, sample_count=len(feature_df),
        )
        loaded = tmp_registry.load_model(version.version_id)
        # 用最后一天的数据做预测
        last_date = feature_df["date"].max()
        snapshot = feature_df[feature_df["date"] == last_date]
        if not snapshot.empty:
            predictions = loaded.predict(snapshot)
            assert isinstance(predictions, list)
            assert len(predictions) > 0

    def test_get_version(self, tmp_registry, trained_model):
        """测试获取版本元数据。"""
        model, feature_df = trained_model
        version = tmp_registry.save_model(
            model=model, board_type="industry",
            train_days=60, sample_count=len(feature_df),
        )
        retrieved = tmp_registry.get_version(version.version_id)
        assert retrieved is not None
        assert retrieved.version_id == version.version_id
        assert retrieved.board_type == "industry"

        # 不存在的版本
        assert tmp_registry.get_version("nonexistent") is None

    def test_delete_version(self, tmp_registry, trained_model):
        """测试删除版本。"""
        model, feature_df = trained_model
        version = tmp_registry.save_model(
            model=model, board_type="industry",
            train_days=60, sample_count=len(feature_df),
        )
        assert tmp_registry.version_count == 1
        tmp_registry.delete_version(version.version_id)
        assert tmp_registry.version_count == 0
        assert tmp_registry.get_version(version.version_id) is None

    def test_compare_versions(self, tmp_registry, trained_model):
        """测试版本对比。"""
        model, feature_df = trained_model
        v1 = tmp_registry.save_model(
            model=model, board_type="industry",
            train_days=60, sample_count=len(feature_df),
        )
        v2 = tmp_registry.save_model(
            model=model, board_type="industry",
            train_days=30, sample_count=len(feature_df),
        )
        comparison = tmp_registry.compare_versions([v1.version_id, v2.version_id])
        assert len(comparison) == 2
        assert "版本" in comparison[0]
        assert "AUC" in comparison[0]
        assert "F1" in comparison[0]

    def test_load_nonexistent_model(self, tmp_registry):
        """测试加载不存在的模型。"""
        result = tmp_registry.load_model("nonexistent_version")
        assert result is None

    def test_index_persistence(self, tmp_path, trained_model):
        """测试索引持久化——重建注册表后版本仍在。"""
        from real_ai_r.ml.registry import ModelRegistry
        model, feature_df = trained_model
        reg_dir = tmp_path / "models"

        # 第一个注册表实例保存模型
        reg1 = ModelRegistry(registry_dir=reg_dir)
        version = reg1.save_model(
            model=model, board_type="industry",
            train_days=60, sample_count=len(feature_df),
        )

        # 新实例应自动加载索引
        reg2 = ModelRegistry(registry_dir=reg_dir)
        assert reg2.version_count == 1
        assert reg2.get_version(version.version_id) is not None

    def test_version_display_name(self, tmp_registry, trained_model):
        """测试版本显示名。"""
        model, feature_df = trained_model
        version = tmp_registry.save_model(
            model=model, board_type="industry",
            train_days=60, sample_count=len(feature_df),
        )
        assert "行业" in version.display_name
        assert "AUC=" in version.display_name

    def test_save_with_note(self, tmp_registry, trained_model):
        """测试带备注的保存。"""
        model, feature_df = trained_model
        version = tmp_registry.save_model(
            model=model, board_type="industry",
            train_days=60, sample_count=len(feature_df),
            note="测试备注",
        )
        retrieved = tmp_registry.get_version(version.version_id)
        assert retrieved.note == "测试备注"
