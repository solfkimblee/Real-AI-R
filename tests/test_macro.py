"""宏观分析模块单元测试"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from real_ai_r.macro.classifier import (
    CYCLE_STAGES,
    REDLINE_ZONES,
    TECH_TRACKS,
    MacroLabel,
    SectorClassifier,
)
from real_ai_r.macro.cycle_tracker import CycleTracker, StageStatus
from real_ai_r.macro.portfolio import AttackDefensePortfolio, PortfolioSlot
from real_ai_r.macro.red_filter import RedLineFilter
from real_ai_r.macro.tech_tracker import TechTracker, TrackSnapshot
from real_ai_r.macro.zeping_strategy import (
    ZepingBoardScore,
    ZepingMacroStrategy,
    ZepingParams,
    ZepingPredictionResult,
    ZepingWeights,
)

# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture()
def mock_board_df() -> pd.DataFrame:
    """模拟板块列表数据（已标准化列名）。"""
    return pd.DataFrame({
        "rank": list(range(1, 11)),
        "name": [
            "半导体", "人工智能", "机器人概念", "房地产开发",
            "白酒", "煤炭", "黄金", "农业", "食品饮料", "银行",
        ],
        "code": [f"BK{i:04d}" for i in range(10)],
        "price": [1500, 1200, 900, 400, 800, 600, 500, 300, 350, 550],
        "change_pct": [3.5, 2.8, 1.5, -2.0, -1.5, 0.8, 1.2, 0.5, -0.3, 0.1],
        "total_mv": [5e12, 4e12, 3e12, 2e12, 2.5e12, 3e12, 1e12, 8e11, 9e11, 6e12],
        "turnover_rate": [4.5, 3.8, 3.2, 1.5, 1.8, 2.0, 1.2, 1.0, 0.8, 0.6],
        "rise_count": [60, 45, 30, 5, 8, 15, 10, 12, 6, 3],
        "fall_count": [10, 15, 20, 35, 25, 10, 5, 8, 14, 17],
        "lead_stock": [
            "中芯国际", "科大讯飞", "拓普集团", "万科A",
            "贵州茅台", "中国神华", "山东黄金", "隆平高科",
            "伊利股份", "工商银行",
        ],
        "lead_stock_pct": [9.5, 7.0, 5.5, -3.0, 1.0, 2.5, 3.0, 1.5, -0.5, 0.3],
    })


@pytest.fixture()
def mock_fund_flow_df() -> pd.DataFrame:
    """模拟资金流向数据。"""
    return pd.DataFrame({
        "序号": list(range(1, 6)),
        "名称": ["半导体", "人工智能", "煤炭", "黄金", "农业"],
        "今日涨跌幅": [3.5, 2.8, 0.8, 1.2, 0.5],
        "今日主力净流入-净额": [8e8, 5e8, -1e8, 2e8, 1e8],
        "今日主力净流入-净占比": [5.0, 3.0, -1.0, 1.5, 0.8],
    })


@pytest.fixture()
def mock_cons_df() -> pd.DataFrame:
    """模拟成分股数据（已标准化）。"""
    return pd.DataFrame({
        "序号": [1, 2, 3, 4, 5],
        "代码": ["688981", "002049", "603501", "688396", "300661"],
        "名称": ["中芯国际", "紫光国微", "韦尔股份", "华峰测控", "圣邦股份"],
        "最新价": [55.0, 65.0, 90.0, 120.0, 150.0],
        "涨跌幅": [5.0, 3.5, 2.0, -1.0, 4.0],
        "涨跌额": [2.5, 2.0, 1.5, -1.2, 5.5],
        "成交量": [1e6, 5e5, 8e5, 3e5, 4e5],
        "成交额": [5.5e7, 3.2e7, 7.2e7, 3.6e7, 6.0e7],
        "振幅": [6.0, 4.5, 3.5, 2.0, 5.0],
        "最高": [56.0, 66.0, 91.0, 121.0, 152.0],
        "最低": [52.0, 63.0, 88.0, 118.0, 145.0],
        "今开": [53.0, 64.0, 89.0, 121.0, 147.0],
        "昨收": [52.5, 63.0, 88.5, 121.2, 144.5],
        "换手率": [2.5, 1.8, 2.0, 1.2, 3.0],
        "市盈率-动态": [25.0, 35.0, 45.0, 60.0, 80.0],
        "市净率": [3.0, 4.5, 6.0, 8.0, 10.0],
    })


# ===================================================================
# SectorClassifier 测试
# ===================================================================


class TestSectorClassifier:
    """板块分类器测试。"""

    def test_classify_tech_board(self) -> None:
        classifier = SectorClassifier()
        label = classifier.classify("半导体")
        assert label.category == "tech"
        assert label.sub_category == "chip"

    def test_classify_cycle_board(self) -> None:
        classifier = SectorClassifier()
        label = classifier.classify("黄金")
        assert label.category == "cycle"
        assert label.sub_category == "precious_metal"

    def test_classify_redline_board(self) -> None:
        classifier = SectorClassifier()
        label = classifier.classify("房地产开发")
        assert label.category == "redline"
        assert label.sub_category == "real_estate"

    def test_classify_neutral_board(self) -> None:
        classifier = SectorClassifier()
        label = classifier.classify("银行")
        assert label.category == "neutral"

    def test_classify_ai_board(self) -> None:
        classifier = SectorClassifier()
        label = classifier.classify("人工智能")
        assert label.category == "tech"
        assert label.sub_category == "ai_model"

    def test_classify_robot_board(self) -> None:
        classifier = SectorClassifier()
        label = classifier.classify("机器人概念")
        assert label.category == "tech"
        assert label.sub_category == "robot"

    def test_classify_liquor_board(self) -> None:
        classifier = SectorClassifier()
        label = classifier.classify("白酒")
        assert label.category == "redline"
        assert label.sub_category == "liquor"

    def test_classify_dataframe(self, mock_board_df: pd.DataFrame) -> None:
        classifier = SectorClassifier()
        result = classifier.classify_dataframe(mock_board_df)
        assert "macro_category" in result.columns
        assert "macro_display" in result.columns
        assert "macro_icon" in result.columns
        # 半导体 should be tech
        semi = result[result["name"] == "半导体"]
        assert semi.iloc[0]["macro_category"] == "tech"
        # 房地产 should be redline
        real_estate = result[result["name"] == "房地产开发"]
        assert real_estate.iloc[0]["macro_category"] == "redline"

    def test_get_category_summary(self, mock_board_df: pd.DataFrame) -> None:
        classifier = SectorClassifier()
        classified = classifier.classify_dataframe(mock_board_df)
        summary = classifier.get_category_summary(classified)
        assert "tech" in summary
        assert "cycle" in summary
        assert "redline" in summary
        assert summary["tech"]["count"] > 0
        assert summary["redline"]["count"] > 0

    def test_get_all_keywords(self) -> None:
        tech_kw = SectorClassifier.get_all_tech_keywords()
        cycle_kw = SectorClassifier.get_all_cycle_keywords()
        redline_kw = SectorClassifier.get_all_redline_keywords()
        assert len(tech_kw) > 10
        assert len(cycle_kw) > 5
        assert len(redline_kw) > 5
        assert "半导体" in tech_kw
        assert "黄金" in cycle_kw
        assert "房地产" in redline_kw

    def test_macro_label_frozen(self) -> None:
        label = MacroLabel(
            category="tech", sub_category="chip",
            display_name="芯片", description="test", icon="🔧",
        )
        assert label.category == "tech"
        # frozen dataclass should not allow modification
        with pytest.raises(AttributeError):
            label.category = "cycle"  # type: ignore[misc]

    def test_custom_tracks(self) -> None:
        custom = {
            "custom_track": {
                "display": "自定义赛道",
                "icon": "🔬",
                "desc": "测试赛道",
                "keywords": ["量子计算"],
            },
        }
        classifier = SectorClassifier(tech_tracks=custom)
        label = classifier.classify("量子计算")
        assert label.category == "tech"
        assert label.sub_category == "custom_track"


# ===================================================================
# CycleTracker 测试
# ===================================================================


class TestCycleTracker:
    """周期轮动追踪器测试。"""

    @patch("real_ai_r.macro.cycle_tracker.SectorMonitor")
    def test_track_basic(
        self,
        mock_monitor: MagicMock,
        mock_board_df: pd.DataFrame,
        mock_fund_flow_df: pd.DataFrame,
    ) -> None:
        mock_monitor.get_board_list.return_value = mock_board_df
        mock_monitor.get_fund_flow.return_value = mock_fund_flow_df

        tracker = CycleTracker()
        stages = tracker.track()

        assert len(stages) == 5
        assert all(isinstance(s, StageStatus) for s in stages)
        # 按阶段编号排序
        assert stages[0].stage == 1
        assert stages[4].stage == 5

    @patch("real_ai_r.macro.cycle_tracker.SectorMonitor")
    def test_get_current_stage(
        self,
        mock_monitor: MagicMock,
        mock_board_df: pd.DataFrame,
        mock_fund_flow_df: pd.DataFrame,
    ) -> None:
        mock_monitor.get_board_list.return_value = mock_board_df
        mock_monitor.get_fund_flow.return_value = mock_fund_flow_df

        tracker = CycleTracker()
        current = tracker.get_current_stage()
        assert current is not None
        assert 0 <= current.temperature <= 100

    @patch("real_ai_r.macro.cycle_tracker.SectorMonitor")
    def test_track_with_empty_fund_flow(
        self,
        mock_monitor: MagicMock,
        mock_board_df: pd.DataFrame,
    ) -> None:
        mock_monitor.get_board_list.return_value = mock_board_df
        mock_monitor.get_fund_flow.side_effect = Exception("网络错误")

        tracker = CycleTracker()
        stages = tracker.track()
        assert len(stages) == 5

    @patch("real_ai_r.macro.cycle_tracker.SectorMonitor")
    def test_track_all_fail(self, mock_monitor: MagicMock) -> None:
        mock_monitor.get_board_list.side_effect = Exception("全部失败")

        tracker = CycleTracker()
        stages = tracker.track()
        # 应返回空状态的5个阶段
        assert len(stages) == 5
        assert all(s.temperature == 0.0 for s in stages)

    def test_change_to_score(self) -> None:
        assert CycleTracker._change_to_score(5.0) == 100.0
        assert CycleTracker._change_to_score(0.0) == 50.0
        assert CycleTracker._change_to_score(-5.0) == 0.0
        assert CycleTracker._change_to_score(-10.0) == 0.0  # 下限
        assert CycleTracker._change_to_score(10.0) == 100.0  # 上限


# ===================================================================
# TechTracker 测试
# ===================================================================


class TestTechTracker:
    """科技赛道追踪器测试。"""

    @patch("real_ai_r.macro.tech_tracker.SectorMonitor")
    def test_track_all(
        self,
        mock_monitor: MagicMock,
        mock_board_df: pd.DataFrame,
    ) -> None:
        mock_monitor.get_board_list.return_value = mock_board_df

        tracker = TechTracker()
        tracks = tracker.track_all()

        assert len(tracks) == len(TECH_TRACKS)
        assert all(isinstance(t, TrackSnapshot) for t in tracks)
        # 按热度排序
        for i in range(len(tracks) - 1):
            assert tracks[i].heat_score >= tracks[i + 1].heat_score

    @patch("real_ai_r.macro.tech_tracker.SectorMonitor")
    def test_get_top_tracks(
        self,
        mock_monitor: MagicMock,
        mock_board_df: pd.DataFrame,
    ) -> None:
        mock_monitor.get_board_list.return_value = mock_board_df
        tracker = TechTracker()
        top3 = tracker.get_top_tracks(3)
        assert len(top3) <= 3

    @patch("real_ai_r.macro.tech_tracker.SectorMonitor")
    def test_get_track_comparison(
        self,
        mock_monitor: MagicMock,
        mock_board_df: pd.DataFrame,
    ) -> None:
        mock_monitor.get_board_list.return_value = mock_board_df
        tracker = TechTracker()
        comparison = tracker.get_track_comparison()
        assert isinstance(comparison, pd.DataFrame)
        assert "赛道" in comparison.columns
        assert "热度" in comparison.columns

    @patch("real_ai_r.macro.tech_tracker.SectorMonitor")
    def test_track_all_fail(self, mock_monitor: MagicMock) -> None:
        mock_monitor.get_board_list.side_effect = Exception("失败")
        tracker = TechTracker()
        tracks = tracker.track_all()
        assert len(tracks) == len(TECH_TRACKS)
        assert all(t.heat_score == 0.0 for t in tracks)


# ===================================================================
# RedLineFilter 测试
# ===================================================================


class TestRedLineFilter:
    """红线过滤器测试。"""

    def test_is_redline_real_estate(self) -> None:
        f = RedLineFilter()
        assert f.is_redline("房地产开发") is True

    def test_is_redline_liquor(self) -> None:
        f = RedLineFilter()
        assert f.is_redline("白酒") is True

    def test_is_not_redline(self) -> None:
        f = RedLineFilter()
        assert f.is_redline("半导体") is False
        assert f.is_redline("银行") is False

    def test_get_redline_reason(self) -> None:
        f = RedLineFilter()
        reason = f.get_redline_reason("房地产开发")
        assert reason is not None
        assert "房地产" in reason

        reason_none = f.get_redline_reason("半导体")
        assert reason_none is None

    def test_filter_boards(self, mock_board_df: pd.DataFrame) -> None:
        f = RedLineFilter()
        safe = f.filter_boards(mock_board_df)
        # 房地产和白酒应被过滤
        assert "房地产开发" not in safe["name"].values
        assert "白酒" not in safe["name"].values
        assert "半导体" in safe["name"].values

    def test_filter_boards_keep_redline(self, mock_board_df: pd.DataFrame) -> None:
        f = RedLineFilter()
        red = f.filter_boards(mock_board_df, keep_redline=True)
        assert len(red) > 0
        assert "房地产开发" in red["name"].values

    def test_filter_stocks_st(self) -> None:
        f = RedLineFilter()
        df = pd.DataFrame({
            "name": ["正常股", "ST问题股", "*ST退市", "好股票"],
            "price": [10, 5, 3, 20],
        })
        result = f.filter_stocks(df)
        assert len(result) == 2
        assert "正常股" in result["name"].values
        assert "好股票" in result["name"].values

    def test_get_redline_summary(self, mock_board_df: pd.DataFrame) -> None:
        f = RedLineFilter()
        summary = f.get_redline_summary(mock_board_df)
        assert len(summary) > 0
        names = [s["name"] for s in summary]
        assert "房地产开发" in names

    def test_get_zone_descriptions(self) -> None:
        zones = RedLineFilter.get_zone_descriptions()
        assert len(zones) == len(REDLINE_ZONES)
        assert all("name" in z for z in zones)
        assert all("description" in z for z in zones)


# ===================================================================
# AttackDefensePortfolio 测试
# ===================================================================


class TestAttackDefensePortfolio:
    """攻防组合测试。"""

    def test_init_default(self) -> None:
        builder = AttackDefensePortfolio()
        assert builder.attack_ratio == 0.6
        assert builder.defense_ratio == 0.4
        assert builder.n_attack == 5
        assert builder.n_defense == 4

    def test_init_custom(self) -> None:
        builder = AttackDefensePortfolio(
            attack_ratio=0.7,
            attack_slots=6,
            defense_slots=3,
        )
        assert builder.attack_ratio == 0.7
        assert builder.defense_ratio == pytest.approx(0.3)
        assert builder.n_attack == 6
        assert builder.n_defense == 3

    @patch("real_ai_r.macro.portfolio.StockRecommender")
    @patch("real_ai_r.macro.portfolio.CycleTracker")
    @patch("real_ai_r.macro.portfolio.TechTracker")
    def test_build_basic(
        self,
        mock_tech: MagicMock,
        mock_cycle: MagicMock,
        mock_recommender: MagicMock,
        mock_cons_df: pd.DataFrame,
    ) -> None:
        # Mock tech tracker
        mock_tech_instance = MagicMock()
        mock_tech.return_value = mock_tech_instance
        mock_tech_instance.track_all.return_value = [
            TrackSnapshot(
                key="chip", display="芯片", icon="🔧",
                description="test", avg_change_pct=3.0,
                total_rise=50, total_fall=10,
                avg_turnover=3.5, top_lead_stock="中芯国际",
                top_lead_pct=8.0, matched_boards=["半导体"],
                heat_score=80.0,
            ),
        ]

        # Mock cycle tracker
        mock_cycle_instance = MagicMock()
        mock_cycle.return_value = mock_cycle_instance
        mock_cycle_instance.track.return_value = [
            StageStatus(
                stage=4, key="agriculture", display="农业",
                icon="🌾", description="test",
                framework_status="重点布局区",
                temperature=40.0, avg_change_pct=0.5,
                avg_turnover=1.0, fund_flow_score=50.0,
                momentum_score=55.0, matched_boards=["农业"],
            ),
            StageStatus(
                stage=5, key="consumer", display="必选消费",
                icon="🛒", description="test",
                framework_status="底部左侧",
                temperature=30.0, avg_change_pct=-0.3,
                avg_turnover=0.8, fund_flow_score=45.0,
                momentum_score=47.0, matched_boards=["食品饮料"],
            ),
        ]

        # Mock recommender
        mock_recommender.recommend.return_value = pd.DataFrame({
            "code": ["688981", "002049"],
            "name": ["中芯国际", "紫光国微"],
            "price": [55.0, 65.0],
            "change_pct": [5.0, 3.5],
            "composite_score": [85.0, 72.0],
        })

        builder = AttackDefensePortfolio(
            attack_slots=2, defense_slots=2,
        )
        result = builder.build()

        assert result.attack_ratio == 0.6
        assert result.defense_ratio == 0.4

    def test_to_dataframe(self) -> None:
        from real_ai_r.macro.portfolio import PortfolioResult

        result = PortfolioResult(
            attack_slots=[
                PortfolioSlot(
                    role="attack", role_display="🗡️ 科技矛",
                    track="🔧 芯片", board_name="半导体",
                    stock_code="688981", stock_name="中芯国际",
                    price=55.0, change_pct=5.0, score=85.0,
                    reason="科技龙头",
                ),
            ],
            defense_slots=[
                PortfolioSlot(
                    role="defense", role_display="🛡️ 周期盾",
                    track="🌾 农业", board_name="农业",
                    stock_code="000998", stock_name="隆平高科",
                    price=15.0, change_pct=1.5, score=60.0,
                    reason="周期洼地",
                ),
            ],
            attack_ratio=0.6,
            defense_ratio=0.4,
            summary={
                "attack_count": 1, "defense_count": 1,
                "attack_ratio": 0.6, "defense_ratio": 0.4,
                "attack_tracks": ["🔧 芯片"],
                "defense_tracks": ["🌾 农业"],
            },
        )

        builder = AttackDefensePortfolio()
        df = builder.to_dataframe(result)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "角色" in df.columns
        assert "名称" in df.columns


# ===================================================================
# 常量完整性测试
# ===================================================================


class TestConstants:
    """配置常量完整性测试。"""

    def test_tech_tracks_structure(self) -> None:
        for key, track in TECH_TRACKS.items():
            assert "display" in track, f"{key} 缺少 display"
            assert "icon" in track, f"{key} 缺少 icon"
            assert "desc" in track, f"{key} 缺少 desc"
            assert "keywords" in track, f"{key} 缺少 keywords"
            assert len(track["keywords"]) > 0, f"{key} keywords 为空"

    def test_cycle_stages_structure(self) -> None:
        for key, stage in CYCLE_STAGES.items():
            assert "display" in stage, f"{key} 缺少 display"
            assert "stage" in stage, f"{key} 缺少 stage"
            assert "icon" in stage, f"{key} 缺少 icon"
            assert "desc" in stage, f"{key} 缺少 desc"
            assert "status" in stage, f"{key} 缺少 status"
            assert "keywords" in stage, f"{key} 缺少 keywords"
            assert 1 <= stage["stage"] <= 5

    def test_redline_zones_structure(self) -> None:
        for key, zone in REDLINE_ZONES.items():
            assert "display" in zone, f"{key} 缺少 display"
            assert "icon" in zone, f"{key} 缺少 icon"
            assert "desc" in zone, f"{key} 缺少 desc"
            assert "keywords" in zone, f"{key} 缺少 keywords"
            assert len(zone["keywords"]) > 0

    def test_cycle_stages_sequential(self) -> None:
        """验证五段论阶段编号连续 1-5。"""
        stage_nums = sorted(s["stage"] for s in CYCLE_STAGES.values())
        assert stage_nums == [1, 2, 3, 4, 5]


# ===================================================================
# ZepingMacroStrategy 测试
# ===================================================================


class TestZepingWeights:
    """泽平策略权重测试。"""

    def test_default_weights(self) -> None:
        w = ZepingWeights()
        assert w.macro == 0.40
        assert w.quant == 0.40
        assert w.cycle == 0.20
        assert w.macro + w.quant + w.cycle == pytest.approx(1.0)

    def test_custom_weights(self) -> None:
        w = ZepingWeights(macro=0.5, quant=0.3, cycle=0.2)
        assert w.macro == 0.5
        assert w.quant == 0.3

    def test_frozen(self) -> None:
        w = ZepingWeights()
        with pytest.raises(AttributeError):
            w.macro = 0.5  # type: ignore[misc]


class TestZepingParams:
    """泽平策略参数测试。"""

    def test_default_params(self) -> None:
        p = ZepingParams()
        assert p.tech_base_score == 70.0
        assert p.cycle_base_score == 55.0
        assert p.neutral_base_score == 40.0
        assert p.momentum_weight == 5.0

    def test_frozen(self) -> None:
        p = ZepingParams()
        with pytest.raises(AttributeError):
            p.tech_base_score = 99.0  # type: ignore[misc]


class TestZepingMacroStrategy:
    """泽平宏观策略核心测试。"""

    def test_init_default(self) -> None:
        strategy = ZepingMacroStrategy()
        assert strategy.weights.macro == 0.40
        assert strategy.weights.quant == 0.40
        assert strategy.weights.cycle == 0.20

    def test_init_custom_weights(self) -> None:
        w = ZepingWeights(macro=0.5, quant=0.3, cycle=0.2)
        strategy = ZepingMacroStrategy(weights=w)
        assert strategy.weights.macro == 0.5

    def test_predict_with_board_data(self, mock_board_df: pd.DataFrame) -> None:
        strategy = ZepingMacroStrategy()
        result = strategy.predict(board_df=mock_board_df, top_n=5)

        assert isinstance(result, ZepingPredictionResult)
        assert len(result.predictions) == 5
        assert result.total_boards > 0
        assert result.filtered_redline > 0  # 房地产+白酒被过滤
        assert result.market_style != ""
        assert result.strategy_summary != ""

    def test_redline_filtered(self, mock_board_df: pd.DataFrame) -> None:
        """红线禁区板块被正确排除。"""
        strategy = ZepingMacroStrategy()
        result = strategy.predict(board_df=mock_board_df, top_n=10)

        board_names = [s.board_name for s in result.predictions]
        assert "房地产开发" not in board_names
        assert "白酒" not in board_names

    def test_tech_boards_score_higher(self, mock_board_df: pd.DataFrame) -> None:
        """科技主线板块应获得更高的宏观分。"""
        strategy = ZepingMacroStrategy()
        result = strategy.predict(board_df=mock_board_df, top_n=10)

        tech_scores = [s for s in result.predictions if s.macro_category == "tech"]
        neutral_scores = [s for s in result.predictions if s.macro_category == "neutral"]

        if tech_scores and neutral_scores:
            avg_tech_macro = sum(s.macro_score for s in tech_scores) / len(tech_scores)
            avg_neutral_macro = sum(s.macro_score for s in neutral_scores) / len(neutral_scores)
            assert avg_tech_macro > avg_neutral_macro

    def test_scores_bounded(self, mock_board_df: pd.DataFrame) -> None:
        """所有子分应在 0-100 范围内。"""
        strategy = ZepingMacroStrategy()
        result = strategy.predict(board_df=mock_board_df, top_n=10)

        for s in result.predictions:
            assert 0 <= s.macro_score <= 100, f"{s.board_name} macro_score={s.macro_score}"
            assert 0 <= s.quant_score <= 100, f"{s.board_name} quant_score={s.quant_score}"
            assert 0 <= s.cycle_score <= 100, f"{s.board_name} cycle_score={s.cycle_score}"

    def test_sorted_by_total_score(self, mock_board_df: pd.DataFrame) -> None:
        """结果应按总分降序排列。"""
        strategy = ZepingMacroStrategy()
        result = strategy.predict(board_df=mock_board_df, top_n=10)

        for i in range(len(result.predictions) - 1):
            assert result.predictions[i].total_score >= result.predictions[i + 1].total_score

    def test_reasons_populated(self, mock_board_df: pd.DataFrame) -> None:
        """每个板块应有推荐理由。"""
        strategy = ZepingMacroStrategy()
        result = strategy.predict(board_df=mock_board_df, top_n=5)

        for s in result.predictions:
            assert len(s.reasons) > 0, f"{s.board_name} 无推荐理由"

    def test_to_dataframe(self, mock_board_df: pd.DataFrame) -> None:
        """DataFrame 输出包含必要列。"""
        strategy = ZepingMacroStrategy()
        result = strategy.predict(board_df=mock_board_df, top_n=5)
        df = strategy.to_dataframe(result.predictions)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "板块" in df.columns
        assert "总分" in df.columns
        assert "宏观分" in df.columns
        assert "量化分" in df.columns
        assert "周期分" in df.columns

    def test_score_snapshot(self, mock_board_df: pd.DataFrame) -> None:
        """score_snapshot 方法应返回正确结果。"""
        strategy = ZepingMacroStrategy()
        scores = strategy.score_snapshot(mock_board_df, top_n=3)

        assert len(scores) == 3
        assert all(isinstance(s, ZepingBoardScore) for s in scores)

    def test_empty_dataframe(self) -> None:
        """空 DataFrame 应返回空结果。"""
        strategy = ZepingMacroStrategy()
        result = strategy.predict(board_df=pd.DataFrame(), top_n=10)

        assert result.predictions == []
        assert result.total_boards == 0

    def test_cycle_stage_detection(self, mock_board_df: pd.DataFrame) -> None:
        """应正确检测最热周期阶段。"""
        strategy = ZepingMacroStrategy()
        result = strategy.predict(board_df=mock_board_df, top_n=5)

        # 黄金(+1.2%) > 煤炭(+0.8%) > 农业(+0.5%) > 食品饮料(-0.3%)
        # 所以最热阶段应该是贵金属 (stage 1)
        assert result.current_hot_stage >= 1

    def test_market_style_judgment(self, mock_board_df: pd.DataFrame) -> None:
        """市场风格判断非空。"""
        strategy = ZepingMacroStrategy()
        result = strategy.predict(board_df=mock_board_df, top_n=5)

        assert "风格" in result.market_style

    def test_shovel_seller_bonus(self) -> None:
        """'卖铲人'赛道应获得额外加分。"""
        strategy = ZepingMacroStrategy()

        # 芯片板块 = 卖铲人
        chip_df = pd.DataFrame({
            "name": ["半导体"],
            "code": ["BK0001"],
            "change_pct": [1.0],
            "turnover_rate": [3.0],
            "rise_count": [50],
            "fall_count": [10],
            "lead_stock_pct": [5.0],
        })
        # AI板块 = 非卖铲人
        ai_df = pd.DataFrame({
            "name": ["人工智能"],
            "code": ["BK0002"],
            "change_pct": [1.0],
            "turnover_rate": [3.0],
            "rise_count": [50],
            "fall_count": [10],
            "lead_stock_pct": [5.0],
        })

        chip_result = strategy.predict(board_df=chip_df, top_n=1)
        ai_result = strategy.predict(board_df=ai_df, top_n=1)

        if chip_result.predictions and ai_result.predictions:
            # 芯片的宏观分应高于AI（因为卖铲人加成）
            assert chip_result.predictions[0].macro_score >= ai_result.predictions[0].macro_score

    def test_predict_from_snapshot(self, mock_board_df: pd.DataFrame) -> None:
        """predict_from_snapshot 应与 predict 结果一致。"""
        strategy = ZepingMacroStrategy()
        result1 = strategy.predict(board_df=mock_board_df, top_n=5)
        result2 = strategy.predict_from_snapshot(mock_board_df, top_n=5)

        assert len(result1.predictions) == len(result2.predictions)
        for s1, s2 in zip(result1.predictions, result2.predictions):
            assert s1.board_name == s2.board_name
            assert s1.total_score == s2.total_score
