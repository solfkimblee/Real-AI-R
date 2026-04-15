"""板块分析模块单元测试"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from real_ai_r.sector.monitor import SectorMonitor
from real_ai_r.sector.predictor import HotSectorPredictor
from real_ai_r.sector.recommender import StockRecommender, _normalize

# ===================================================================
# Fixtures — 模拟数据
# ===================================================================

@pytest.fixture()
def mock_board_df() -> pd.DataFrame:
    """模拟板块列表数据。"""
    return pd.DataFrame({
        "排名": [1, 2, 3, 4, 5],
        "板块名称": ["半导体", "新能源", "白酒", "银行", "医药"],
        "板块代码": ["BK1036", "BK1015", "BK0477", "BK0475", "BK0465"],
        "最新价": [1500.0, 1200.0, 800.0, 600.0, 700.0],
        "涨跌额": [30.0, 20.0, -10.0, -5.0, 0.0],
        "涨跌幅": [2.0, 1.5, -1.2, -0.8, 0.0],
        "总市值": [5e12, 4e12, 3e12, 2e12, 2.5e12],
        "换手率": [3.5, 4.2, 1.8, 0.9, 2.1],
        "上涨家数": [50, 40, 10, 5, 20],
        "下跌家数": [10, 15, 30, 25, 20],
        "领涨股票": ["中芯国际", "宁德时代", "贵州茅台", "工商银行", "恒瑞医药"],
        "领涨股票-涨跌幅": [8.5, 6.0, 2.0, 1.0, 3.0],
    })


@pytest.fixture()
def mock_fund_flow_df() -> pd.DataFrame:
    """模拟资金流向数据。"""
    return pd.DataFrame({
        "序号": [1, 2, 3, 4, 5],
        "名称": ["半导体", "新能源", "白酒", "银行", "医药"],
        "今日涨跌幅": [2.0, 1.5, -1.2, -0.8, 0.0],
        "今日主力净流入-净额": [5e8, 3e8, -2e8, -1e8, 1e8],
        "今日主力净流入-净占比": [5.0, 3.0, -2.0, -1.0, 1.0],
        "今日超大单净流入-净额": [3e8, 2e8, -1e8, -5e7, 5e7],
        "今日超大单净流入-净占比": [3.0, 2.0, -1.0, -0.5, 0.5],
        "今日大单净流入-净额": [2e8, 1e8, -1e8, -5e7, 5e7],
        "今日大单净流入-净占比": [2.0, 1.0, -1.0, -0.5, 0.5],
        "今日中单净流入-净额": [-1e8, -5e7, 5e7, 3e7, -3e7],
        "今日中单净流入-净占比": [-1.0, -0.5, 0.5, 0.3, -0.3],
        "今日小单净流入-净额": [-1e8, -5e7, 5e7, 2e7, -2e7],
        "今日小单净流入-净占比": [-1.0, -0.5, 0.5, 0.2, -0.2],
        "今日主力净流入最大股": ["中芯国际", "宁德时代", "贵州茅台", "工商银行", "恒瑞医药"],
    })


@pytest.fixture()
def mock_cons_df() -> pd.DataFrame:
    """模拟板块成分股数据。"""
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
# SectorMonitor 测试
# ===================================================================

class TestSectorMonitor:
    """板块监控器测试。"""

    @patch("real_ai_r.sector.monitor.ak")
    def test_get_board_list_industry(self, mock_ak: MagicMock, mock_board_df: pd.DataFrame) -> None:
        mock_ak.stock_board_industry_name_em.return_value = mock_board_df
        result = SectorMonitor.get_board_list("industry")
        assert len(result) == 5
        assert "name" in result.columns
        assert "change_pct" in result.columns
        mock_ak.stock_board_industry_name_em.assert_called_once()

    @patch("real_ai_r.sector.monitor.ak")
    def test_get_board_list_concept(self, mock_ak: MagicMock, mock_board_df: pd.DataFrame) -> None:
        mock_ak.stock_board_concept_name_em.return_value = mock_board_df
        result = SectorMonitor.get_board_list("concept")
        assert len(result) == 5
        mock_ak.stock_board_concept_name_em.assert_called_once()

    @patch("real_ai_r.sector.monitor.ak")
    def test_get_fund_flow(self, mock_ak: MagicMock, mock_fund_flow_df: pd.DataFrame) -> None:
        mock_ak.stock_sector_fund_flow_rank.return_value = mock_fund_flow_df
        result = SectorMonitor.get_fund_flow("今日", "行业资金流")
        assert len(result) == 5
        mock_ak.stock_sector_fund_flow_rank.assert_called_once_with(
            indicator="今日", sector_type="行业资金流",
        )

    @patch("real_ai_r.sector.monitor.ak")
    def test_get_board_stats(self, mock_ak: MagicMock, mock_board_df: pd.DataFrame) -> None:
        mock_ak.stock_board_industry_name_em.return_value = mock_board_df
        stats = SectorMonitor.get_board_stats("industry")
        assert stats["rise_count"] == 2
        assert stats["fall_count"] == 2
        assert stats["flat_count"] == 1
        assert stats["total_count"] == 5
        assert len(stats["top5"]) == 5
        assert len(stats["bottom5"]) == 5


# ===================================================================
# HotSectorPredictor 测试
# ===================================================================

class TestHotSectorPredictor:
    """热门板块预测器测试。"""

    @patch("real_ai_r.sector.predictor.SectorMonitor")
    def test_predict_basic(
        self,
        mock_monitor: MagicMock,
        mock_board_df: pd.DataFrame,
        mock_fund_flow_df: pd.DataFrame,
    ) -> None:
        # 构造标准化后的板块数据
        board_std = mock_board_df.rename(columns={
            "排名": "rank", "板块名称": "name", "板块代码": "code",
            "最新价": "price", "涨跌额": "change_amount",
            "涨跌幅": "change_pct", "总市值": "total_mv",
            "换手率": "turnover_rate", "上涨家数": "rise_count",
            "下跌家数": "fall_count", "领涨股票": "lead_stock",
            "领涨股票-涨跌幅": "lead_stock_pct",
        })
        mock_monitor.get_board_list.return_value = board_std
        mock_monitor.get_fund_flow.return_value = mock_fund_flow_df

        predictor = HotSectorPredictor(board_type="industry")
        result = predictor.predict(top_n=3)

        assert len(result) == 3
        assert "name" in result.columns
        assert "total_score" in result.columns
        # 评分应该在 0-100 之间
        assert result["total_score"].min() >= 0
        assert result["total_score"].max() <= 100

    @patch("real_ai_r.sector.predictor.SectorMonitor")
    def test_predict_with_empty_fund_flow(
        self,
        mock_monitor: MagicMock,
        mock_board_df: pd.DataFrame,
    ) -> None:
        board_std = mock_board_df.rename(columns={
            "排名": "rank", "板块名称": "name", "板块代码": "code",
            "最新价": "price", "涨跌额": "change_amount",
            "涨跌幅": "change_pct", "总市值": "total_mv",
            "换手率": "turnover_rate", "上涨家数": "rise_count",
            "下跌家数": "fall_count", "领涨股票": "lead_stock",
            "领涨股票-涨跌幅": "lead_stock_pct",
        })
        mock_monitor.get_board_list.return_value = board_std
        mock_monitor.get_fund_flow.side_effect = Exception("网络错误")

        predictor = HotSectorPredictor(board_type="industry")
        result = predictor.predict(top_n=3)
        # 即使资金流向失败也能返回结果
        assert len(result) == 3

    def test_custom_weights(self) -> None:
        weights = {
            "fund_flow": 0.5,
            "momentum": 0.2,
            "activity": 0.1,
            "breadth": 0.1,
            "lead_strength": 0.1,
        }
        predictor = HotSectorPredictor(weights=weights)
        assert predictor.weights["fund_flow"] == 0.5


# ===================================================================
# StockRecommender 测试
# ===================================================================

class TestStockRecommender:
    """个股推荐器测试。"""

    @patch("real_ai_r.sector.recommender.ak")
    def test_get_board_stocks(self, mock_ak: MagicMock, mock_cons_df: pd.DataFrame) -> None:
        mock_ak.stock_board_industry_cons_em.return_value = mock_cons_df
        result = StockRecommender.get_board_stocks("半导体", "industry")
        assert len(result) == 5
        assert "code" in result.columns
        assert "name" in result.columns

    @patch("real_ai_r.sector.recommender.ak")
    def test_recommend_composite(self, mock_ak: MagicMock, mock_cons_df: pd.DataFrame) -> None:
        mock_ak.stock_board_industry_cons_em.return_value = mock_cons_df
        result = StockRecommender.recommend("半导体", "industry", top_n=3)
        assert len(result) == 3
        assert "composite_score" in result.columns

    @patch("real_ai_r.sector.recommender.ak")
    def test_recommend_by_change_pct(self, mock_ak: MagicMock, mock_cons_df: pd.DataFrame) -> None:
        mock_ak.stock_board_industry_cons_em.return_value = mock_cons_df
        result = StockRecommender.recommend(
            "半导体", "industry", top_n=3, sort_by="change_pct",
        )
        assert len(result) == 3
        # 应按涨幅降序
        changes = result["change_pct"].tolist()
        assert changes == sorted(changes, reverse=True)

    @patch("real_ai_r.sector.recommender.ak")
    def test_recommend_filters_st(self, mock_ak: MagicMock) -> None:
        """ST 股应被过滤。"""
        df = pd.DataFrame({
            "序号": [1, 2, 3],
            "代码": ["000001", "000002", "000003"],
            "名称": ["正常股", "ST退市股", "*ST问题股"],
            "最新价": [10.0, 5.0, 3.0],
            "涨跌幅": [2.0, 1.0, -1.0],
            "涨跌额": [0.2, 0.05, -0.03],
            "成交量": [1e6, 5e5, 3e5],
            "成交额": [1e7, 2.5e6, 9e5],
            "振幅": [3.0, 2.0, 1.0],
            "最高": [10.5, 5.1, 3.1],
            "最低": [9.5, 4.9, 2.9],
            "今开": [10.0, 5.0, 3.0],
            "昨收": [9.8, 4.95, 3.03],
            "换手率": [2.0, 1.0, 0.5],
            "市盈率-动态": [20.0, -5.0, 30.0],
            "市净率": [2.0, 1.0, 0.5],
        })
        mock_ak.stock_board_industry_cons_em.return_value = df
        result = StockRecommender.recommend("测试板块", "industry", top_n=5)
        assert len(result) == 1
        assert result.iloc[0]["name"] == "正常股"

    @patch("real_ai_r.sector.recommender.ak")
    def test_recommend_concept(self, mock_ak: MagicMock, mock_cons_df: pd.DataFrame) -> None:
        mock_ak.stock_board_concept_cons_em.return_value = mock_cons_df
        result = StockRecommender.recommend("人工智能", "concept", top_n=3)
        assert len(result) == 3


# ===================================================================
# 辅助函数测试
# ===================================================================

class TestNormalize:
    """标准化函数测试。"""

    def test_normalize_basic(self) -> None:
        s = pd.Series([0, 50, 100])
        result = _normalize(s)
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 50.0
        assert result.iloc[2] == 100.0

    def test_normalize_all_same(self) -> None:
        s = pd.Series([5, 5, 5])
        result = _normalize(s)
        assert (result == 50.0).all()

    def test_normalize_negative(self) -> None:
        s = pd.Series([-10, 0, 10])
        result = _normalize(s)
        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 50.0
        assert result.iloc[2] == 100.0
