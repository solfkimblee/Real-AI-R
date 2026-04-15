"""板块分析模块 — 板块监控、热门预测、个股推荐"""

from real_ai_r.sector.monitor import SectorMonitor
from real_ai_r.sector.predictor import HotSectorPredictor
from real_ai_r.sector.recommender import StockRecommender

__all__ = ["SectorMonitor", "HotSectorPredictor", "StockRecommender"]
